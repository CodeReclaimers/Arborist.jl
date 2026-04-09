# Configurable prompt enrichment for LLM mutation operators.
#
# The solve loop populates a MutationContext each generation; AbstractPromptSection
# subtypes render slices of that context into text that is prepended to the
# serialized genome in the LLM request.

# =============================================================================
# MutationContext — per-generation + per-call state from the solve loop
# =============================================================================

"""
    MutationContext

Mutable container for population state that the solve loop fills in each
generation. LLM mutation operators read this to build enriched prompts.

Population-level fields (`generation`, `max_generations`, `fitnesses`,
`genomes_serialized`) are set once per generation by `_update_llm_contexts!`.
Per-call fields (`parent_fitness`, `parent_rank`) are set inside the breeding
loop by `_set_parent_context!`, after tournament selection but before `mutate`.
"""
mutable struct MutationContext
    generation::Int
    max_generations::Int
    fitnesses::Vector{Float64}           # sorted ascending (best first)
    genomes_serialized::Vector{String}   # top-K only, same order as fitnesses
    parent_fitness::Float64              # fitness of the parent being mutated
    parent_rank::Int                     # 1-based rank in sorted population
end

MutationContext() = MutationContext(0, 0, Float64[], String[], Inf, 0)


# =============================================================================
# AbstractPromptSection — composable prompt building blocks
# =============================================================================

"""
    AbstractPromptSection

Base type for prompt enrichment sections. Each subtype implements
`render(section, context) -> String` that produces a text block (or `""`
to skip) given the current `MutationContext`.

Users compose a `Vector{AbstractPromptSection}` on the `LLMMutationOperator`
to control what context the LLM sees.
"""
abstract type AbstractPromptSection end

"""
    render(section::AbstractPromptSection, ctx::MutationContext) -> String

Render this section into a text block for the LLM prompt. Return `""`
if the context does not contain enough information (graceful degradation).
"""
function render(::AbstractPromptSection, ::MutationContext)::String
    return ""
end


# =============================================================================
# Concrete sections
# =============================================================================

"""
    FitnessSection()

Shows the parent genome's fitness and rank, plus population best and mean.
Gives the LLM a sense of how good the current program is and what the
target looks like.
"""
struct FitnessSection <: AbstractPromptSection end

function render(::FitnessSection, ctx::MutationContext)::String
    isempty(ctx.fitnesses) && return ""
    best = ctx.fitnesses[1]
    finite_fits = filter(isfinite, ctx.fitnesses)
    mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
    return string(
        "--- Fitness Context ---\n",
        "This program's fitness: ", round(ctx.parent_fitness, digits=6),
        " (rank ", ctx.parent_rank, "/", length(ctx.fitnesses), ")\n",
        "Population best: ", round(best, digits=6), "\n",
        "Population mean: ", round(mean_fit, digits=6), "\n",
        "(Lower fitness is better.)")
end


"""
    ElitesSection(k=3)

Shows the top-K programs from the current population with their fitnesses.
This is the core FunSearch/AlphaEvolve pattern: giving the LLM scored
examples of what works well so it can learn from them.
"""
struct ElitesSection <: AbstractPromptSection
    k::Int
end
ElitesSection() = ElitesSection(3)

function render(s::ElitesSection, ctx::MutationContext)::String
    isempty(ctx.genomes_serialized) && return ""
    k = min(s.k, length(ctx.genomes_serialized))
    lines = String["--- Top $k Programs ---"]
    for i in 1:k
        push!(lines, "# Program $i (fitness=$(round(ctx.fitnesses[i], digits=6))):")
        push!(lines, ctx.genomes_serialized[i])
        push!(lines, "")
    end
    return join(lines, "\n")
end


"""
    GenerationSection()

Shows the current generation number and a phase label (early/mid/late).
Allows the LLM to calibrate mutation aggressiveness: explore early,
exploit late.
"""
struct GenerationSection <: AbstractPromptSection end

function render(::GenerationSection, ctx::MutationContext)::String
    ctx.max_generations == 0 && return ""
    progress = ctx.generation / ctx.max_generations
    phase = progress < 0.33 ? "early (exploration)" :
            progress < 0.67 ? "mid (exploitation)" :
                              "late (refinement)"
    return string(
        "--- Generation Progress ---\n",
        "Generation ", ctx.generation, "/", ctx.max_generations, " (", phase, ")")
end


# =============================================================================
# Prompt assembly
# =============================================================================

"""
    render_enrichment(sections, ctx) -> String

Concatenate the non-empty renders of all sections. Returns `""` if
every section produces empty output or the sections list is empty.
"""
function render_enrichment(sections::Vector{<:AbstractPromptSection},
                           ctx::MutationContext)::String
    parts = String[]
    for s in sections
        text = render(s, ctx)
        isempty(text) || push!(parts, text)
    end
    return join(parts, "\n\n")
end


# =============================================================================
# Solve-loop integration helpers
# =============================================================================

# Forward declaration — the actual LLMMutationOperator type is defined in
# llm_operator.jl which is included after this file. We use duck-typing
# (hasproperty checks) in the helpers below so we don't need to reference
# the concrete type here.

"""
    _find_llm_operators(ops) -> Vector

Walk `mutation_ops`, unwrapping any wrapper that has an `.inner` field.
Returns all objects that have both `.sections` and `.context` fields
(i.e. LLMMutationOperator instances with prompt enrichment support).
"""
function _find_llm_operators(ops)
    result = []
    for op in ops
        if hasproperty(op, :sections) && hasproperty(op, :context)
            push!(result, op)
        elseif hasproperty(op, :inner)
            inner = getproperty(op, :inner)
            if hasproperty(inner, :sections) && hasproperty(inner, :context)
                push!(result, inner)
            end
        end
    end
    return result
end

"""
    _update_llm_contexts!(mutation_ops, generation, max_generations, fitnesses, genomes)

Find all LLM operators in `mutation_ops` (unwrapping wrappers) and update
their `MutationContext` with current population state. Serializes only
the top-K genomes where K is the maximum `k` across all `ElitesSection`s.

Called once per generation, after sorting and speciation, before breeding.
"""
function _update_llm_contexts!(mutation_ops, generation::Int,
                                max_generations::Int,
                                fitnesses::Vector{Float64},
                                genomes::Vector{<:AbstractGenome})
    llm_ops = _find_llm_operators(mutation_ops)
    isempty(llm_ops) && return

    # Determine max k needed across all operators' ElitesSections.
    max_k = 0
    for op in llm_ops
        for s in op.sections
            if s isa ElitesSection
                max_k = max(max_k, s.k)
            end
        end
    end

    # Serialize only the top max_k genomes (expensive, do once per generation).
    k = min(max_k, length(genomes))
    serialized = String[serialize(genomes[i]) for i in 1:k]

    # Update each operator's context.
    for op in llm_ops
        if op.context === nothing
            op.context = MutationContext()
        end
        ctx = op.context
        ctx.generation = generation
        ctx.max_generations = max_generations
        ctx.fitnesses = fitnesses
        ctx.genomes_serialized = serialized
    end
end

"""
    _set_parent_context!(mutation_ops, parent_idx, fitnesses)

Set per-call parent fields on all LLM operators' contexts. Called inside
the breeding loop after tournament selection, before `mutate`.

`parent_idx` is the 1-based index into the (sorted) `fitnesses` vector,
which doubles as the parent's rank.
"""
function _set_parent_context!(mutation_ops, parent_idx::Int,
                               fitnesses::Vector{Float64})
    for op in _find_llm_operators(mutation_ops)
        ctx = op.context
        ctx === nothing && continue
        ctx.parent_fitness = fitnesses[parent_idx]
        ctx.parent_rank = parent_idx
    end
end
