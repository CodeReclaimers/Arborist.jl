# nsga2.jl — NSGA-II multi-objective genetic programming
#
# Self-contained implementation of NSGA-II (Deb et al. 2002) for Arborist.
# Provides multi-objective evaluators, the NSGA-II algorithm type, and
# solve methods for TreeGenome and ExprGenome.

# =============================================================================
# Multi-objective evaluator types
# =============================================================================

"""
    AbstractMultiObjectiveEvaluator <: AbstractEvaluator

Base type for multi-objective fitness evaluators. All objectives are minimized
(lower is better), consistent with the rest of Arborist.

Concrete subtypes must implement:
- `evaluate_multi(e, genome::AbstractGenome) -> Vector{Float64}`
- `objective_names(e) -> Vector{String}`
- `input_signature(e)` and `output_signature(e)` (inherited from AbstractEvaluator)
"""
abstract type AbstractMultiObjectiveEvaluator <: AbstractEvaluator end

"""
    evaluate_multi(e::AbstractMultiObjectiveEvaluator, genome::AbstractGenome) -> Vector{Float64}

Evaluate a genome on all objectives. Returns a vector of fitness values,
one per objective, all minimized (lower is better). Returns a vector of `Inf`
values on evaluation failure.
"""
function evaluate_multi end

"""
    objective_names(e::AbstractMultiObjectiveEvaluator) -> Vector{String}

Return human-readable names for each objective.
"""
function objective_names end

"""
    ParsimonyEvaluator{E<:AbstractEvaluator} <: AbstractMultiObjectiveEvaluator

Wraps a single-objective evaluator to produce two objectives:
1. The original fitness (from the wrapped evaluator)
2. Genome complexity (from `complexity(genome)`)

This is the canonical GP multi-objective setup: accuracy vs. parsimony.

# Example
```julia
inner = TreeFitnessEvaluator(X, y, operators)
evaluator = ParsimonyEvaluator(inner)
problem = GPProblem(evaluator, TreeGenome{Float32}; seed=42)
result = solve(problem, NSGAII(pop_size=100, generations=50))
```
"""
struct ParsimonyEvaluator{E<:AbstractEvaluator} <: AbstractMultiObjectiveEvaluator
    inner::E
end

function evaluate_multi(e::ParsimonyEvaluator, genome::AbstractGenome)
    raw = evaluate_genome(genome, e.inner)
    c = Float64(complexity(genome))
    return [raw, c]
end

objective_names(::ParsimonyEvaluator) = ["fitness", "complexity"]
input_signature(e::ParsimonyEvaluator) = input_signature(e.inner)
output_signature(e::ParsimonyEvaluator) = output_signature(e.inner)

# =============================================================================
# NSGAII algorithm type
# =============================================================================

"""
    NSGAII <: AbstractEvolutionaryAlgorithm

NSGA-II (Non-dominated Sorting Genetic Algorithm II) for multi-objective
genetic programming. Uses Pareto non-dominated sorting and crowding distance
for selection, with a (mu+lambda) replacement strategy.

No `elitism` field — NSGA-II's combined parent+offspring sorting IS the
elitism mechanism. No `speciation` — crowding distance maintains diversity.
No `bloat_penalty` — complexity should be an explicit objective via
`ParsimonyEvaluator`.

# Fields
- `pop_size::Int`: population size, must be even (default: 100)
- `generations::Int`: number of generations (default: 200)
- `mutation_rate::Float64`: probability of mutation (default: 0.3)
- `crossover_rate::Float64`: probability of crossover (default: 0.3)
- `parallel::Bool`: enable threaded evaluation (default: true)
- `mutation_ops::Vector{AbstractMutationOperator}`: mutation operators
- `crossover_ops::Vector{AbstractCrossoverOperator}`: crossover operators
"""
struct NSGAII <: AbstractEvolutionaryAlgorithm
    pop_size::Int
    generations::Int
    mutation_rate::Float64
    crossover_rate::Float64
    parallel::Bool
    mutation_ops::Vector{AbstractMutationOperator}
    crossover_ops::Vector{AbstractCrossoverOperator}
end

"""
    NSGAII(; kwargs...) -> NSGAII

Construct an `NSGAII` algorithm with keyword arguments and sensible defaults.
"""
function NSGAII(;
    pop_size::Int = 100,
    generations::Int = 200,
    mutation_rate::Float64 = 0.3,
    crossover_rate::Float64 = 0.3,
    parallel::Bool = true,
    mutation_ops::Vector{<:AbstractMutationOperator} = AbstractMutationOperator[SubtreeMutation(), PointMutation()],
    crossover_ops::Vector{<:AbstractCrossoverOperator} = AbstractCrossoverOperator[SubtreeCrossover()]
)
    if crossover_rate + mutation_rate > 1.0
        throw(ArgumentError(
            "crossover_rate ($crossover_rate) + mutation_rate ($mutation_rate) = " *
            "$(crossover_rate + mutation_rate) exceeds 1.0."))
    end
    if pop_size % 2 != 0
        throw(ArgumentError("pop_size ($pop_size) must be even for NSGA-II."))
    end
    if pop_size < 4
        throw(ArgumentError("pop_size ($pop_size) must be at least 4 for NSGA-II."))
    end
    NSGAII(
        pop_size, generations, mutation_rate, crossover_rate,
        parallel,
        convert(Vector{AbstractMutationOperator}, mutation_ops),
        convert(Vector{AbstractCrossoverOperator}, crossover_ops)
    )
end

# =============================================================================
# NSGAIIResult type
# =============================================================================

"""
    NSGAIIResult{G<:AbstractGenome} <: AbstractEvolutionResult

Result returned by `solve` with `NSGAII`. Contains the Pareto front,
full population, fitness vectors, and convergence history.

# Fields
- `pareto_front::Vector{G}`: non-dominated genomes from the final population
- `pareto_fitnesses::Vector{Vector{Float64}}`: fitness vectors for the Pareto front
- `population::Vector{G}`: full final population
- `all_fitnesses::Vector{Vector{Float64}}`: fitness vectors for all individuals
- `hypervolume_history::Vector{Float64}`: hypervolume of front 1 per generation
- `generations_run::Int`: number of generations completed
- `wall_time::Float64`: elapsed wall-clock time in seconds
- `objective_names::Vector{String}`: human-readable names for each objective
"""
struct NSGAIIResult{G<:AbstractGenome} <: AbstractEvolutionResult
    pareto_front::Vector{G}
    pareto_fitnesses::Vector{Vector{Float64}}
    population::Vector{G}
    all_fitnesses::Vector{Vector{Float64}}
    hypervolume_history::Vector{Float64}
    generations_run::Int
    wall_time::Float64
    objective_names::Vector{String}
end

# =============================================================================
# Core NSGA-II algorithms
# =============================================================================

"""
    _dominates(a::Vector{Float64}, b::Vector{Float64}) -> Bool

Return true if `a` Pareto-dominates `b` (all objectives <= and at least one <).
All objectives are minimized.
"""
function _dominates(a::Vector{Float64}, b::Vector{Float64})
    dominated = false
    for i in eachindex(a)
        if a[i] > b[i]
            return false
        elseif a[i] < b[i]
            dominated = true
        end
    end
    return dominated
end

"""
    _nondominated_sort(fitnesses::Vector{Vector{Float64}}) -> Vector{Int}

Assign each individual a Pareto rank (1 = first front, 2 = second, etc.).
Uses the O(MN^2) algorithm from Deb et al. (2002).
All objectives are minimized (lower is better).
"""
function _nondominated_sort(fitnesses::Vector{Vector{Float64}})
    n = length(fitnesses)
    ranks = zeros(Int, n)
    domination_count = zeros(Int, n)       # how many dominate individual i
    dominated_set = [Int[] for _ in 1:n]   # which individuals does i dominate

    # Build domination relationships.
    for i in 1:n
        for j in (i+1):n
            if _dominates(fitnesses[i], fitnesses[j])
                push!(dominated_set[i], j)
                domination_count[j] += 1
            elseif _dominates(fitnesses[j], fitnesses[i])
                push!(dominated_set[j], i)
                domination_count[i] += 1
            end
        end
    end

    # First front: individuals with domination_count == 0.
    current_front = Int[]
    for i in 1:n
        if domination_count[i] == 0
            ranks[i] = 1
            push!(current_front, i)
        end
    end

    # Subsequent fronts.
    rank = 1
    while !isempty(current_front)
        next_front = Int[]
        for i in current_front
            for j in dominated_set[i]
                domination_count[j] -= 1
                if domination_count[j] == 0
                    ranks[j] = rank + 1
                    push!(next_front, j)
                end
            end
        end
        rank += 1
        current_front = next_front
    end

    return ranks
end

"""
    _crowding_distance(fitnesses::Vector{Vector{Float64}}, front_indices::Vector{Int}) -> Vector{Float64}

Compute crowding distance for individuals in a single Pareto front.
Boundary individuals (best/worst in any objective) get `Inf`.
Returns a vector of distances indexed by position in `front_indices`.
"""
function _crowding_distance(fitnesses::Vector{Vector{Float64}}, front_indices::Vector{Int})
    nf = length(front_indices)
    if nf <= 2
        return fill(Inf, nf)
    end

    n_objectives = length(fitnesses[front_indices[1]])
    distances = zeros(Float64, nf)

    for m in 1:n_objectives
        # Sort front members by objective m.
        obj_vals = [fitnesses[front_indices[k]][m] for k in 1:nf]
        sorted_order = sortperm(obj_vals)

        # Boundary individuals get infinite distance.
        distances[sorted_order[1]] = Inf
        distances[sorted_order[end]] = Inf

        # Objective range for normalization.
        obj_min = obj_vals[sorted_order[1]]
        obj_max = obj_vals[sorted_order[end]]
        obj_range = obj_max - obj_min

        if obj_range < 1e-30
            continue  # all values identical for this objective
        end

        # Interior individuals.
        for k in 2:(nf - 1)
            distances[sorted_order[k]] += (obj_vals[sorted_order[k + 1]] - obj_vals[sorted_order[k - 1]]) / obj_range
        end
    end

    return distances
end

"""
    _nsga2_tournament_select(ranks::Vector{Int}, crowding::Vector{Float64}, rng) -> Int

Binary tournament using NSGA-II's crowded comparison operator:
prefer lower rank; if tied, prefer higher crowding distance.
"""
function _nsga2_tournament_select(ranks::Vector{Int}, crowding::Vector{Float64},
                                   rng::AbstractRNG)
    n = length(ranks)
    a = rand(rng, 1:n)
    b = rand(rng, 1:n)

    if ranks[a] < ranks[b]
        return a
    elseif ranks[b] < ranks[a]
        return b
    else
        # Same rank: prefer higher crowding distance.
        return crowding[a] >= crowding[b] ? a : b
    end
end

"""
    _hypervolume_2d(front_fitnesses::Vector{Vector{Float64}}, ref_point::Vector{Float64}) -> Float64

Compute 2D hypervolume indicator for a Pareto front (minimization).
Uses the O(n log n) sweep algorithm. Points that exceed the reference
point in any objective are excluded.
"""
function _hypervolume_2d(front_fitnesses::Vector{Vector{Float64}},
                          ref_point::Vector{Float64})
    # Filter points dominated by reference point.
    valid = [f for f in front_fitnesses if f[1] <= ref_point[1] && f[2] <= ref_point[2]]
    isempty(valid) && return 0.0

    # Sort by first objective ascending.
    sort!(valid; by=f -> f[1])

    hv = 0.0
    prev_y = ref_point[2]
    for f in valid
        width = ref_point[1] - f[1]
        # Only count the strip if this point improves the second objective.
        if f[2] < prev_y
            # The strip from f[1] to the reference extends upward from f[2].
            # But we only count up to the previous best y.
            hv += (prev_y - f[2]) * (ref_point[1] - f[1])
            prev_y = f[2]
        end
    end

    return hv
end

"""
    _compute_hypervolume(fitnesses, ranks, ref_margin=1.1) -> Float64

Compute hypervolume of the first Pareto front. Reference point is derived
from the worst objective values across the entire population, scaled by
`ref_margin`. Currently supports 2 objectives.
"""
function _compute_hypervolume(fitnesses::Vector{Vector{Float64}},
                               ranks::Vector{Int};
                               ref_margin::Float64 = 1.1)
    n_objectives = length(fitnesses[1])
    if n_objectives != 2
        return 0.0  # only 2D hypervolume implemented
    end

    # Compute reference point from worst finite values.
    ref_point = fill(-Inf, n_objectives)
    for f in fitnesses
        for m in 1:n_objectives
            if isfinite(f[m]) && f[m] > ref_point[m]
                ref_point[m] = f[m]
            end
        end
    end
    # Handle case where all values are Inf.
    for m in 1:n_objectives
        if !isfinite(ref_point[m])
            ref_point[m] = 1.0
        end
    end
    ref_point .*= ref_margin

    # Extract front 1 fitnesses.
    front1 = [fitnesses[i] for i in 1:length(fitnesses) if ranks[i] == 1]
    # Filter out individuals with any Inf objective.
    front1 = [f for f in front1 if all(isfinite, f)]

    return _hypervolume_2d(front1, ref_point)
end

# =============================================================================
# Breeding (NSGA-II selection + genetic operators)
# =============================================================================

"""
    _nsga2_breed!(next_genomes, genomes, ranks, crowding, alg, rng, start_idx)

Fill `next_genomes[start_idx:end]` using NSGA-II tournament selection and
genetic operators. Same crossover/mutation/reproduction split as the
single-objective `_breed_next_generation!`.
"""
function _nsga2_breed!(next_genomes::Vector{G},
                        genomes::Vector{G},
                        ranks::Vector{Int},
                        crowding::Vector{Float64},
                        alg::NSGAII,
                        rng::AbstractRNG,
                        start_idx::Int) where G
    pop_size = length(next_genomes)
    idx = start_idx
    while idx <= pop_size
        r = rand(rng)
        if r < alg.crossover_rate && idx + 1 <= pop_size
            p1 = _nsga2_tournament_select(ranks, crowding, rng)
            p2 = _nsga2_tournament_select(ranks, crowding, rng)
            op = rand(rng, alg.crossover_ops)
            (c1, c2) = crossover(op, genomes[p1], genomes[p2], rng)
            next_genomes[idx] = c1
            next_genomes[idx + 1] = c2
            idx += 2
        elseif r < alg.crossover_rate + alg.mutation_rate
            p_idx = _nsga2_tournament_select(ranks, crowding, rng)
            op = rand(rng, alg.mutation_ops)
            next_genomes[idx] = mutate(op, genomes[p_idx], rng)
            idx += 1
        else
            p_idx = _nsga2_tournament_select(ranks, crowding, rng)
            next_genomes[idx] = deepcopy(genomes[p_idx])
            idx += 1
        end
    end
end

# =============================================================================
# Multi-objective evaluation
# =============================================================================

"""
    _parallel_evaluate_multi!(fitnesses, genomes, evaluator, indices, parallel)

Evaluate genomes at the given indices using `evaluate_multi`, optionally
using threads. Each entry in `fitnesses` is a `Vector{Float64}`.
"""
function _parallel_evaluate_multi!(fitnesses::Vector{Vector{Float64}},
                                    genomes::Vector,
                                    evaluator::AbstractMultiObjectiveEvaluator,
                                    indices,
                                    parallel::Bool)
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in collect(indices)
            fitnesses[i] = evaluate_multi(evaluator, genomes[i])
        end
    else
        for i in indices
            fitnesses[i] = evaluate_multi(evaluator, genomes[i])
        end
    end
end

# =============================================================================
# Population initialization helpers
# =============================================================================

"""
    _nsga2_init_population(problem, algorithm, rng) -> Vector{G}

Initialize a population for NSGA-II. Dispatches on genome type.
"""
function _nsga2_init_population end

# --- ExprGenome initialization ---
function _nsga2_init_population(problem::GPProblem{ExprGenome, E},
                                 algorithm::NSGAII,
                                 rng::AbstractRNG) where {E<:AbstractMultiObjectiveEvaluator}
    inputs = input_signature(problem.evaluator)
    outputs = output_signature(problem.evaluator)
    state = GenState(rng, problem.function_set, inputs, outputs, problem.num_temps)

    genomes = Vector{ExprGenome}(undef, algorithm.pop_size)
    for i in 1:algorithm.pop_size
        body = [create_random_assignment(state) for _ in 1:3]
        genomes[i] = ExprGenome(body, state)
    end
    return genomes
end

# --- TreeGenome initialization ---
function _nsga2_init_population(problem::GPProblem{TreeGenome{T}, E},
                                 algorithm::NSGAII,
                                 rng::AbstractRNG) where {T, E<:AbstractMultiObjectiveEvaluator}
    # Unwrap ParsimonyEvaluator to get the inner TreeFitnessEvaluator.
    evaluator = problem.evaluator
    inner = evaluator isa ParsimonyEvaluator ? evaluator.inner : evaluator
    if !(inner isa TreeFitnessEvaluator)
        error("NSGA-II with TreeGenome requires the evaluator (or its inner evaluator) " *
              "to be a TreeFitnessEvaluator. Got: $(typeof(inner))")
    end

    ops = inner.operators
    n_feat = size(inner.X, 1)
    pop_size = algorithm.pop_size

    genomes = Vector{TreeGenome{T}}(undef, pop_size)
    for i in 1:pop_size
        method = i <= pop_size ÷ 2 ? :full : :grow
        depth = 2 + (i % 3)  # depths 2, 3, 4
        tree = _random_tree(rng, ops, n_feat, T, depth, method)
        genomes[i] = TreeGenome{T}(tree, ops, n_feat)
    end
    return genomes
end

# =============================================================================
# NSGA-II survivor selection
# =============================================================================

"""
    _nsga2_select_survivors!(genomes, fitnesses, pop_size) -> (Vector{G}, Vector{Vector{Float64}}, Vector{Int}, Vector{Float64})

From a combined parent+offspring population of 2N, select the best N
individuals using non-dominated sorting and crowding distance.
Returns (selected_genomes, selected_fitnesses, ranks, crowding_distances).
"""
function _nsga2_select_survivors(genomes::Vector{G},
                                  fitnesses::Vector{Vector{Float64}},
                                  pop_size::Int) where G
    n = length(genomes)
    ranks = _nondominated_sort(fitnesses)

    selected_genomes = Vector{G}(undef, pop_size)
    selected_fitnesses = Vector{Vector{Float64}}(undef, pop_size)
    selected_ranks = zeros(Int, pop_size)
    selected_crowding = zeros(Float64, pop_size)

    filled = 0
    max_rank = maximum(ranks)

    for rank in 1:max_rank
        front_indices = [i for i in 1:n if ranks[i] == rank]
        crowding = _crowding_distance(fitnesses, front_indices)

        if filled + length(front_indices) <= pop_size
            # Entire front fits.
            for (k, idx) in enumerate(front_indices)
                filled += 1
                selected_genomes[filled] = genomes[idx]
                selected_fitnesses[filled] = fitnesses[idx]
                selected_ranks[filled] = rank
                selected_crowding[filled] = crowding[k]
            end
        else
            # Partial front: select by crowding distance (higher is better).
            remaining = pop_size - filled
            order = sortperm(crowding; rev=true)
            for k in 1:remaining
                idx = front_indices[order[k]]
                filled += 1
                selected_genomes[filled] = genomes[idx]
                selected_fitnesses[filled] = fitnesses[idx]
                selected_ranks[filled] = rank
                selected_crowding[filled] = crowding[order[k]]
            end
            break
        end
    end

    return (selected_genomes, selected_fitnesses, selected_ranks, selected_crowding)
end

# =============================================================================
# Solve method
# =============================================================================

"""
    solve(problem::GPProblem{G,E}, algorithm::NSGAII; verbose=false, callback=nothing) -> NSGAIIResult{G}

Run NSGA-II multi-objective evolution. Returns an `NSGAIIResult` containing the
Pareto front, fitness vectors, and hypervolume history.

# Arguments
- `problem::GPProblem{G,E}`: problem with a multi-objective evaluator
- `algorithm::NSGAII`: NSGA-II configuration

# Keyword Arguments
- `verbose::Bool=false`: print generation statistics
- `callback`: optional `(gen::Int, front_size::Int, hypervolume::Float64) -> nothing`
"""
function solve(problem::GPProblem{G, E},
               algorithm::NSGAII;
               verbose::Bool = false,
               callback = nothing) where {G, E<:AbstractMultiObjectiveEvaluator}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    evaluator = problem.evaluator
    pop_size = algorithm.pop_size
    n_objectives = length(objective_names(evaluator))

    # Initialize population.
    genomes = _nsga2_init_population(problem, algorithm, rng)

    # Evaluate initial population.
    fitnesses = [fill(Inf, n_objectives) for _ in 1:pop_size]
    _parallel_evaluate_multi!(fitnesses, genomes, evaluator, 1:pop_size, algorithm.parallel)

    hypervolume_history = Float64[]
    t0 = time()

    for gen in 1:algorithm.generations
        # Compute ranks and crowding for parent selection.
        ranks = _nondominated_sort(fitnesses)
        crowding = zeros(Float64, pop_size)
        max_rank = maximum(ranks)
        for rank in 1:max_rank
            front_indices = [i for i in 1:pop_size if ranks[i] == rank]
            front_crowding = _crowding_distance(fitnesses, front_indices)
            for (k, idx) in enumerate(front_indices)
                crowding[idx] = front_crowding[k]
            end
        end

        # Record hypervolume of first front.
        hv = _compute_hypervolume(fitnesses, ranks)
        push!(hypervolume_history, hv)

        if verbose
            front1_size = count(r -> r == 1, ranks)
            println("Generation $gen: front1=$front1_size, hypervolume=$(round(hv, digits=6))")
            flush(stdout)
        end

        if callback !== nothing
            front1_size = count(r -> r == 1, ranks)
            callback(gen, front1_size, hv)
        end

        # Update LLM operator contexts — use first objective as scalar fitness proxy.
        _scalar_fits = Float64[f[1] for f in fitnesses]
        _scalar_order = sortperm(_scalar_fits)
        _update_llm_contexts!(algorithm.mutation_ops, gen, algorithm.generations,
                               _scalar_fits[_scalar_order], genomes[_scalar_order])

        # Create offspring population via NSGA-II tournament selection.
        offspring = Vector{G}(undef, pop_size)
        _nsga2_breed!(offspring, genomes, ranks, crowding, algorithm, rng, 1)

        # Evaluate offspring.
        offspring_fitnesses = [fill(Inf, n_objectives) for _ in 1:pop_size]
        _parallel_evaluate_multi!(offspring_fitnesses, offspring, evaluator,
                                   1:pop_size, algorithm.parallel)

        # Combine parents + offspring (2N).
        combined_genomes = vcat(genomes, offspring)
        combined_fitnesses = vcat(fitnesses, offspring_fitnesses)

        # Select N survivors via non-dominated sorting + crowding.
        genomes, fitnesses, _, _ = _nsga2_select_survivors(
            combined_genomes, combined_fitnesses, pop_size)
    end

    wall_time = time() - t0

    # Extract final Pareto front (rank 1).
    final_ranks = _nondominated_sort(fitnesses)
    front_indices = [i for i in 1:pop_size if final_ranks[i] == 1]

    pareto_front = [genomes[i] for i in front_indices]
    pareto_fitnesses = [fitnesses[i] for i in front_indices]

    # Sort Pareto front by first objective for presentation.
    pf_order = sortperm(pareto_fitnesses; by=f -> f[1])
    pareto_front = pareto_front[pf_order]
    pareto_fitnesses = pareto_fitnesses[pf_order]

    return NSGAIIResult{G}(
        pareto_front,
        pareto_fitnesses,
        genomes,
        fitnesses,
        hypervolume_history,
        algorithm.generations,
        wall_time,
        objective_names(evaluator)
    )
end
