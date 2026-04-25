"""
    solve(problem::GPProblem{G,E}, algorithm::GeneticProgramming;
          verbose=false, callback=nothing, log=nothing,
          checkpoint_every=0, checkpoint_path=nothing,
          resume_from=nothing, allow_signature_mismatch=false,
          initial_population=nothing, hall_of_fame_size=0) -> GPResult{G}

Run a genetic programming evolution using the specified problem and algorithm configuration.
Returns a `GPResult` containing the best genome, fitness history, and run metadata.

# Arguments
- `problem::GPProblem{G,E}`: the problem specification (evaluator, genome type, function set)
- `algorithm::GeneticProgramming`: the algorithm configuration

# Keyword Arguments
- `verbose::Bool=false`: if true, print generation statistics
- `callback`: optional callback function `(gen::Int, best_fitness::Float64, best_genome::G) -> nothing`
- `log::Union{Nothing, RunLog}=nothing`: optional structured per-generation log.
  When provided, `record!` is called once per generation with aggregate fitness,
  speciation snapshot, structural diversity, and wall-time.
- `checkpoint_every::Int=0`: write a checkpoint every N generations. `0` disables
  checkpointing. When `> 0`, `checkpoint_path` must also be set. Supported on
  `ExprGenome` and `TreeGenome` only.
- `checkpoint_path::Union{Nothing, AbstractString}=nothing`: file path the
  periodic checkpoints (and the final-generation checkpoint) are written to via
  `save_checkpoint`. Writes are atomic (write-tmp + rename).
- `resume_from::Union{Nothing, AbstractString}=nothing`: path to a checkpoint
  produced by a prior `solve`. Loads the population, generation counter, RNG
  state, fitness history, and all-time best from the checkpoint and continues
  from there. Mutually exclusive with `initial_population`.
- `allow_signature_mismatch::Bool=false`: by default, `resume_from` rejects a
  checkpoint whose `_algorithm_signature` does not match the current
  `algorithm`. Pass `true` to override intentionally (e.g., changing
  hyperparameters mid-run).
- `initial_population::Union{Nothing, Vector{<:AbstractGenome}}=nothing`: warm-
  start the GA from a user-provided seed pool. Length must equal
  `algorithm.pop_size` and element type must match `G`. Mutually exclusive with
  `resume_from`.
- `hall_of_fame_size::Int=0`: when `> 0`, attach a `HallOfFame{G}` archive of
  this capacity to the result. The archive tracks the top-K distinct fitnesses
  observed across all generations (best-first), surviving elitism loss. The
  archive is exposed as `result.hall_of_fame`. `0` (default) disables the
  archive and leaves `result.hall_of_fame === nothing`.
"""
function solve(problem::GPProblem{G,E},
               algorithm::GeneticProgramming;
               verbose::Bool = false,
               callback = nothing,
               log::Union{Nothing, RunLog} = nothing,
               checkpoint_every::Int = 0,
               checkpoint_path::Union{Nothing, AbstractString} = nothing,
               resume_from::Union{Nothing, AbstractString} = nothing,
               allow_signature_mismatch::Bool = false,
               initial_population::Union{Nothing, Vector{<:AbstractGenome}} = nothing,
               hall_of_fame_size::Int = 0) where {G,E}
    checkpoint_every >= 0 || throw(ArgumentError("checkpoint_every must be >= 0"))
    if checkpoint_every > 0 && checkpoint_path === nothing
        throw(ArgumentError("checkpoint_every > 0 requires a checkpoint_path"))
    end
    if resume_from !== nothing && initial_population !== nothing
        throw(ArgumentError(
            "solve: pass either `resume_from` or `initial_population`, not both"))
    end
    hall_of_fame_size >= 0 || throw(ArgumentError(
        "hall_of_fame_size must be >= 0, got $hall_of_fame_size"))

    # Resume path: load the checkpoint, validate the algorithm signature,
    # and hand the prior state to _run_evolution!. Seed RNG from the checkpoint.
    if resume_from !== nothing
        ckpt = load_checkpoint(resume_from)
        ckpt.population isa Vector{G} || throw(ArgumentError(
            "checkpoint population type $(eltype(ckpt.population)) does not match " *
            "current genome type $G"))
        expected_sig = _algorithm_signature(algorithm)
        if ckpt.algorithm_signature != expected_sig && !allow_signature_mismatch
            throw(ArgumentError(
                "algorithm signature mismatch on resume. " *
                "Checkpoint signature 0x$(string(ckpt.algorithm_signature, base=16)), " *
                "current algorithm 0x$(string(expected_sig, base=16)). " *
                "Pass `allow_signature_mismatch=true` to override intentionally."))
        end
        hof_resume = hall_of_fame_size > 0 ? HallOfFame{G}(hall_of_fame_size) : nothing
        return _run_evolution!(ckpt, problem, algorithm;
                               verbose=verbose, callback=callback, log=log,
                               checkpoint_every=checkpoint_every,
                               checkpoint_path=checkpoint_path,
                               hall_of_fame=hof_resume)
    end

    # Fresh start.
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    pop = if initial_population === nothing
        _initialize_population(problem, algorithm, rng)
    else
        _validate_initial_population(initial_population, algorithm.pop_size, G)
        fresh = _initialize_population(problem, algorithm, rng)
        # Keep the state carrier (GenState / TreeGenomeContext / ...) built by
        # _initialize_population but swap in the user-provided genomes.
        (deepcopy(Vector{G}(initial_population)), fresh[2])
    end
    hof = hall_of_fame_size > 0 ? HallOfFame{G}(hall_of_fame_size) : nothing
    return _run_evolution!(pop, problem, algorithm, rng;
                           verbose=verbose, callback=callback, log=log,
                           checkpoint_every=checkpoint_every,
                           checkpoint_path=checkpoint_path,
                           hall_of_fame=hof)
end

"""
    _validate_initial_population(pop, pop_size, ::Type{G})

Throw `ArgumentError` when a user-supplied warm-start population does
not match the algorithm's population size or genome type. Called from
every `solve()` method that accepts `initial_population`.
"""
function _validate_initial_population(pop::AbstractVector,
                                      pop_size::Int,
                                      ::Type{G}) where {G}
    length(pop) == pop_size || throw(ArgumentError(
        "initial_population length ($(length(pop))) must equal " *
        "algorithm.pop_size ($pop_size). Warm-start does not pad or truncate."))
    for (i, g) in enumerate(pop)
        g isa G || throw(ArgumentError(
            "initial_population[$i] has type $(typeof(g)); " *
            "expected $G to match the problem's genome type."))
    end
    return nothing
end

"""
    _initialize_population(problem, algorithm, rng) -> Tuple{Vector{G}, state}

Create the initial population of genomes. Internal function. Dispatches
on the genome type. The second tuple element is a per-island state
carrier that exposes `.rng` (e.g. `GenState` for `ExprGenome`,
`TreeGenomeContext` for `TreeGenome`); the IslandModel loop reads
`state.rng` uniformly across genome types.
"""
function _initialize_population(problem::GPProblem{ExprGenome, E},
                                 algorithm::GeneticProgramming,
                                 rng::AbstractRNG) where {E}
    inputs = input_signature(problem.evaluator)
    outputs = output_signature(problem.evaluator)
    state = GenState(rng, problem.function_set, inputs, outputs, problem.num_temps)

    genomes = Vector{ExprGenome}(undef, algorithm.pop_size)
    for i in 1:algorithm.pop_size
        body = [create_random_assignment(state) for _ in 1:3]
        genomes[i] = ExprGenome(body, state)
    end

    return (genomes, state)
end

# Fallback for genome types that do not yet support IslandModel.
# TreeGenome has its own method defined in tree_genome.jl.
function _initialize_population(problem::GPProblem{G,E},
                                 algorithm::GeneticProgramming,
                                 rng::AbstractRNG) where {G,E}
    error("_initialize_population does not support $(G). " *
          "IslandModel currently supports ExprGenome and TreeGenome; " *
          "other genome types (AntGenome, GraphGenome) require a " *
          "specialized solve method.")
end

"""
    _validate_ops(mutation_ops, crossover_ops, ::Type{G})

Verify that at least one operator in each vector has a `mutate` / `crossover`
method dispatched on genome type `G`. Throws `ArgumentError` with a pointer
at the right default operators if no compatible operator is present. This
catches the common case where a user constructs `GeneticProgramming` or
`NSGAII` with the default ExprGenome operators and hands it a GraphGenome
problem — without this check the failure would surface as a cryptic
`MethodError` deep inside the breed loop.
"""
function _validate_ops(mutation_ops::Vector{AbstractMutationOperator},
                       crossover_ops::Vector{AbstractCrossoverOperator},
                       ::Type{G}) where {G}
    have_mut = any(op -> hasmethod(mutate, Tuple{typeof(op), G, AbstractRNG}), mutation_ops)
    have_xo  = any(op -> hasmethod(crossover, Tuple{typeof(op), G, G, AbstractRNG}), crossover_ops)

    hint = if G === GraphGenome
        "Use `neat_defaults()` to get (mutation_ops, crossover_ops) for GraphGenome: " *
        "`ops = neat_defaults(); GeneticProgramming(; mutation_ops=ops.mutation_ops, " *
        "crossover_ops=ops.crossover_ops, ...)`."
    else
        "Pass operators that dispatch on $G."
    end

    if !have_mut
        throw(ArgumentError(
            "No mutation operator in `mutation_ops` dispatches on $G. " *
            "Got types: $([typeof(op) for op in mutation_ops]). $hint"))
    end
    if !have_xo
        throw(ArgumentError(
            "No crossover operator in `crossover_ops` dispatches on $G. " *
            "Got types: $([typeof(op) for op in crossover_ops]). $hint"))
    end
    return nothing
end

"""
    _breed_next_generation!(next_genomes, genomes, selection_fitnesses, alg, rng, start_idx;
                             case_fitnesses=nothing)

Fill `next_genomes[start_idx:end]` via selection and genetic operators
(crossover, mutation, or copy). Shared across all solve paths.

Parent selection dispatches through `select_parent(alg.selection, ...)`:
`TournamentSelection` uses `selection_fitnesses` only; lexicase strategies
consume `case_fitnesses`. The solve loop materializes `case_fitnesses`
when `needs_cases(alg.selection)` is true.

Crossover and mutation dispatch through operator objects from `alg.crossover_ops`
and `alg.mutation_ops`. Genome types that use direct dispatch (AntGenome,
GraphGenome) provide fallback methods that ignore the operator argument.
"""
function _breed_next_generation!(next_genomes::Vector{G},
                                  genomes::Vector{G},
                                  selection_fitnesses::Vector{Float64},
                                  alg::GeneticProgramming,
                                  rng::AbstractRNG,
                                  start_idx::Int;
                                  case_fitnesses = nothing,
                                  op_track::Union{Nothing, Vector{Symbol}} = nothing,
                                  ) where G
    pop_size = length(next_genomes)
    sel = alg.selection
    idx = start_idx
    while idx <= pop_size
        r = rand(rng)
        if r < alg.crossover_rate && idx + 1 <= pop_size
            p1 = select_parent(sel, selection_fitnesses, case_fitnesses, rng)
            p2 = select_parent(sel, selection_fitnesses, case_fitnesses, rng)
            op = rand(rng, alg.crossover_ops)
            (c1, c2) = crossover(op, genomes[p1], genomes[p2], rng)
            next_genomes[idx] = c1
            next_genomes[idx + 1] = c2
            if op_track !== nothing
                name = operator_name(op)
                op_track[idx] = name
                op_track[idx + 1] = name
            end
            idx += 2
        elseif r < alg.crossover_rate + alg.mutation_rate
            p_idx = select_parent(sel, selection_fitnesses, case_fitnesses, rng)
            op = rand(rng, alg.mutation_ops)
            _set_parent_context!(alg.mutation_ops, p_idx, selection_fitnesses)
            next_genomes[idx] = mutate(op, genomes[p_idx], rng)
            if op_track !== nothing
                op_track[idx] = operator_name(op)
            end
            idx += 1
        else
            p_idx = select_parent(sel, selection_fitnesses, case_fitnesses, rng)
            next_genomes[idx] = deepcopy(genomes[p_idx])
            if op_track !== nothing
                op_track[idx] = :reproduction
            end
            idx += 1
        end
    end
end

"""
    _save_ckpt(genomes, fitnesses, rng, best_genome, best_fitness,
               fitness_history, mean_history, wall_time, gen, algorithm, path)

Build a `Checkpoint` from the current evolution state and atomically persist
it to `path`. Invoked from `_run_evolution!` at `checkpoint_every` boundaries.
"""
function _save_ckpt(genomes::Vector{G}, fitnesses::Vector{Float64},
                    rng::AbstractRNG, best_genome::G, best_fitness::Float64,
                    fitness_history::Vector{Float64},
                    mean_history::Vector{Float64},
                    wall_time::Float64, gen::Int,
                    algorithm::GeneticProgramming,
                    path::AbstractString) where {G}
    sig = _algorithm_signature(algorithm)
    ckpt = Checkpoint{G}(
        CHECKPOINT_FORMAT_VERSION,
        _arborist_version(),
        VERSION,
        gen,
        deepcopy(genomes),
        copy(fitnesses),
        copy(rng),
        deepcopy(best_genome),
        best_fitness,
        copy(fitness_history),
        copy(mean_history),
        wall_time,
        sig,
    )
    save_checkpoint(ckpt, path)
    return nothing
end

# Read the running project's version from Project.toml. Falls back to v0.0.0
# if the file is unreachable (e.g. under unusual test harnesses).
const _ARBORIST_VERSION_CACHE = Ref{Union{Nothing, VersionNumber}}(nothing)
function _arborist_version()
    _ARBORIST_VERSION_CACHE[] !== nothing && return _ARBORIST_VERSION_CACHE[]
    v = try
        proj = joinpath(dirname(@__DIR__), "Project.toml")
        m = match(r"^version\s*=\s*\"([^\"]+)\""m, read(proj, String))
        m === nothing ? v"0.0.0" : VersionNumber(m.captures[1])
    catch
        v"0.0.0"
    end
    _ARBORIST_VERSION_CACHE[] = v
    return v
end

"""
    _run_evolution!(ckpt::Checkpoint{G}, problem, algorithm; kwargs...) -> GPResult{G}

Resume dispatch: reconstructs the state carrier from a checkpoint and calls
the main `_run_evolution!` with `start_gen = ckpt.generation + 1` and the
prior RNG, population, fitnesses, histories, and best-so-far.
"""
function _run_evolution!(ckpt::Checkpoint{G},
                         problem::GPProblem{G,E},
                         algorithm::GeneticProgramming;
                         verbose::Bool = false,
                         callback = nothing,
                         log::Union{Nothing, RunLog} = nothing,
                         checkpoint_every::Int = 0,
                         checkpoint_path::Union{Nothing, AbstractString} = nothing,
                         hall_of_fame::Union{Nothing, HallOfFame{G}} = nothing,
                         ) where {G,E}
    rng = deepcopy(ckpt.rng_state)
    # Rebuild the per-genome state carrier (GenState for ExprGenome, etc).
    # For ExprGenome, we need a fresh GenState sharing the same RNG.
    fresh_pop = _initialize_population(problem, algorithm, rng)
    state = fresh_pop[2]  # keep the state carrier; discard its population
    pop = (deepcopy(ckpt.population), state)

    return _run_evolution!(pop, problem, algorithm, rng;
                           verbose=verbose, callback=callback, log=log,
                           checkpoint_every=checkpoint_every,
                           checkpoint_path=checkpoint_path,
                           start_gen=ckpt.generation + 1,
                           start_wall=ckpt.wall_time,
                           initial_fitness_history=ckpt.fitness_history,
                           initial_mean_history=ckpt.mean_history,
                           initial_best=(ckpt.best_genome, ckpt.best_fitness),
                           initial_fitnesses=ckpt.fitnesses,
                           hall_of_fame=hall_of_fame)
end

"""
    _compute_case_fitnesses(genomes, evaluator, parallel) -> Vector{Vector{Float64}}

Populate a per-individual per-case loss matrix by calling `evaluate_cases`
on each genome. Used by lexicase-family selection strategies. Each inner
vector has length equal to `length(evaluate_cases(genomes[1], evaluator))`.
"""
function _compute_case_fitnesses(genomes::AbstractVector,
                                 evaluator::AbstractEvaluator,
                                 parallel::Bool)
    n = length(genomes)
    out = Vector{Vector{Float64}}(undef, n)
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            out[i] = evaluate_cases(genomes[i], evaluator)
        end
    else
        for i in 1:n
            out[i] = evaluate_cases(genomes[i], evaluator)
        end
    end
    return out
end

"""
    _evaluate_with_penalty(genome, evaluator, bloat_penalty) -> Float64

Evaluate a genome and apply bloat penalty if non-zero.
"""
function _evaluate_with_penalty(genome, evaluator::AbstractEvaluator, bloat_penalty::Float64)
    raw = evaluate_genome(genome, evaluator)
    if bloat_penalty > 0.0 && isfinite(raw)
        raw += bloat_penalty * complexity(genome)
    end
    return raw
end

"""
    _parallel_evaluate!(fitnesses, genomes, evaluator, bp, indices, parallel)

Evaluate genomes at the given indices, optionally using threads.
Thread-safe: each evaluation is independent with no shared mutable state.
"""
function _parallel_evaluate!(fitnesses::Vector{Float64}, genomes::Vector,
                             evaluator::AbstractEvaluator, bp::Float64,
                             indices, parallel::Bool)
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in collect(indices)
            fitnesses[i] = _evaluate_with_penalty(genomes[i], evaluator, bp)
        end
    else
        for i in indices
            fitnesses[i] = _evaluate_with_penalty(genomes[i], evaluator, bp)
        end
    end
end

"""
    _run_evolution!(pop, problem, algorithm, rng; verbose, callback) -> GPResult

The internal evolution loop. Not part of the public API.
Supports bloat penalty (via algorithm.bloat_penalty) and speciation
(via algorithm.speciation).
"""
function _run_evolution!(pop::Tuple{Vector{G}, GenState},
                         problem::GPProblem{G,E},
                         algorithm::GeneticProgramming,
                         rng::AbstractRNG;
                         verbose::Bool = false,
                         callback = nothing,
                         log::Union{Nothing, RunLog} = nothing,
                         checkpoint_every::Int = 0,
                         checkpoint_path::Union{Nothing, AbstractString} = nothing,
                         start_gen::Int = 1,
                         start_wall::Float64 = 0.0,
                         initial_fitness_history::Vector{Float64} = Float64[],
                         initial_mean_history::Vector{Float64} = Float64[],
                         initial_best::Union{Nothing, Tuple{G, Float64}} = nothing,
                         initial_fitnesses::Union{Nothing, Vector{Float64}} = nothing,
                         hall_of_fame::Union{Nothing, HallOfFame{G}} = nothing,
                         ) where {G,E}
    genomes, state = pop
    pop_size = algorithm.pop_size
    bp = algorithm.bloat_penalty

    # Initial fitnesses: use what the checkpoint provided if resuming;
    # otherwise evaluate the fresh initial population.
    fitnesses = if initial_fitnesses === nothing
        fits = fill(Inf, pop_size)
        _parallel_evaluate!(fits, genomes, problem.evaluator, bp, 1:pop_size, algorithm.parallel)
        fits
    else
        copy(initial_fitnesses)
    end

    species_state = _init_species_state(algorithm.speciation)

    fitness_history = copy(initial_fitness_history)
    mean_history    = copy(initial_mean_history)

    # Wall-clock reference: a fresh run starts at now; a resumed run continues
    # accumulating on top of the prior run's wall time.
    t0 = time() - start_wall

    # Track the all-time best — resume hands this in; fresh runs take it from
    # the first sorted population.
    best_genome_all_time  = initial_best === nothing ? genomes[1] : initial_best[1]
    best_fitness_all_time = initial_best === nothing ? Inf         : initial_best[2]

    for gen in start_gen:algorithm.generations
        # Sort by fitness ascending (best first).
        order = sortperm(fitnesses)
        genomes = genomes[order]
        fitnesses = fitnesses[order]

        # Record history (raw/bloat-penalized fitness, before sharing).
        push!(fitness_history, fitnesses[1])
        finite_fits = filter(isfinite, fitnesses)
        mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
        push!(mean_history, mean_fit)

        if verbose
            println("Generation $gen: best=$(round(fitnesses[1], digits=6)), mean=$(round(mean_fit, digits=6))")
        end

        if callback !== nothing
            callback(gen, fitnesses[1], genomes[1])
        end

        # Apply speciation and compute selection fitnesses (fitness sharing).
        species_snapshot = log === nothing ? nothing : SpeciationSnapshot()
        selection_fitnesses = _apply_speciation!(genomes, fitnesses,
                                                  algorithm.speciation, species_state, rng;
                                                  snapshot=species_snapshot)

        # Materialize case-fitness matrix for lexicase-family selection.
        case_fitnesses = needs_cases(algorithm.selection) ?
            _compute_case_fitnesses(genomes, problem.evaluator, algorithm.parallel) :
            nothing

        if log !== nothing
            record!(log, gen, fitnesses, genomes, time() - t0;
                    snapshot=species_snapshot)
        end

        # Update LLM operator contexts with current population state.
        _update_llm_contexts!(algorithm.mutation_ops, gen, algorithm.generations,
                               fitnesses, genomes)

        # Build next generation.
        next_genomes = Vector{G}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        # Elitism: carry top individuals forward (using raw fitness ranking).
        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = deepcopy(genomes[i])
            next_fitnesses[i] = fitnesses[i]
        end

        # Fill the rest via selection + genetic operators. Track per-child
        # operator identity only when a RunLog is listening — zero overhead otherwise.
        op_track = log === nothing ? nothing :
            fill(:elitism, pop_size)  # elites default to :elitism; _breed overwrites
        _breed_next_generation!(next_genomes, genomes, selection_fitnesses,
                                 algorithm, rng, algorithm.elitism + 1;
                                 case_fitnesses=case_fitnesses,
                                 op_track=op_track)

        # Evaluate new individuals (skip elites which already have fitness).
        _parallel_evaluate!(next_fitnesses, next_genomes, problem.evaluator, bp,
                           (algorithm.elitism + 1):pop_size, algorithm.parallel)

        # Operator success/attempt tally for the most recent log entry.
        if log !== nothing && !isempty(entries(log))
            last = log.entries[end]
            for i in (algorithm.elitism + 1):pop_size
                name = op_track[i]
                last.operator_attempted[name] = get(last.operator_attempted, name, 0) + 1
                if isfinite(next_fitnesses[i])
                    last.operator_success[name] = get(last.operator_success, name, 0) + 1
                end
            end
        end

        # Update the all-time best across the whole run (not just the
        # current generation). This survives elitism loss.
        cur_best = argmin(next_fitnesses)
        if next_fitnesses[cur_best] < best_fitness_all_time
            best_fitness_all_time = next_fitnesses[cur_best]
            best_genome_all_time  = deepcopy(next_genomes[cur_best])
        end

        # Hall-of-Fame update: offer every individual from this generation.
        # push! filters by finiteness, dedup, and capacity internally.
        if hall_of_fame !== nothing
            for i in 1:pop_size
                push!(hall_of_fame, next_genomes[i], next_fitnesses[i])
            end
        end

        genomes = next_genomes
        fitnesses = next_fitnesses

        # Periodic checkpoint — writes the just-completed generation's state.
        if checkpoint_every > 0 && checkpoint_path !== nothing &&
           gen % checkpoint_every == 0
            _save_ckpt(genomes, fitnesses, rng, best_genome_all_time,
                       best_fitness_all_time, fitness_history, mean_history,
                       time() - t0, gen, algorithm, checkpoint_path)
        end
    end

    # Final sort.
    order = sortperm(fitnesses)
    genomes = genomes[order]
    fitnesses = fitnesses[order]

    wall_time = time() - t0

    # Prefer the all-time best over the final-population best.
    final_best_genome  = best_fitness_all_time < fitnesses[1] ?
        best_genome_all_time : genomes[1]
    final_best_fitness = min(best_fitness_all_time, fitnesses[1])

    return GPResult{G}(
        final_best_genome,
        final_best_fitness,
        genomes,
        fitness_history,
        mean_history,
        algorithm.generations,
        wall_time,
        final_best_fitness < algorithm.convergence_threshold,
        hall_of_fame
    )
end


# =============================================================================
# Behavioral initialization (MAP-Elites-style diverse seeding)
# =============================================================================

"""
    behavioral_initialize(state, evaluator, fingerprint_fn, distance_fn,
                          target_size; kwargs...) -> Vector{ExprGenome}

Generate a behaviorally diverse initial population using a MAP-Elites-style
procedure:

1. Generate `pool_size` random programs
2. Evaluate each, discard crashed/timed-out programs (fitness == Inf)
3. Compute behavioral fingerprints
4. Greedily bin by fingerprint distance (threshold-based clustering)
5. Select the best-fitness representative from each bin
6. Return `target_size` genomes drawn from distinct bins

This replaces the standard random initialization with a pool that
guarantees behavioral diversity — programs that *do different things*,
not just look different syntactically.

# Arguments
- `state::GenState`: the genome state (function set, variable types, RNG)
- `evaluator::AbstractEvaluator`: for fitness evaluation
- `fingerprint_fn`: `genome -> fingerprint` (any type; same as BehavioralSpeciation)
- `distance_fn`: `(fp_a, fp_b) -> Float64` (same as BehavioralSpeciation)
- `target_size::Int`: desired population size

# Keyword Arguments
- `pool_size::Int = target_size * 50`: number of random programs to generate
- `bin_threshold::Float64 = 0.1`: distance below which two fingerprints
  are considered the same behavior
- `body_generator = nothing`: optional `(state) -> Vector{Expr}` for
  custom random program generation. Default: 3-7 random statements via
  `create_random_statement`.
- `bloat_penalty::Float64 = 0.0`: bloat penalty for evaluation
- `parallel::Bool = false`: use threads for evaluation
- `verbose::Bool = false`: print progress

# Returns
A `Vector{ExprGenome}` of length `target_size` with diverse behaviors.
If fewer than `target_size` distinct behaviors are found in the pool,
remaining slots are filled with the best-fitness programs from the pool.
"""
function behavioral_initialize(
    state::GenState,
    evaluator::AbstractEvaluator,
    fingerprint_fn,
    distance_fn,
    target_size::Int;
    pool_size::Int = target_size * 50,
    bin_threshold::Float64 = 0.1,
    body_generator = nothing,
    bloat_penalty::Float64 = 0.0,
    parallel::Bool = false,
    verbose::Bool = false
)::Vector{ExprGenome}

    # 1. Generate pool of random programs.
    verbose && println("behavioral_initialize: generating $pool_size random programs...")
    verbose && flush(stdout)
    pool = Vector{ExprGenome}(undef, pool_size)
    for i in 1:pool_size
        body = if body_generator !== nothing
            body_generator(state)
        else
            n_stmts = rand(state.rng, 3:7)
            Expr[create_random_statement(state; depth=2) for _ in 1:n_stmts]
        end
        pool[i] = ExprGenome(body, state)
    end

    # 2. Evaluate all programs.
    verbose && println("behavioral_initialize: evaluating pool...")
    verbose && flush(stdout)
    fitnesses = fill(Inf, pool_size)
    _parallel_evaluate!(fitnesses, pool, evaluator, bloat_penalty,
                        1:pool_size, parallel)

    # Filter to programs that didn't crash (finite fitness).
    valid_indices = [i for i in 1:pool_size if isfinite(fitnesses[i])]
    verbose && println("behavioral_initialize: $(length(valid_indices))/$pool_size programs survived evaluation")
    verbose && flush(stdout)

    if isempty(valid_indices)
        @warn "behavioral_initialize: no programs survived evaluation, falling back to random init"
        return pool[1:min(target_size, pool_size)]
    end

    # 3. Compute fingerprints.
    verbose && println("behavioral_initialize: computing fingerprints...")
    verbose && flush(stdout)
    fingerprints = Vector{Any}(undef, length(valid_indices))
    for (j, i) in enumerate(valid_indices)
        fingerprints[j] = try
            fingerprint_fn(pool[i])
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
    end

    # Filter out fingerprint failures.
    fp_ok = [(valid_indices[j], fingerprints[j]) for j in 1:length(valid_indices)
             if fingerprints[j] !== nothing]
    verbose && println("behavioral_initialize: $(length(fp_ok)) programs fingerprinted successfully")
    verbose && flush(stdout)

    if isempty(fp_ok)
        @warn "behavioral_initialize: all fingerprints failed, falling back to best-fitness selection"
        order = sortperm(fitnesses)
        return pool[order[1:min(target_size, pool_size)]]
    end

    # 4. Greedy binning by fingerprint distance.
    # Each bin stores (genome_index, fingerprint, fitness).
    bins = Vector{Vector{Tuple{Int, Any, Float64}}}()

    # Shuffle to avoid ordering bias.
    shuffled = fp_ok[sortperm(rand(state.rng, length(fp_ok)))]

    for (idx, fp) in shuffled
        fit = fitnesses[idx]
        placed = false
        for bin in bins
            rep_fp = bin[1][2]  # fingerprint of first member (representative)
            d = try
                distance_fn(fp, rep_fp)
            catch e
                e isa InterruptException && rethrow()
                Inf
            end
            if d < bin_threshold
                push!(bin, (idx, fp, fit))
                placed = true
                break
            end
        end
        if !placed
            push!(bins, [(idx, fp, fit)])
        end
    end

    verbose && println("behavioral_initialize: $(length(bins)) distinct behavior bins found")
    verbose && flush(stdout)

    # 5. Select best-fitness representative from each bin.
    representatives = Tuple{Int, Float64}[]  # (genome_index, fitness)
    for bin in bins
        best_in_bin = argmin(entry -> entry[3], bin)
        push!(representatives, (best_in_bin[1], best_in_bin[3]))
    end

    # Sort by fitness (best first) so we keep the best bins if we have more than target_size.
    sort!(representatives, by=r -> r[2])

    # 6. Build output population.
    result = Vector{ExprGenome}(undef, target_size)
    n_from_bins = min(length(representatives), target_size)
    for i in 1:n_from_bins
        result[i] = deepcopy(pool[representatives[i][1]])
    end

    # Fill remaining slots with best-fitness programs from the pool (may duplicate behaviors).
    if n_from_bins < target_size
        remaining_order = sortperm(fitnesses)
        fill_idx = n_from_bins + 1
        for i in remaining_order
            fill_idx > target_size && break
            isfinite(fitnesses[i]) || continue
            result[fill_idx] = deepcopy(pool[i])
            fill_idx += 1
        end
        # If still not enough (very few valid programs), pad with random.
        while fill_idx <= target_size
            body = if body_generator !== nothing
                body_generator(state)
            else
                Expr[create_random_statement(state; depth=2) for _ in 1:rand(state.rng, 3:7)]
            end
            result[fill_idx] = ExprGenome(body, state)
            fill_idx += 1
        end
    end

    verbose && println("behavioral_initialize: returning $target_size programs " *
                       "($n_from_bins from distinct bins, $(target_size - n_from_bins) filled)")
    verbose && flush(stdout)

    return result
end


# =============================================================================
# IslandModel solver
# =============================================================================

"""
    solve(problem::GPProblem{G,E}, algorithm::IslandModel; verbose=false, callback=nothing) -> GPResult{G}

Run an island-model evolution with multiple independent populations and periodic migration.
Islands run sequentially (no threading). Migration uses a ring topology.

Returns a `GPResult` containing the global best genome across all islands.
"""
function solve(problem::GPProblem{G,E},
               algorithm::IslandModel;
               verbose::Bool = false,
               callback = nothing,
               log::Union{Nothing, RunLog} = nothing,
               auto_addprocs::Bool = false,
               auto_rmprocs::Bool = false,
               initial_population::Union{Nothing, Vector{<:AbstractGenome}} = nothing) where {G,E}
    if initial_population !== nothing && algorithm.distributed
        throw(ArgumentError(
            "IslandModel warm-start (`initial_population`) is currently " *
            "supported only for the sequential path (`distributed=false`). " *
            "For distributed runs, bootstrap each worker's population manually."))
    end

    # Dispatch to distributed solvers if requested
    if algorithm.distributed && !algorithm.async
        return _distributed_sync_solve(problem, algorithm;
                                        verbose=verbose, callback=callback, log=log,
                                        auto_addprocs=auto_addprocs,
                                        auto_rmprocs=auto_rmprocs)
    elseif algorithm.distributed && algorithm.async
        return _distributed_async_solve(problem, algorithm;
                                         verbose=verbose, callback=callback, log=log,
                                         auto_addprocs=auto_addprocs,
                                         auto_rmprocs=auto_rmprocs)
    end

    # --- Sequential in-process mode (original behavior) ---
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    alg = algorithm.island_algorithm
    _validate_ops(alg.mutation_ops, alg.crossover_ops, G)

    # GraphGenome maintains a process-global innovation counter that all
    # islands share in sequential mode. Reset it once before island setup so
    # innovations are coherent across islands (required for NEAT crossover
    # when migrants are swapped in).
    if G === GraphGenome
        reset_innovation_counter!()
    end

    n = algorithm.n_islands
    pop_size = alg.pop_size
    bp = alg.bloat_penalty

    if initial_population !== nothing
        _validate_initial_population(initial_population, n * pop_size, G)
    end

    # Initialize n independent islands, each with its own per-island state
    # (GenState for ExprGenome, TreeGenomeContext for TreeGenome, ...).
    # Both state types expose `.rng` so the island loop can read state.rng
    # uniformly. Typed Vector{Any} to allow heterogeneous state without
    # coupling the IslandModel path to one genome's state representation.
    island_genomes = Vector{Vector{G}}(undef, n)
    island_states = Vector{Any}(undef, n)
    island_fitnesses = Vector{Vector{Float64}}(undef, n)
    island_species = Vector{Any}(undef, n)

    for i in 1:n
        island_rng_seed = rand(rng, UInt64)
        island_rng = Random.MersenneTwister(island_rng_seed)
        pop = _initialize_population(problem, alg, island_rng)
        # Swap in the i-th slice of the warm-start vector if provided. State
        # carrier from _initialize_population is retained so operators that
        # consult it (e.g., ExprGenome's GenState) continue to work.
        if initial_population !== nothing
            slice_lo = (i - 1) * pop_size + 1
            slice_hi = i * pop_size
            seeded = Vector{G}(deepcopy.(initial_population[slice_lo:slice_hi]))
            island_genomes[i] = seeded
            island_states[i] = pop[2]
        else
            island_genomes[i], island_states[i] = pop
        end
        island_fitnesses[i] = fill(Inf, pop_size)
        island_species[i] = _init_species_state(alg.speciation)
    end

    # Evaluate initial populations.
    for i in 1:n
        for j in 1:pop_size
            island_fitnesses[i][j] = _evaluate_with_penalty(
                island_genomes[i][j], problem.evaluator, bp)
        end
    end

    # Track global best.
    global_best_genome = deepcopy(island_genomes[1][1])
    global_best_fitness = Inf
    fitness_history = Float64[]
    mean_history = Float64[]

    t0 = time()

    for gen in 1:alg.generations
        for isle in 1:n
            genomes = island_genomes[isle]
            fitnesses = island_fitnesses[isle]
            state = island_states[isle]
            species_st = island_species[isle]

            # Sort by fitness.
            order = sortperm(fitnesses)
            genomes = genomes[order]
            fitnesses = fitnesses[order]

            # Apply speciation.
            selection_fitnesses = _apply_speciation!(genomes, fitnesses,
                                                      alg.speciation, species_st, state.rng)
            # (IslandModel log aggregates across all islands; per-island snapshots
            #  are not plumbed through here in F.0.)

            # Case fitnesses for lexicase-family selection (per-island).
            case_fitnesses_isle = needs_cases(alg.selection) ?
                _compute_case_fitnesses(genomes, problem.evaluator, alg.parallel) :
                nothing

            # Update LLM operator contexts with this island's population state.
            _update_llm_contexts!(alg.mutation_ops, gen, alg.generations,
                                   fitnesses, genomes)

            # Build next generation.
            next_genomes = Vector{G}(undef, pop_size)
            next_fitnesses = fill(Inf, pop_size)

            # Elitism.
            for i in 1:min(alg.elitism, pop_size)
                next_genomes[i] = deepcopy(genomes[i])
                next_fitnesses[i] = fitnesses[i]
            end

            # Fill rest via selection + genetic operators.
            _breed_next_generation!(next_genomes, genomes, selection_fitnesses,
                                     alg, state.rng, alg.elitism + 1;
                                     case_fitnesses=case_fitnesses_isle)

            # Evaluate new individuals.
            for i in (alg.elitism + 1):pop_size
                next_fitnesses[i] = _evaluate_with_penalty(
                    next_genomes[i], problem.evaluator, bp)
            end

            island_genomes[isle] = next_genomes
            island_fitnesses[isle] = next_fitnesses
        end

        # Update global best across all islands.
        for isle in 1:n
            best_idx = argmin(island_fitnesses[isle])
            if island_fitnesses[isle][best_idx] < global_best_fitness
                global_best_fitness = island_fitnesses[isle][best_idx]
                global_best_genome = deepcopy(island_genomes[isle][best_idx])
            end
        end

        push!(fitness_history, global_best_fitness)
        # Compute global mean across all islands.
        all_fits = reduce(vcat, island_fitnesses)
        finite_fits = filter(isfinite, all_fits)
        mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
        push!(mean_history, mean_fit)

        if verbose
            println("Generation $gen: global_best=$(round(global_best_fitness, digits=6)), mean=$(round(mean_fit, digits=6))")
        end

        if callback !== nothing
            callback(gen, global_best_fitness, global_best_genome)
        end

        if log !== nothing
            all_genomes_so_far = reduce(vcat, island_genomes)
            record!(log, gen, all_fits, all_genomes_so_far, time() - t0)
        end

        # Migration: ring topology, every migration_interval generations.
        if gen % algorithm.migration_interval == 0
            _migrate!(island_genomes, island_fitnesses, algorithm, rng;
                      island_states=island_states)
        end
    end

    # Collect final population from all islands, sorted by fitness.
    all_genomes = reduce(vcat, island_genomes)
    all_fitnesses = reduce(vcat, island_fitnesses)
    order = sortperm(all_fitnesses)
    all_genomes = all_genomes[order]
    all_fitnesses = all_fitnesses[order]

    wall_time = time() - t0

    return GPResult{G}(
        global_best_genome,
        global_best_fitness,
        all_genomes,
        fitness_history,
        mean_history,
        alg.generations,
        wall_time,
        global_best_fitness < alg.convergence_threshold
    )
end

"""
    _naturalize_migrant(genome, dest_state)

Rebind a migrant genome's GenState to the destination island's shared
GenState. This ensures all genomes on an island share the same RNG,
preventing dual-RNG mutation when the migrant's source-island RNG
diverges from the destination's.

No-op for genome types that don't carry a GenState (e.g., AntGenome,
GraphGenome, TreeGenome).
"""
_naturalize_migrant(genome, dest_state) = genome
_naturalize_migrant(genome::ExprGenome, dest_state::GenState) = ExprGenome(genome.body, dest_state)

"""
    _migrate!(island_genomes, island_fitnesses, algorithm, rng; island_states=nothing)

Perform migration using the algorithm's topology. Sends the top
`migration_size` individuals from each island to its topology-determined
destinations, replacing the worst individuals on those destinations.

When `island_states` is provided, migrant genomes are re-bound to the
destination island's GenState to prevent RNG contamination.
"""
function _migrate!(island_genomes, island_fitnesses, algorithm::IslandModel, rng::AbstractRNG;
                   island_states=nothing)
    n = algorithm.n_islands
    ms = algorithm.migration_size

    # Collect emigrants from each island (top ms by fitness) with their fitnesses.
    emigrants = Vector{Vector{Any}}(undef, n)
    emigrant_fitnesses = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        order = sortperm(island_fitnesses[i])
        k = min(ms, length(order))
        emigrants[i] = [deepcopy(island_genomes[i][order[j]]) for j in 1:k]
        emigrant_fitnesses[i] = [island_fitnesses[i][order[j]] for j in 1:k]
    end

    # Send emigrants to topology-determined destinations, replacing worst.
    # Preserve source fitness — all islands share the same evaluator.
    for i in 1:n
        destinations = migration_targets(algorithm.topology, i, n, rng)
        for dest in destinations
            order = sortperm(island_fitnesses[dest], rev=true)  # worst first
            incoming = emigrants[i]
            for (k, genome) in enumerate(incoming)
                if k <= length(order)
                    worst_idx = order[k]
                    if island_states !== nothing
                        island_genomes[dest][worst_idx] = _naturalize_migrant(genome, island_states[dest])
                    else
                        island_genomes[dest][worst_idx] = genome
                    end
                    island_fitnesses[dest][worst_idx] = emigrant_fitnesses[i][k]
                end
            end
        end
    end
end
