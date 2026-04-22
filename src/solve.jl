"""
    solve(problem::GPProblem{G,E}, algorithm::GeneticProgramming; verbose=false, callback=nothing, log=nothing) -> GPResult{G}

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
"""
function solve(problem::GPProblem{G,E},
               algorithm::GeneticProgramming;
               verbose::Bool = false,
               callback = nothing,
               log::Union{Nothing, RunLog} = nothing) where {G,E}
    # Create a dedicated RNG from the seed, or use the default RNG.
    # All stochastic operations use this RNG explicitly — no global rand().
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    pop = _initialize_population(problem, algorithm, rng)
    return _run_evolution!(pop, problem, algorithm, rng;
                           verbose=verbose, callback=callback, log=log)
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
    _tournament_select(fitnesses, tournament_size, rng) -> Int

Perform tournament selection. Returns the index of the selected individual.
"""
function _tournament_select(fitnesses::Vector{Float64}, tournament_size::Int, rng::AbstractRNG)
    n = length(fitnesses)
    best_idx = rand(rng, 1:n)
    for _ in 2:tournament_size
        idx = rand(rng, 1:n)
        if fitnesses[idx] < fitnesses[best_idx]
            best_idx = idx
        end
    end
    return best_idx
end

"""
    _breed_next_generation!(next_genomes, genomes, selection_fitnesses, alg, rng, start_idx)

Fill `next_genomes[start_idx:end]` via tournament selection and genetic
operators (crossover, mutation, or copy). Shared across all solve paths.

Crossover and mutation dispatch through operator objects from `alg.crossover_ops`
and `alg.mutation_ops`. Genome types that use direct dispatch (AntGenome,
GraphGenome) provide fallback methods that ignore the operator argument.
"""
function _breed_next_generation!(next_genomes::Vector{G},
                                  genomes::Vector{G},
                                  selection_fitnesses::Vector{Float64},
                                  alg::GeneticProgramming,
                                  rng::AbstractRNG,
                                  start_idx::Int) where G
    pop_size = length(next_genomes)
    t_size = alg.selection.tournament_size
    idx = start_idx
    while idx <= pop_size
        r = rand(rng)
        if r < alg.crossover_rate && idx + 1 <= pop_size
            p1 = _tournament_select(selection_fitnesses, t_size, rng)
            p2 = _tournament_select(selection_fitnesses, t_size, rng)
            op = rand(rng, alg.crossover_ops)
            (c1, c2) = crossover(op, genomes[p1], genomes[p2], rng)
            next_genomes[idx] = c1
            next_genomes[idx + 1] = c2
            idx += 2
        elseif r < alg.crossover_rate + alg.mutation_rate
            p_idx = _tournament_select(selection_fitnesses, t_size, rng)
            op = rand(rng, alg.mutation_ops)
            _set_parent_context!(alg.mutation_ops, p_idx, selection_fitnesses)
            next_genomes[idx] = mutate(op, genomes[p_idx], rng)
            idx += 1
        else
            p_idx = _tournament_select(selection_fitnesses, t_size, rng)
            next_genomes[idx] = deepcopy(genomes[p_idx])
            idx += 1
        end
    end
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
                         log::Union{Nothing, RunLog} = nothing) where {G,E}
    genomes, state = pop
    pop_size = algorithm.pop_size
    fitnesses = fill(Inf, pop_size)
    bp = algorithm.bloat_penalty

    # Evaluate initial population (with bloat penalty).
    _parallel_evaluate!(fitnesses, genomes, problem.evaluator, bp, 1:pop_size, algorithm.parallel)

    # Initialize speciation state.
    species_state = _init_species_state(algorithm.speciation)

    fitness_history = Float64[]
    mean_history = Float64[]

    t0 = time()

    for gen in 1:algorithm.generations
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

        # Fill the rest via tournament selection + genetic operators.
        _breed_next_generation!(next_genomes, genomes, selection_fitnesses,
                                 algorithm, rng, algorithm.elitism + 1)

        # Evaluate new individuals (skip elites which already have fitness).
        _parallel_evaluate!(next_fitnesses, next_genomes, problem.evaluator, bp,
                           (algorithm.elitism + 1):pop_size, algorithm.parallel)

        genomes = next_genomes
        fitnesses = next_fitnesses
    end

    # Final sort.
    order = sortperm(fitnesses)
    genomes = genomes[order]
    fitnesses = fitnesses[order]

    wall_time = time() - t0

    return GPResult{G}(
        genomes[1],
        fitnesses[1],
        genomes,
        fitness_history,
        mean_history,
        algorithm.generations,
        wall_time,
        fitnesses[1] < algorithm.convergence_threshold
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
               auto_rmprocs::Bool = false) where {G,E}
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
        island_genomes[i], island_states[i] = pop
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

            # Fill rest via tournament selection + genetic operators.
            _breed_next_generation!(next_genomes, genomes, selection_fitnesses,
                                     alg, state.rng, alg.elitism + 1)

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
