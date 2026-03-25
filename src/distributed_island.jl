# distributed_island.jl — Distributed worker support for island models.
#
# When IslandModel(distributed=true), each island runs in its own worker
# process via Distributed.jl. This eliminates @eval contention between
# islands: each process has its own compilation lock, world age counter,
# and module-level state.

# =============================================================================
# Worker-local island state
# =============================================================================

"""
    IslandState{G}

Mutable state for a single island running on a worker process.
Holds everything needed to evolve one generation independently.
"""
mutable struct IslandState{G<:AbstractGenome}
    genomes::Vector{G}
    fitnesses::Vector{Float64}
    state::GenState
    species_state::Any
    evaluator::AbstractEvaluator
    algorithm::GeneticProgramming
    rng::AbstractRNG
    generation::Int
    best_fitness::Float64
    best_genome::G
end

# Worker-local island storage. Each worker holds at most one island per solve.
const _local_islands = Dict{Int, Any}()

"""
    _init_island_local(problem, alg, seed, island_id) -> Int

Initialize an island on this worker. Stores the state in the worker-local
dict and returns the island_id.
"""
function _init_island_local(problem::GPProblem{G,E}, alg::GeneticProgramming,
                            seed::UInt64, island_id::Int) where {G,E}
    rng = Random.MersenneTwister(seed)
    pop = _initialize_population(problem, alg, rng)
    genomes, gen_state = pop
    fitnesses = fill(Inf, alg.pop_size)
    bp = alg.bloat_penalty

    for i in 1:alg.pop_size
        fitnesses[i] = _evaluate_with_penalty(genomes[i], problem.evaluator, bp)
    end

    species_state = _init_species_state(alg.speciation)

    best_idx = argmin(fitnesses)
    island = IslandState{G}(genomes, fitnesses, gen_state, species_state,
                             problem.evaluator, alg, rng, 0,
                             fitnesses[best_idx], deepcopy(genomes[best_idx]))
    _local_islands[island_id] = island
    return island_id
end

"""
    _evolve_one_gen_local!(id) -> NamedTuple

Run one generation of evolution on the island stored at `id`.
Returns a status tuple for the coordinator.
"""
function _evolve_one_gen_local!(id::Int)
    island = _local_islands[id]::IslandState
    alg = island.algorithm
    rng = island.rng
    pop_size = alg.pop_size
    bp = alg.bloat_penalty
    genomes = island.genomes
    fitnesses = island.fitnesses

    # Sort by fitness
    order = sortperm(fitnesses)
    genomes = genomes[order]
    fitnesses = fitnesses[order]

    # Apply speciation
    selection_fitnesses = _apply_speciation!(genomes, fitnesses,
                                              alg.speciation, island.species_state, rng)

    t_size = alg.selection isa TournamentSelection ?
             alg.selection.tournament_size : alg.tournament_size

    # Build next generation
    G = eltype(genomes)
    next_genomes = Vector{G}(undef, pop_size)
    next_fitnesses = fill(Inf, pop_size)

    for i in 1:min(alg.elitism, pop_size)
        next_genomes[i] = deepcopy(genomes[i])
        next_fitnesses[i] = fitnesses[i]
    end

    idx = alg.elitism + 1
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
            child = mutate(op, genomes[p_idx], rng)
            next_genomes[idx] = child
            idx += 1
        else
            p_idx = _tournament_select(selection_fitnesses, t_size, rng)
            next_genomes[idx] = deepcopy(genomes[p_idx])
            idx += 1
        end
    end

    # Evaluate new individuals
    for i in (alg.elitism + 1):pop_size
        next_fitnesses[i] = _evaluate_with_penalty(next_genomes[i], island.evaluator, bp)
    end

    island.genomes = next_genomes
    island.fitnesses = next_fitnesses
    island.generation += 1

    best_idx = argmin(next_fitnesses)
    if next_fitnesses[best_idx] < island.best_fitness
        island.best_fitness = next_fitnesses[best_idx]
        island.best_genome = deepcopy(next_genomes[best_idx])
    end

    finite_fits = filter(isfinite, next_fitnesses)
    mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)

    return (best_fitness=island.best_fitness, mean_fitness=mean_fit,
            generation=island.generation)
end

"""
    _get_migrants_local(id, n) -> Vector{MigrantGenome}

Extract the top `n` individuals from island `id` as MigrantGenome for
cross-process transfer.
"""
function _get_migrants_local(id::Int, n::Int)
    island = _local_islands[id]::IslandState
    order = sortperm(island.fitnesses)
    k = min(n, length(order))
    return [to_migrant(island.genomes[order[i]], island.fitnesses[order[i]]) for i in 1:k]
end

"""
    _inject_migrants_local!(id, migrants) -> Nothing

Replace worst individuals on island `id` with incoming migrants.
"""
function _inject_migrants_local!(id::Int, migrants::Vector{MigrantGenome})
    island = _local_islands[id]::IslandState
    order = sortperm(island.fitnesses, rev=true)  # worst first
    for (k, m) in enumerate(migrants)
        if k <= length(order)
            worst_idx = order[k]
            island.genomes[worst_idx] = from_migrant(m, island.state)
            island.fitnesses[worst_idx] = Inf  # will be re-evaluated next gen
        end
    end
    return nothing
end

"""
    _get_final_state_local(id) -> NamedTuple

Return final island state for result construction on the coordinator.
"""
function _get_final_state_local(id::Int)
    island = _local_islands[id]::IslandState
    return (genomes=island.genomes, fitnesses=island.fitnesses,
            best_genome=island.best_genome, best_fitness=island.best_fitness)
end

"""
    _cleanup_island_local(id) -> Nothing

Remove island state from worker-local storage.
"""
function _cleanup_island_local(id::Int)
    delete!(_local_islands, id)
    return nothing
end

# =============================================================================
# Worker acquisition and setup
# =============================================================================

"""
    _acquire_workers(n_islands; auto_add) -> (workers, added_pids)

Ensure enough workers are available for distributed island execution.
Returns the worker pids to use and the pids that were auto-added.
"""
function _acquire_workers(n_islands::Int; auto_add::Bool=false)
    available = Distributed.workers()
    # workers() returns [1] when no workers have been added (pid 1 is the main process)
    n_available = length(available) == 1 && available[1] == 1 ? 0 : length(available)

    if n_available >= n_islands
        return (available[1:n_islands], Int[])
    end

    if !auto_add
        error("distributed=true requires at least $n_islands workers, " *
              "but only $n_available are available. Either:\n" *
              "  1. Add workers: addprocs($n_islands; exeflags=\"--project=.\")\n" *
              "  2. Pass auto_addprocs=true to solve()")
    end

    needed = n_islands - n_available
    project = Base.active_project()
    @info "Adding $needed worker processes" project
    added = addprocs(needed; exeflags="--project=$project")
    # Load Arborist on newly added workers
    for w in added
        remotecall_fetch(w) do
            Core.eval(Main, :(using Arborist))
        end
    end
    all_workers = Distributed.workers()
    return (all_workers[1:n_islands], added)
end

"""
    setup_workers(script_path::AbstractString)

Convenience: load a user script on all workers. Equivalent to:

    @everywhere include(abspath("your_script.jl"))

Also ensures `Arborist` is loaded on all workers.
The primary documented approach is `@everywhere include(...)` directly.
"""
function setup_workers(script_path::AbstractString)
    ws = Distributed.workers()
    if length(ws) == 1 && ws[1] == 1
        @warn "No worker processes available. Call addprocs(n) first."
        return
    end
    path = abspath(script_path)
    for w in ws
        remotecall_fetch(w) do
            Core.eval(Main, :(using Arborist))
            Core.eval(Main, :(include($path)))
        end
    end
end

# =============================================================================
# Distributed synchronous solve
# =============================================================================

"""
    _distributed_sync_solve(problem, algorithm; kwargs...) -> GPResult

Synchronous distributed island model: each island runs on a separate worker,
coordinated generation-by-generation from the main process.
"""
function _distributed_sync_solve(problem::GPProblem{G,E}, algorithm::IslandModel;
                                  verbose::Bool=false, callback=nothing,
                                  auto_addprocs::Bool=false,
                                  auto_rmprocs::Bool=false) where {G,E}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    n = algorithm.n_islands
    alg = algorithm.island_algorithm

    workers_used, added_pids = _acquire_workers(n; auto_add=auto_addprocs)

    # Verify custom evaluator types are available on workers
    eval_type = typeof(problem.evaluator)
    eval_mod = parentmodule(eval_type)
    if eval_mod !== @__MODULE__  # not defined in Arborist
        type_sym = nameof(eval_type)
        for w in workers_used
            available = remotecall_fetch(isdefined, w, Main, type_sym)
            if !available
                error("Evaluator type $type_sym is not available on worker $w.\n" *
                      "Load your problem code on all workers with:\n" *
                      "  @everywhere include(\"your_script.jl\")")
            end
        end
    end

    try
        # Generate per-island seeds
        island_seeds = [rand(rng, UInt64) for _ in 1:n]

        # Initialize islands on workers
        init_futures = [remotecall(_init_island_local, w, problem, alg, seed, i)
                        for (i, (w, seed)) in enumerate(zip(workers_used, island_seeds))]
        for f in init_futures
            fetch(f)
        end

        global_best_fitness = Inf
        global_best_island = 1
        fitness_history = Float64[]
        mean_history = Float64[]
        t0 = time()

        for gen in 1:alg.generations
            # Evolve one generation on each worker in parallel
            gen_futures = [remotecall(_evolve_one_gen_local!, w, i)
                           for (i, w) in enumerate(workers_used)]
            results = fetch.(gen_futures)

            # Track global best
            for (i, r) in enumerate(results)
                if r.best_fitness < global_best_fitness
                    global_best_fitness = r.best_fitness
                    global_best_island = i
                end
            end

            push!(fitness_history, global_best_fitness)
            all_means = [r.mean_fitness for r in results]
            finite_means = filter(isfinite, all_means)
            global_mean = isempty(finite_means) ? Inf : sum(finite_means) / length(finite_means)
            push!(mean_history, global_mean)

            if verbose
                elapsed = round(time() - t0, digits=1)
                println("Generation $gen: global_best=$(round(global_best_fitness, digits=6)), mean=$(round(global_mean, digits=6)), elapsed=$(elapsed)s")
                flush(stdout)
            end

            if callback !== nothing
                callback(gen, global_best_fitness, nothing)
            end

            # Migration
            if gen % algorithm.migration_interval == 0
                _distributed_migrate!(workers_used, n, algorithm, rng)
            end
        end

        # Gather final populations from all workers
        final_futures = [remotecall(_get_final_state_local, w, i)
                         for (i, w) in enumerate(workers_used)]
        final_states = fetch.(final_futures)

        all_genomes = reduce(vcat, [s.genomes for s in final_states])
        all_fitnesses = reduce(vcat, [s.fitnesses for s in final_states])
        order = sortperm(all_fitnesses)
        all_genomes = all_genomes[order]
        all_fitnesses = all_fitnesses[order]

        # Get the actual best genome from the best island
        best_state = final_states[global_best_island]

        wall_time = time() - t0

        # Cleanup worker-local state
        cleanup_futures = [remotecall(_cleanup_island_local, w, i)
                           for (i, w) in enumerate(workers_used)]
        for f in cleanup_futures
            fetch(f)
        end

        return GPResult{G}(
            best_state.best_genome,
            best_state.best_fitness,
            all_genomes,
            fitness_history,
            mean_history,
            alg.generations,
            wall_time,
            best_state.best_fitness < 1.0
        )
    finally
        if auto_rmprocs && !isempty(added_pids)
            rmprocs(added_pids)
        end
    end
end

"""
    _distributed_migrate!(workers, n_islands, algorithm, rng)

Perform migration across distributed workers: fetch emigrants from each
island, compute topology targets, and inject into destinations.
"""
function _distributed_migrate!(workers::Vector{Int}, n_islands::Int,
                                algorithm::IslandModel, rng::AbstractRNG)
    ms = algorithm.migration_size

    # Fetch emigrants from each island in parallel
    emigrant_futures = [remotecall(_get_migrants_local, w, i, ms)
                        for (i, w) in enumerate(workers)]
    all_emigrants = fetch.(emigrant_futures)

    # Inject into topology-determined destinations
    inject_futures = Future[]
    for i in 1:n_islands
        destinations = migration_targets(algorithm.topology, i, n_islands, rng)
        for dest in destinations
            f = remotecall(_inject_migrants_local!, workers[dest], dest, all_emigrants[i])
            push!(inject_futures, f)
        end
    end
    for f in inject_futures
        fetch(f)
    end
end
