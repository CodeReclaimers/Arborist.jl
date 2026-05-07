# distributed_island.jl — Distributed worker support for island models.
#
# When IslandModel(distributed=true), each island runs in its own worker
# process via Distributed.jl. This eliminates @eval contention between
# islands: each process has its own compilation lock, world age counter,
# and module-level state.

using Distributed
#
# GraphGenome (NEAT) uses a process-local innovation counter. In
# distributed mode, each worker initializes its counter to a disjoint range
# (`init_innovation_range!((island_id - 1) * INNOVATION_STRIDE)`) so that
# NEAT crossover on migrants does not align structurally unrelated genes
# under the same innovation number. The cost of this scheme is that
# independent identical structural mutations on different workers receive
# different innovation numbers — they are treated as disjoint genes by
# crossover rather than matching. In practice this is the standard trade
# distributed NEAT implementations make.

# =============================================================================
# Worker-local island state
# =============================================================================

"""
    IslandState{G}

Mutable state for a single island running on a worker process.
Holds everything needed to evolve one generation independently.

The `state` field carries the per-island state object returned by
`_initialize_population` — a `GenState` for `ExprGenome`, or a
`TreeGenomeContext` for `TreeGenome`. It is typed `Any` rather than a
Union because (a) the island hot loop reads only `state.rng` and never
the state field directly, and (b) `_inject_migrants_local!` dispatches
`from_migrant(m, state)` on the runtime type without needing a static
narrowing.
"""
# Per-worker innovation ID range width for distributed GraphGenome runs.
# Default 10^9 supports up to one billion structural mutations per worker
# before adjacent ranges collide.
const INNOVATION_STRIDE = 10^9

mutable struct IslandState{G<:AbstractGenome}
    genomes::Vector{G}
    fitnesses::Vector{Float64}
    state::Any
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
    # GraphGenome's innovation counter is process-local; under distributed
    # execution each worker starts its own counter at 0. Without a disjoint
    # per-worker range, two workers would assign the same innovation number
    # to structurally unrelated mutations, and NEAT crossover on migrants
    # would then align unrelated genes. Set the offset BEFORE
    # `_initialize_population` so the initial connections get IDs in this
    # worker's range.
    if G === GraphGenome
        init_innovation_range!((island_id - 1) * INNOVATION_STRIDE)
    end

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

    # Update LLM operator contexts with this island's population state.
    _update_llm_contexts!(alg.mutation_ops, island.generation, alg.generations,
                           fitnesses, genomes)

    # Build next generation
    G = eltype(genomes)
    next_genomes = Vector{G}(undef, pop_size)
    next_fitnesses = fill(Inf, pop_size)

    for i in 1:min(alg.elitism, pop_size)
        next_genomes[i] = deepcopy(genomes[i])
        next_fitnesses[i] = fitnesses[i]
    end

    case_fitnesses = needs_cases(alg.selection) ?
        _compute_case_fitnesses(genomes, island.evaluator, alg.parallel) :
        nothing

    _breed_next_generation!(next_genomes, genomes, selection_fitnesses,
                             alg, rng, alg.elitism + 1;
                             case_fitnesses=case_fitnesses)

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
            island.fitnesses[worst_idx] = m.fitness  # preserve source fitness
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
# Async island support
# =============================================================================

"""
    AsyncStatusMessage

Status update sent from an async island worker to the coordinator.
Concrete struct (not NamedTuple) to ensure reliable cross-process serialization.
"""
struct AsyncStatusMessage
    island_id::Int
    generation::Int
    best_fitness::Float64
    mean_fitness::Float64
end

"""
    _nonblocking_put!(ch, val; timeout=0.5) -> Bool

Attempt to put `val` into channel `ch` without blocking the caller.
Spawns the `put!` in an @async task and waits up to `timeout` seconds.
Returns `true` if the put completed, `false` if it timed out (channel
full) or the channel was closed. Timed-out tasks resolve later when
space opens or the channel closes — they do not leak indefinitely.
"""
function _nonblocking_put!(ch, val; timeout::Float64=0.5)
    t = @async try
        put!(ch, val)
        true
    catch e
        e isa InvalidStateException && return false
        rethrow()
    end
    if timedwait(() -> istaskdone(t), timeout) === :timed_out
        return false
    end
    try
        return fetch(t)
    catch
        return false
    end
end

"""
    _run_island_async(id, inboxes, status_ch, n_islands, topology,
                      migration_interval, migration_size) -> NamedTuple

Run the full async evolution loop for island `id`. Called via remotecall
on a worker process. Each generation: evolve, report status, and at
migration intervals drain inbox and push emigrants to destinations.

All channel operations are non-blocking. Status and migration put! calls
use `_nonblocking_put!` which wraps the operation in an @async task with
a short timeout — if the channel is full, the data is dropped rather
than blocking the evolution loop. Inbox draining uses `isready` to avoid
blocking on `take!`.
"""
function _run_island_async(id::Int,
                           inboxes::Vector{RemoteChannel},
                           status_ch::RemoteChannel,
                           n_islands::Int,
                           topology::AbstractTopology,
                           migration_interval::Int,
                           migration_size::Int)
    island = _local_islands[id]::IslandState
    alg = island.algorithm
    rng = island.rng

    for gen in 1:alg.generations
        result = _evolve_one_gen_local!(id)

        # Report status (non-blocking; drop if channel full or closed).
        _nonblocking_put!(status_ch, AsyncStatusMessage(id, gen,
                                                         result.best_fitness,
                                                         result.mean_fitness))

        # Migration at intervals
        if gen % migration_interval == 0
            # Drain inbox (non-blocking)
            my_inbox = inboxes[id]
            while isready(my_inbox)
                try
                    migrants = take!(my_inbox)
                    _inject_migrants_local!(id, migrants)
                catch e
                    e isa InvalidStateException && break
                    rethrow()
                end
            end

            # Push emigrants to destinations (non-blocking; drop if full).
            emigrants = _get_migrants_local(id, migration_size)
            destinations = migration_targets(topology, id, n_islands, rng)
            for dest in destinations
                _nonblocking_put!(inboxes[dest], emigrants)
            end
        end
    end

    return _get_final_state_local(id)
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
                                  log::Union{Nothing, RunLog} = nothing,
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

            if log !== nothing
                # Aggregate stats from the per-worker result summaries. No global
                # population materialization here (workers retain state) — so
                # `unique_structures` is reported as 0 via an empty genome vector.
                all_fits_gen = Float64[]
                for r in results
                    push!(all_fits_gen, r.best_fitness)
                    push!(all_fits_gen, r.mean_fitness)
                end
                record!(log, gen, all_fits_gen, Any[], time() - t0)
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
            _converged(best_state.best_fitness, alg.convergence_threshold)
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

# =============================================================================
# Distributed asynchronous solve
# =============================================================================

"""
    _distributed_async_solve(problem, algorithm; kwargs...) -> GPResult

Asynchronous distributed island model: each island evolves independently
on its own worker process. Migration happens peer-to-peer via RemoteChannels.
The coordinator monitors progress and gathers results when all islands finish.

Unlike sync mode, there is no lock-step generation coordination. Each island
runs at its own pace. This provides natural load balancing and implicit
parsimony pressure (smaller/faster programs get more evolutionary turns).

The `fitness_history` in the returned GPResult contains one entry per status
message received from workers, not one per generation. This is inherent to
async — there is no well-defined global generation counter.
"""
function _distributed_async_solve(problem::GPProblem{G,E}, algorithm::IslandModel;
                                   verbose::Bool=false, callback=nothing,
                                   log::Union{Nothing, RunLog} = nothing,
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
    if eval_mod !== @__MODULE__
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

        # Create bounded migration inboxes (one per island, on the island's worker).
        # _nonblocking_put! ensures a fast island never blocks on a slow island's
        # full inbox — data is dropped rather than blocking the evolution loop.
        inbox_capacity = 4
        inboxes = RemoteChannel[
            RemoteChannel(() -> Channel{Vector{MigrantGenome}}(inbox_capacity), w)
            for (i, w) in enumerate(workers_used)
        ]

        # Status channel: bounded, coordinator polls it.
        # _nonblocking_put! drops status messages if channel is full.
        status_ch = RemoteChannel(() -> Channel{AsyncStatusMessage}(n * 10))

        t0 = time()

        # Launch async evolution on each worker
        island_futures = [
            remotecall(_run_island_async, w, i,
                       inboxes, status_ch,
                       n, algorithm.topology,
                       algorithm.migration_interval, algorithm.migration_size)
            for (i, w) in enumerate(workers_used)
        ]

        # Monitor progress until all islands complete
        global_best_fitness = Inf
        global_best_island = 1
        fitness_history = Float64[]
        mean_history = Float64[]
        island_gen = zeros(Int, n)
        island_best = fill(Inf, n)

        all_done = false
        while !all_done
            # Drain all available status messages
            while isready(status_ch)
                msg = take!(status_ch)
                island_gen[msg.island_id] = msg.generation
                island_best[msg.island_id] = msg.best_fitness

                if msg.best_fitness < global_best_fitness
                    global_best_fitness = msg.best_fitness
                    global_best_island = msg.island_id
                end

                # Record history entry per status message
                push!(fitness_history, global_best_fitness)
                push!(mean_history, msg.mean_fitness)
            end

            if verbose && any(g -> g > 0, island_gen)
                min_gen = minimum(island_gen)
                max_gen = maximum(island_gen)
                elapsed = round(time() - t0, digits=1)
                println("Progress: gens=$(min_gen)-$(max_gen), " *
                        "global_best=$(round(global_best_fitness, digits=6)), " *
                        "elapsed=$(elapsed)s")
                flush(stdout)
            end

            if log !== nothing && any(g -> g > 0, island_gen)
                # Async log: aggregate status messages since last tick. We record
                # once per monitor poll using the status-message bests/means as
                # the fitness vector. Genome list is empty (workers own state).
                fits_snapshot = [f for f in island_best if isfinite(f)]
                record!(log, minimum(island_gen), fits_snapshot, Any[], time() - t0)
            end

            if callback !== nothing && any(g -> g > 0, island_gen)
                callback(minimum(island_gen), global_best_fitness, nothing)
            end

            all_done = all(isready(f) for f in island_futures)
            if !all_done
                sleep(0.1)
            end
        end

        # Drain remaining status messages
        while isready(status_ch)
            msg = take!(status_ch)
            if msg.best_fitness < global_best_fitness
                global_best_fitness = msg.best_fitness
                global_best_island = msg.island_id
            end
            push!(fitness_history, global_best_fitness)
            push!(mean_history, msg.mean_fitness)
        end

        # Collect final states
        final_states = [fetch(f) for f in island_futures]

        # Ensure we have at least one history entry
        if isempty(fitness_history)
            push!(fitness_history, global_best_fitness)
            push!(mean_history, Inf)
        end

        # Merge final populations
        all_genomes = reduce(vcat, [s.genomes for s in final_states])
        all_fitnesses = reduce(vcat, [s.fitnesses for s in final_states])
        order = sortperm(all_fitnesses)
        all_genomes = all_genomes[order]
        all_fitnesses = all_fitnesses[order]

        best_state = final_states[global_best_island]
        wall_time = time() - t0

        # Cleanup
        cleanup_futures = [remotecall(_cleanup_island_local, w, i)
                           for (i, w) in enumerate(workers_used)]
        for f in cleanup_futures
            fetch(f)
        end

        # Close channels
        close(status_ch)
        for inbox in inboxes
            close(inbox)
        end

        return GPResult{G}(
            best_state.best_genome,
            best_state.best_fitness,
            all_genomes,
            fitness_history,
            mean_history,
            alg.generations,
            wall_time,
            _converged(best_state.best_fitness, alg.convergence_threshold)
        )
    finally
        if auto_rmprocs && !isempty(added_pids)
            rmprocs(added_pids)
        end
    end
end
