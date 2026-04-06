# Plan: Distributed Island Model via Worker Processes

## Motivation

The current `IslandModel` runs all islands sequentially in a single process. Every `@eval` shares the same global compilation lock and the same world age counter, which means:

1. **No parallel compilation** — `@eval` is serialized even with `Threads.@threads`
2. **`invokelatest` overhead** — every evolved function call goes through `Base.invokelatest` because the function was defined after the calling code was compiled
3. **Thread-local state workarounds** — side-effectful evaluators (bin packing, sorting, ant trail) need per-thread state to avoid data races, adding complexity for users writing custom evaluators

Running each island in its own Julia worker process solves all three problems: each process has its own compilation lock, its own world age, and its own module-level state. The user writes a plain evaluator with module-level `Ref` state (the existing simulator pattern), and the framework handles process isolation transparently.

## Literature Context

- **Synchronous** (all islands advance in lockstep, migrate at barriers): Simpler, reproducible, better solution quality in many benchmarks. The standard default in ECJ, DEAP, and the current Arborist.jl. Suffers from the straggler problem when evaluation times are heterogeneous — which is common in GP because tree sizes vary.

- **Asynchronous** (islands evolve independently, migration is opportunistic): Better wall-clock scalability, natural load balancing, and an implicit parsimony pressure in GP (smaller/faster programs get more evolutionary turns per unit time). Non-deterministic. PaGMO/PyGMO is built entirely around async. Fernandez et al. (EuroGP 2002) found async GP has competitive quality with significantly better wall-clock performance.

- **Topology interaction**: Sparse topologies (ring) pair well with synchronous migration. Dense topologies (random, complete) pair better with async because the high connectivity compensates for stale migrants. Ring migration with rare intervals was shown to help bypass local optima (Frahnow & Kotzing, PPSN 2018).

Both modes have legitimate use cases. We should support both.

## Design

### API

```julia
# Current API (preserved, backward compatible, in-process sequential)
algorithm = IslandModel(n_islands=4, island_algorithm=alg)
result = solve(problem, algorithm)

# New: distributed sync (each island in a worker process, lockstep generations)
algorithm = IslandModel(n_islands=4, island_algorithm=alg, distributed=true, async=false)
result = solve(problem, algorithm)

# New: distributed async (each island evolves independently)
algorithm = IslandModel(n_islands=4, island_algorithm=alg, distributed=true, async=true)
result = solve(problem, algorithm)
```

When `distributed=true`, the framework uses `Distributed.jl` worker processes. If `nworkers() < n_islands`, the framework calls `addprocs(n_islands - nworkers())` automatically (with a warning). If the user has already added workers with custom setup (e.g., `addprocs(4; exeflags="--project=.")`) the framework uses those.

### IslandModel Struct Changes

```julia
struct IslandModel <: AbstractEvolutionaryAlgorithm
    n_islands::Int
    island_algorithm::GeneticProgramming
    migration_interval::Int
    migration_size::Int
    topology::AbstractTopology       # NEW: ring, complete, random
    distributed::Bool                # NEW: use worker processes
    async::Bool                      # NEW: async evolution (requires distributed=true)
end
```

### Topology Types

```julia
abstract type AbstractTopology end

struct RingTopology <: AbstractTopology end          # current behavior
struct CompleteTopology <: AbstractTopology end       # all-to-all
struct RandomTopology <: AbstractTopology             # random target per event
    n_targets::Int                                    # how many neighbors to send to
end

"""Return the list of destination island indices for emigrants from island `i`."""
function migration_targets(t::RingTopology, i::Int, n_islands::Int)
    return [(i % n_islands) + 1]
end

function migration_targets(t::CompleteTopology, i::Int, n_islands::Int)
    return [j for j in 1:n_islands if j != i]
end

function migration_targets(t::RandomTopology, i::Int, n_islands::Int, rng)
    others = [j for j in 1:n_islands if j != i]
    return others[randperm(rng, length(others))[1:min(t.n_targets, length(others))]]
end
```

### Worker Setup

Each worker needs the Arborist module and the user's problem-specific code. The framework handles this in two stages:

**Stage 1: Module loading.** The framework ensures `Arborist` is available on all workers:
```julia
@everywhere using Arborist
```

**Stage 2: Problem setup.** The user's evaluator, function set, and primitives must be available on workers. Two approaches, both supported:

**(a) User loads code on workers manually (recommended for custom evaluators):**
```julia
using Distributed
addprocs(4; exeflags="--project=.")
@everywhere include("examples/sorting.jl")

# Now sort_get, sort_swap!, SortingEvaluator etc. exist on all workers
result = solve(problem, algorithm)
```

**(b) Framework ships serializable problem data (works for built-in evaluators):**
For `TableFitnessEvaluator`, `GraphEvaluator`, and other evaluators that don't require custom primitives, the framework serializes the problem to each worker. The evaluator struct, function set, and genome type are all serializable via Julia's native serialization.

The framework detects which case applies: if the evaluator type is defined in the `Arborist` module, approach (b) works automatically. If it's defined in user code (e.g., `BinPackingEvaluator`), the framework checks that `remotecall_fetch(() -> isdefined(Main, :BinPackingEvaluator), worker_pid)` returns `true` and errors with a clear message if not:

```
ERROR: Evaluator type BinPackingEvaluator is not available on worker 2.
Load your problem code on all workers with:
  @everywhere include("your_script.jl")
```

### Migration Data Format

Genomes are sent across process boundaries via Julia's native serialization, not the string-based `serialize`/`deserialize`. For `ExprGenome`, we send a lightweight struct:

```julia
struct MigrantGenome
    body::Vector{Expr}    # the program body — natively serializable
    fitness::Float64       # fitness on the source island
end
```

On the receiving end, the worker wraps the body in a fresh `ExprGenome` using its local `GenState`:

```julia
function receive_migrant(m::MigrantGenome, local_state::GenState)
    return ExprGenome(deepcopy(m.body), local_state)
end
```

For `AntGenome`, the migrant carries `program::Expr`. For `GraphGenome`, the migrant carries the node/connection gene vectors (already serializable).

### Synchronous Distributed Mode

The main process acts as coordinator. Each generation:

```
Main process                    Worker 1         Worker 2         Worker 3
     |                              |                |                |
     |--- evolve_one_gen! -------->|                |                |
     |--- evolve_one_gen! ---------|-------------->|                |
     |--- evolve_one_gen! ---------|----------------|------------->|
     |                              |                |                |
     |<-- (genomes, fitnesses) ----|                |                |
     |<-- (genomes, fitnesses) ----|----------------|                |
     |<-- (genomes, fitnesses) ----|----------------|----------------|
     |                              |                |                |
     | [migration: compute targets, send migrants]  |                |
     | [logging: best fitness, mean, etc.]          |                |
     |                              |                |                |
     |--- inject_migrants + evolve_one_gen -------->|  ...          |
```

Implementation via `remotecall_fetch`:

```julia
function _distributed_sync_solve(problem, algorithm, rng;
                                  verbose, callback,
                                  auto_addprocs=false, auto_rmprocs=false)
    workers = _acquire_workers(algorithm.n_islands;
                               auto_add=auto_addprocs)
    added_pids = ...  # track which pids we added, for auto_rmprocs

    try
        # Initialize islands on workers
        island_refs = [remotecall(init_island, w, problem, seed)
                       for (w, seed) in zip(workers, island_seeds)]

        for gen in 1:algorithm.island_algorithm.generations
            # Evolve one generation on each worker (parallel across processes)
            futures = [remotecall(evolve_one_gen!, w, ref)
                       for (w, ref) in zip(workers, island_refs)]
            results = fetch.(futures)  # blocks until all finish

            # Migration
            if gen % algorithm.migration_interval == 0
                _distributed_migrate!(workers, island_refs, algorithm, rng)
            end

            # Logging from results
            ...
        end

        # Gather final populations
        ...
    finally
        if auto_rmprocs && !isempty(added_pids)
            rmprocs(added_pids)
        end
    end
end
```

Each worker holds its island state in a module-level `Ref` (or a global `Dict{Int, IslandState}` keyed by island ID). The `evolve_one_gen!` function does selection, mutation, crossover, compilation via `@eval`, and evaluation — all within the worker's own world age.

### Asynchronous Distributed Mode

Each worker runs its full evolution loop independently. Migration uses `RemoteChannel` inboxes:

```julia
function _distributed_async_solve(problem, algorithm, rng;
                                   verbose, callback,
                                   auto_addprocs=false, auto_rmprocs=false)
    workers = _acquire_workers(algorithm.n_islands;
                               auto_add=auto_addprocs)
    added_pids = ...  # track which pids we added

    # One inbox channel per island
    inboxes = [RemoteChannel(() -> Channel{MigrantGenome}(32), w)
               for w in workers]

    # Each worker gets its inbox + the outbox channels for its migration targets
    topology = algorithm.topology

    futures = []
    for (i, w) in enumerate(workers)
        targets = migration_targets(topology, i, length(workers))
        outboxes = [inboxes[t] for t in targets]
        f = remotecall(run_island_async, w,
                       problem, algorithm.island_algorithm,
                       island_seeds[i], inboxes[i], outboxes,
                       algorithm.migration_interval, algorithm.migration_size)
        push!(futures, f)
    end

    # Monitor progress (non-blocking poll)
    ...

    # Gather results when all workers finish
    results = fetch.(futures)
    ...
end
```

Worker-side async loop:

```julia
function run_island_async(problem, alg, seed, inbox, outboxes,
                          mig_interval, mig_size)
    rng = MersenneTwister(seed)
    state = _init_island_state(problem, alg, rng)

    for gen in 1:alg.generations
        evolve_one_gen!(state, alg, rng)

        if gen % mig_interval == 0
            # Non-blocking receive: drain inbox
            while isready(inbox)
                migrant = take!(inbox)
                inject_migrant!(state, migrant)
            end

            # Send best individuals to outboxes
            emigrants = get_top_individuals(state, mig_size)
            for ch in outboxes
                for m in emigrants
                    put!(ch, MigrantGenome(m.body, m.fitness))
                end
            end
        end
    end

    return IslandResult(state.genomes, state.fitnesses, state.history)
end
```

### Progress Monitoring for Async Mode

The main process needs visibility into island progress without blocking. Two options:

**(a) Logging channel:** Each worker sends periodic status updates to a shared `RemoteChannel`:

```julia
status_channel = RemoteChannel(() -> Channel{IslandStatus}(256))

struct IslandStatus
    island_id::Int
    generation::Int
    best_fitness::Float64
    mean_fitness::Float64
    timestamp::Float64
end
```

The main process polls this channel and prints progress.

**(b) Callback on workers:** The existing `callback` parameter works per-worker. The main process provides a callback that sends status to a channel. This is cleaner because it uses the existing API.

### Worker Lifecycle

By default, the framework does **not** auto-manage workers. If `distributed=true` and `nworkers() < n_islands`, the framework errors with a clear message telling the user how many workers are needed. Two keyword arguments on `solve` override this:

```julia
result = solve(problem, algorithm;
               auto_addprocs=true,   # add workers if needed (default: false)
               auto_rmprocs=false)   # remove added workers after solve (default: false)
```

When `auto_addprocs=true`, the framework calls `addprocs(n_islands - nworkers(); exeflags="--project=$(Base.active_project())")` and logs an `@info` message. When `auto_rmprocs=true`, workers that were added by this solve call (tracked by pid) are removed on completion. Workers the user added manually are never removed.

### Convenience: `setup_workers`

For custom evaluators, the user must load their code on all workers. The primary documented approach is `@everywhere include(...)`. A convenience helper is provided:

```julia
Arborist.setup_workers("examples/sorting.jl")
# Equivalent to: @everywhere include(abspath("examples/sorting.jl"))
# Also runs @everywhere using Arborist if not already loaded
```

This is a thin wrapper, not magic — it's documented as a shorthand, and the `@everywhere` pattern is shown first in all examples and docstrings.

### Fallback Behavior

| Condition | Behavior |
|-----------|----------|
| `distributed=false` | Current in-process sequential behavior (backward compatible) |
| `distributed=true, async=false, nworkers() >= n_islands` | Sync distributed, one island per worker |
| `distributed=true, async=true, nworkers() >= n_islands` | Async distributed, one island per worker |
| `distributed=true, nworkers() < n_islands, auto_addprocs=false` | Error with clear message |
| `distributed=true, nworkers() < n_islands, auto_addprocs=true` | Auto `addprocs`, with `@info` message |

### What Changes in Existing Code

**`src/Arborist.jl`**: Add `using Distributed`. Export new types (`RingTopology`, `CompleteTopology`, `RandomTopology`, `MigrantGenome`, `setup_workers`).

**`src/algorithm.jl`**: Add `topology`, `distributed`, `async` fields to `IslandModel`. Add topology types. Backward-compatible defaults (`RingTopology()`, `false`, `false`).

**`src/solve.jl`**: Add two new `solve` methods that dispatch on `distributed=true`:
- `_distributed_sync_solve`
- `_distributed_async_solve`

Add `auto_addprocs` and `auto_rmprocs` keyword arguments to the distributed `solve` methods. The existing sequential solve method is untouched.

**`src/genome/expr_genome.jl`**: Add `MigrantGenome` struct and `receive_migrant` function.

**No changes to**: codegen.jl, evolution.jl, evaluators.jl, operators/, speciation.jl, defaults.jl, ant_genome.jl, graph_genome.jl, examples/. Existing behavior is fully preserved.

### What Does NOT Go Across the Wire

- `GenState` — reconstructed locally on each worker from the problem spec
- `AbstractRNG` — each worker creates its own from a seed
- Compiled functions — each worker `@eval`s its own copies
- Module-level simulator state — each worker has its own

### What Goes Across the Wire

- `MigrantGenome` (body + fitness) — small, natively serializable
- `IslandStatus` (progress updates) — trivial
- Problem spec (evaluator + function set + config) — once at setup time
- Final results (genomes + fitnesses + histories) — once at teardown

### Testing Strategy

1. **Unit tests**: Topology types, MigrantGenome serialization round-trip, migration target computation
2. **Integration tests (sync)**: 2-island sync distributed solve on `TableFitnessEvaluator` (uses `addprocs(2)` in test, cleans up after)
3. **Integration tests (async)**: Same as sync but `async=true`, verify results are finite and population sizes are correct (non-deterministic, so no exact fitness checks)
4. **Fallback test**: Verify `distributed=false` produces identical results to current behavior
5. **Benchmark**: sorting example with 4 distributed islands vs 4 sequential islands — measure wall-clock speedup

### Implementation Phases

**Phase 1**: Topology types and `MigrantGenome` struct. Refactor existing in-process `_migrate!` to use topology dispatch. No behavioral change. Tests pass.

**Phase 2**: Synchronous distributed mode. `_distributed_sync_solve` with `remotecall_fetch`. Worker setup, one-gen-at-a-time coordination. Progress logging from main process.

**Phase 3**: Asynchronous distributed mode. `RemoteChannel`-based migration. Worker-side evolution loop. Progress monitoring via status channel.

**Phase 4**: Sorting example updated to demonstrate distributed islands. Verify that the `@eval` bottleneck is eliminated (measure compilation time per generation with 4 processes vs 1).

### Resolved Design Decisions

1. **Distributed.jl is always imported.** It's stdlib, the overhead is minimal, and gating on runtime detection would add complexity for no real benefit. `using Distributed` goes in `src/Arborist.jl`.

2. **Worker lifecycle is user-managed by default.** The framework does not auto-add or auto-remove workers unless the user passes `auto_addprocs=true` and/or `auto_rmprocs=true` to `solve`. This avoids surprising side effects. Clear error messages tell the user exactly what to do when workers are insufficient.

3. **Convenience helper provided, `@everywhere` is primary.** `Arborist.setup_workers(path)` is a thin wrapper documented as shorthand. All examples and docstrings show `@everywhere include(...)` first.

4. **Mixed genome types are out of scope.** Each island uses the same genome type, parameterized via `GPProblem{G,E}`. Cross-island heterogeneous genomes are a future extension.
