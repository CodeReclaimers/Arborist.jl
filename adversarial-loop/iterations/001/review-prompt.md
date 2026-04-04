You are conducting a focused adversarial review of a research project.

**Project:** Arborist.jl — a generic, extensible genetic programming framework for Julia. Uses a Problem/Algorithm/Solve pattern modeled on SciML/DiffEq. The project implements ExprGenome (Expr-tree based GP), TreeGenome (DynamicExpressions.jl for symbolic regression), GraphGenome (NEAT-style neural topology evolution), an island model, and various operators and speciation strategies. All 1579 tests pass. A paper is being written about the framework.

Your role is to find flaws, untested assumptions, and failure modes that the development process has missed. You are the adversarial reviewer in a two-model workflow: you generate critical hypotheses, and another system tests them empirically.

## What you are NOT being asked to do

- Suggest wholesale alternatives or redesign the project
- Restate known limitations unless you have a NEW argument about why an existing resolution is wrong
- Comment on writing style, documentation organization, or naming conventions
- Provide general praise or encouragement
- Suggest performance optimizations — this review is about correctness only
- Review examples/ directory code (application-level, not core algorithm correctness)
- Review TreeGenome, GraphGenome, AntGenome, LLM operator, or distributed island model code (these are for subsequent iterations)

## Known limitations (do not re-discover these):
- ExprGenome `@eval` adds methods to Julia's method table on every evaluation; long runs accumulate thousands of methods
- AntGenome not thread-safe — uses module-level `Ref` for simulator state
- TreeGenome serialize/deserialize format mismatch — `serialize` outputs infix, `deserialize` parses prefix; binary ops do not round-trip
- GraphGenome.deserialize not implemented — returns `nothing` with a warning
- Distributed NEAT innovation ID collisions across workers
- ExprGenome serialize round-trip is partial (~80% success rate due to `Float32(literal)` forms)
- Fitness sharing sign error (division instead of multiplication) was already found and fixed
- The ExprGenome `deserialize` parser was fixed (2026-03-25) to accept control flow

## What you ARE being asked to do

Find specific, testable problems in the core evolutionary loop, genetic operators, and selection/speciation logic. For each finding, provide enough detail that someone could design an experiment or investigation to test whether you're right.

**Categories to look for:**

1. **Circular dependencies** — A validates B which feeds A. Logic that appears sound but contains hidden self-reference.

2. **Context mismatch** — Something validated in one context (synthetic data, small scale, specific conditions) is being relied on in a different context. Does it still hold?

3. **Untested boundaries** — Two components or ideas work individually but their interaction hasn't been examined. Where are the seams?

4. **Claims that don't follow from evidence** — The project says "we showed X" — does X actually follow from what was measured? Are there alternative explanations?

5. **Scaling walls** — Something works at current scale but might break as the system grows. What are the limits?

6. **Silent failure modes** — The system produces output that looks correct but is subtly wrong. Where could this happen?

7. **Missing mechanisms** — The project assumes something will work but hasn't specified how. Where are the hand-waves?

## Focus Area for This Review

**Area:** Core evolutionary loop, genetic operators, selection, and speciation — the heart of the GP system.

**Why this area:** This is the first adversarial review iteration. We start with the core algorithms that everything else builds on. The evolutionary loop, operator implementations, and selection/speciation logic are where correctness matters most — bugs here silently corrupt all downstream results. The boundaries between components (operators ↔ solve loop, speciation ↔ selection, island model ↔ base algorithm) are the highest-value targets.

**Specific questions:**
1. Does the operator selection logic in `_breed_next_generation!` correctly implement the standard GP reproduction pipeline? Are the probability semantics right?
2. Does the speciation/fitness-sharing interact correctly with tournament selection and elitism?
3. Are the mutation operators (SubtreeMutation, PointMutation, HoistMutation, ExpansionMutation) correctly implemented per standard GP literature?
4. Does the island model's migration logic correctly implement the intended topology?
5. Are there any subtle issues in the crossover implementation (type safety, tree structure preservation)?
6. Does the paper observations document (included below) accurately describe what the code actually does?

## Source Material

### src/solve.jl — Main solve loop

```julia
"""
    solve(problem::GPProblem{G,E}, algorithm::GeneticProgramming; verbose=false, callback=nothing) -> GPResult{G}

Run a genetic programming evolution using the specified problem and algorithm configuration.
Returns a `GPResult` containing the best genome, fitness history, and run metadata.

# Arguments
- `problem::GPProblem{G,E}`: the problem specification (evaluator, genome type, function set)
- `algorithm::GeneticProgramming`: the algorithm configuration

# Keyword Arguments
- `verbose::Bool=false`: if true, print generation statistics
- `callback`: optional callback function `(gen::Int, best_fitness::Float64, best_genome::G) -> nothing`
"""
function solve(problem::GPProblem{G,E},
               algorithm::GeneticProgramming;
               verbose::Bool = false,
               callback = nothing) where {G,E}
    # Create a dedicated RNG from the seed, or use the default RNG.
    # All stochastic operations use this RNG explicitly — no global rand().
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    pop = _initialize_population(problem, algorithm, rng)
    return _run_evolution!(pop, problem, algorithm, rng; verbose=verbose, callback=callback)
end

"""
    _initialize_population(problem, algorithm, rng) -> Tuple{Vector{G}, GenState}

Create the initial population of genomes. Internal function.
"""
function _initialize_population(problem::GPProblem{G,E}, algorithm::GeneticProgramming, rng::AbstractRNG) where {G,E}
    if G !== ExprGenome
        error("_initialize_population only supports ExprGenome. " *
              "$(G) requires a specialized solve method (see AntGenome, GraphGenome).")
    end
    inputs = input_signature(problem.evaluator)
    outputs = output_signature(problem.evaluator)
    state = GenState(rng, problem.function_set, inputs, outputs, problem.num_temps)

    genomes = Vector{G}(undef, algorithm.pop_size)
    for i in 1:algorithm.pop_size
        body = [create_random_assignment(state) for _ in 1:3]
        genomes[i] = ExprGenome(body, state)
    end

    return (genomes, state)
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
                         callback = nothing) where {G,E}
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
        selection_fitnesses = _apply_speciation!(genomes, fitnesses,
                                                  algorithm.speciation, species_state, rng)

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
               auto_addprocs::Bool = false,
               auto_rmprocs::Bool = false) where {G,E}
    # Dispatch to distributed solvers if requested
    if algorithm.distributed && !algorithm.async
        return _distributed_sync_solve(problem, algorithm;
                                        verbose=verbose, callback=callback,
                                        auto_addprocs=auto_addprocs,
                                        auto_rmprocs=auto_rmprocs)
    elseif algorithm.distributed && algorithm.async
        return _distributed_async_solve(problem, algorithm;
                                         verbose=verbose, callback=callback,
                                         auto_addprocs=auto_addprocs,
                                         auto_rmprocs=auto_rmprocs)
    end

    # --- Sequential in-process mode (original behavior) ---
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    alg = algorithm.island_algorithm
    n = algorithm.n_islands
    pop_size = alg.pop_size
    bp = alg.bloat_penalty

    # Initialize n independent islands, each with its own RNG-seeded GenState.
    island_genomes = Vector{Vector{G}}(undef, n)
    island_states = Vector{GenState}(undef, n)
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

        # Migration: ring topology, every migration_interval generations.
        if gen % algorithm.migration_interval == 0
            _migrate!(island_genomes, island_fitnesses, algorithm, rng)
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
    _migrate!(island_genomes, island_fitnesses, algorithm, rng)

Perform migration using the algorithm's topology. Sends the top
`migration_size` individuals from each island to its topology-determined
destinations, replacing the worst individuals on those destinations.
"""
function _migrate!(island_genomes, island_fitnesses, algorithm::IslandModel, rng::AbstractRNG)
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
                    island_genomes[dest][worst_idx] = genome
                    island_fitnesses[dest][worst_idx] = emigrant_fitnesses[i][k]
                end
            end
        end
    end
end
```

### src/algorithm.jl — Algorithm configuration

```julia
struct GeneticProgramming <: AbstractEvolutionaryAlgorithm
    pop_size::Int
    generations::Int
    mutation_rate::Float64
    crossover_rate::Float64
    elitism::Int
    max_depth::Int
    bloat_penalty::Float64
    parallel::Bool
    speciation::AbstractSpeciation
    mutation_ops::Vector{AbstractMutationOperator}
    crossover_ops::Vector{AbstractCrossoverOperator}
    selection::AbstractSelectionStrategy
    convergence_threshold::Float64
end

function GeneticProgramming(;
    pop_size::Int = 100,
    generations::Int = 200,
    mutation_rate::Float64 = 0.3,
    crossover_rate::Float64 = 0.3,
    elitism::Int = 2,
    max_depth::Int = 8,
    bloat_penalty::Float64 = 0.0,
    parallel::Bool = true,
    speciation::AbstractSpeciation = NoSpeciation(),
    mutation_ops::Vector{<:AbstractMutationOperator} = AbstractMutationOperator[SubtreeMutation(), PointMutation()],
    crossover_ops::Vector{<:AbstractCrossoverOperator} = AbstractCrossoverOperator[SubtreeCrossover()],
    selection::AbstractSelectionStrategy = TournamentSelection(3),
    convergence_threshold::Float64 = Inf
)
    GeneticProgramming(
        pop_size, generations, mutation_rate, crossover_rate,
        elitism, max_depth, bloat_penalty,
        parallel, speciation,
        convert(Vector{AbstractMutationOperator}, mutation_ops),
        convert(Vector{AbstractCrossoverOperator}, crossover_ops),
        selection, convergence_threshold
    )
end

struct IslandModel <: AbstractEvolutionaryAlgorithm
    n_islands::Int
    island_algorithm::GeneticProgramming
    migration_interval::Int
    migration_size::Int
    topology::AbstractTopology
    distributed::Bool
    async::Bool
end

function IslandModel(;
    n_islands::Int = 4,
    island_algorithm::GeneticProgramming = GeneticProgramming(),
    migration_interval::Int = 10,
    migration_size::Int = 2,
    topology::AbstractTopology = RingTopology(),
    distributed::Bool = false,
    async::Bool = false
)
    if async && !distributed
        throw(ArgumentError("async=true requires distributed=true"))
    end
    IslandModel(n_islands, island_algorithm, migration_interval, migration_size,
                topology, distributed, async)
end
```

### src/operators/mutation.jl — Mutation operators

```julia
struct SubtreeMutation <: AbstractMutationOperator end
struct PointMutation <: AbstractMutationOperator end
struct HoistMutation <: AbstractMutationOperator end
struct ExpansionMutation <: AbstractMutationOperator end

# --- SubtreeMutation ---
function mutate(op::SubtreeMutation, g::ExprGenome, rng::AbstractRNG)
    new_body = deepcopy(g.body)
    if !isempty(new_body)
        idx = rand(rng, 1:length(new_body))
        new_body[idx] = create_random_statement(g.state)
    end
    return ExprGenome(new_body, g.state)
end

# --- PointMutation ---
function mutate(op::PointMutation, g::ExprGenome, rng::AbstractRNG)
    new_body = deepcopy(g.body)
    if !isempty(new_body)
        all_nodes = Expr[]
        for stmt in new_body
            append!(all_nodes, unravel(stmt))
        end
        if !isempty(all_nodes)
            target = rand(rng, all_nodes)
            try
                mutate!(g.state, target)
            catch e
                e isa InterruptException && rethrow()
            end
        end
    end
    return ExprGenome(new_body, g.state)
end

# --- HoistMutation ---
function mutate(op::HoistMutation, g::ExprGenome, rng::AbstractRNG)
    new_body = deepcopy(g.body)
    if isempty(new_body)
        return ExprGenome(new_body, g.state)
    end

    all_nodes = Expr[]
    for stmt in new_body
        append!(all_nodes, unravel(stmt))
    end

    non_leaf = [node for node in all_nodes if any(a -> a isa Expr, node.args)]

    if isempty(non_leaf)
        return mutate(SubtreeMutation(), g, rng)
    end

    target = rand(rng, non_leaf)
    expr_children = [a for a in target.args if a isa Expr]
    child = deepcopy(rand(rng, expr_children))

    # Replace target in the body.
    for i in 1:length(new_body)
        if new_body[i] === target
            new_body[i] = child
            return ExprGenome(new_body, g.state)
        end
    end

    for stmt in new_body
        if replace_subtree!(stmt, target, child)
            return ExprGenome(new_body, g.state)
        end
    end

    return ExprGenome(new_body, g.state)
end

# --- ExpansionMutation ---
function mutate(op::ExpansionMutation, g::ExprGenome, rng::AbstractRNG)
    new_body = deepcopy(g.body)
    if isempty(new_body)
        return ExprGenome(new_body, g.state)
    end

    leaf_positions = Tuple{Expr, Int}[]
    for stmt in new_body
        _collect_leaf_positions!(leaf_positions, stmt, g.state)
    end

    if isempty(leaf_positions)
        return ExprGenome(new_body, g.state)
    end

    (parent, idx) = rand(rng, leaf_positions)
    leaf_value = parent.args[idx]

    wrapped = try
        wrap_rvalue(g.state, leaf_value)
    catch e
        e isa InterruptException && rethrow()
        nothing
    end

    if wrapped !== nothing && wrapped !== leaf_value
        parent.args[idx] = wrapped
    end

    return ExprGenome(new_body, g.state)
end

function _collect_leaf_positions!(positions::Vector{Tuple{Expr, Int}}, expr::Expr, s::GenState)
    for i in 1:length(expr.args)
        a = expr.args[i]
        if a isa Expr
            _collect_leaf_positions!(positions, a, s)
        elseif (a isa Symbol || a isa Number)
            if (expr.head == :(=) && i == 1) || (expr.head == :call && i == 1)
                continue
            end
            try
                get_rvalue_type(s, a)
                push!(positions, (expr, i))
            catch e
                e isa InterruptException && rethrow()
            end
        end
    end
end
```

### src/operators/crossover.jl — Crossover operator

```julia
struct SubtreeCrossover <: AbstractCrossoverOperator end

function crossover(op::SubtreeCrossover, g1::ExprGenome, g2::ExprGenome, rng::AbstractRNG)
    block_a = Expr(:block, deepcopy(g1.body)...)
    block_b = Expr(:block, deepcopy(g2.body)...)

    (offspring_a, offspring_b) = crossover(g1.state, block_a, block_b)

    expr_a = offspring_a.head == :block ? collect(offspring_a.args) : [offspring_a]
    expr_b = offspring_b.head == :block ? collect(offspring_b.args) : [offspring_b]

    filter_exprs(v) = Expr[e for e in v if e isa Expr]

    body_a = filter_exprs(expr_a)
    body_b = filter_exprs(expr_b)

    if isempty(body_a)
        body_a = deepcopy(g1.body)
    end
    if isempty(body_b)
        body_b = deepcopy(g2.body)
    end

    return (ExprGenome(body_a, g1.state), ExprGenome(body_b, g1.state))
end
```

### src/genome/evolution.jl — Crossover implementation

```julia
function replace_subtree!(tree::Expr, target::Expr, replacement::Expr)
    for i in 1:length(tree.args)
        if tree.args[i] isa Expr && tree.args[i] === target
            tree.args[i] = replacement
            return true
        elseif tree.args[i] isa Expr
            if replace_subtree!(tree.args[i], target, replacement)
                return true
            end
        end
    end
    return false
end

function crossover(s::GenState, parent_a::Expr, parent_b::Expr)
    nodes_a = unravel(parent_a)
    nodes_b = unravel(parent_b)

    compatible_pairs = Tuple{Int, Int}[]
    types_a = Vector{Union{DataType, Nothing}}(undef, length(nodes_a))
    for (i, na) in enumerate(nodes_a)
        types_a[i] = try
            get_rvalue_type(s, na)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
    end

    for (j, nb) in enumerate(nodes_b)
        type_b = try
            get_rvalue_type(s, nb)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        type_b === nothing && continue
        for (i, type_a) in enumerate(types_a)
            type_a === nothing && continue
            if type_a == type_b
                push!(compatible_pairs, (i, j))
            end
        end
    end

    if isempty(compatible_pairs)
        return (deepcopy(parent_a), deepcopy(parent_b))
    end

    (idx_a, idx_b) = rand(s.rng, compatible_pairs)

    offspring_a = deepcopy(parent_a)
    offspring_b = deepcopy(parent_b)

    copy_nodes_a = unravel(offspring_a)
    copy_nodes_b = unravel(offspring_b)

    subtree_from_a = deepcopy(copy_nodes_a[idx_a])
    subtree_from_b = deepcopy(copy_nodes_b[idx_b])

    if copy_nodes_a[idx_a] === offspring_a
        offspring_a = subtree_from_b
    else
        replace_subtree!(offspring_a, copy_nodes_a[idx_a], subtree_from_b)
    end

    if copy_nodes_b[idx_b] === offspring_b
        offspring_b = subtree_from_a
    else
        replace_subtree!(offspring_b, copy_nodes_b[idx_b], subtree_from_a)
    end

    return (offspring_a, offspring_b)
end
```

### src/speciation.jl — Speciation and fitness sharing

```julia
struct NoSpeciation <: AbstractSpeciation end

struct ThresholdSpeciation <: AbstractSpeciation
    threshold::Float64
    min_species_size::Int
    stagnation_limit::Int
    sharing_formula::Symbol
end

struct BehavioralSpeciation <: AbstractSpeciation
    fingerprint_fn::Function
    distance_fn::Function
    threshold::Float64
    min_species_size::Int
    stagnation_limit::Int
    sharing_formula::Symbol
end

function apply_sharing(raw_fitness::Float64, species_size::Int, formula::Symbol)::Float64
    species_size <= 1 && return raw_fitness
    formula == :none   && return raw_fitness
    formula == :linear && return raw_fitness * species_size
    formula == :sqrt   && return raw_fitness * sqrt(species_size)
    formula == :log2   && return raw_fitness * (1.0 + log2(species_size))
    throw(ArgumentError("Unknown sharing formula: $formula. Use :none, :linear, :sqrt, or :log2"))
end

mutable struct _SpeciesInfo
    representative::Any
    best_fitness::Float64
    stagnation::Int
end

_init_species_state(::NoSpeciation) = nothing
_init_species_state(::ThresholdSpeciation) = _SpeciesInfo[]
_init_species_state(::BehavioralSpeciation) = _SpeciesInfo[]

_apply_speciation!(genomes, fitnesses, ::NoSpeciation, ::Nothing, rng) = fitnesses

function _apply_speciation!(genomes::Vector{G}, fitnesses::Vector{Float64},
                            spec::ThresholdSpeciation,
                            species_list::Vector{_SpeciesInfo},
                            rng::AbstractRNG) where {G}
    n = length(genomes)

    n_species = length(species_list)
    member_lists = [Int[] for _ in 1:n_species]

    assignments = zeros(Int, n)
    for i in 1:n
        assigned = false
        for si in 1:length(species_list)
            if distance(genomes[i], species_list[si].representative) <= spec.threshold
                push!(member_lists[si], i)
                assignments[i] = si
                assigned = true
                break
            end
        end
        if !assigned
            push!(species_list, _SpeciesInfo(deepcopy(genomes[i]), fitnesses[i], 0))
            push!(member_lists, [i])
            assignments[i] = length(species_list)
        end
    end

    _update_stagnation!(species_list, member_lists, fitnesses, genomes)
    global_best_species = assignments[1]
    _cull_species!(species_list, member_lists, assignments, genomes, fitnesses,
                   spec.stagnation_limit, spec.min_species_size, global_best_species, n)
    _remove_empty_species!(species_list, member_lists)

    return _compute_shared_fitnesses(fitnesses, member_lists, spec.sharing_formula)
end

function _apply_speciation!(genomes::Vector{G}, fitnesses::Vector{Float64},
                            spec::BehavioralSpeciation,
                            species_list::Vector{_SpeciesInfo},
                            rng::AbstractRNG) where {G}
    n = length(genomes)

    fingerprints = Vector{Any}(undef, n)
    for i in 1:n
        fingerprints[i] = spec.fingerprint_fn(genomes[i])
    end

    n_species = length(species_list)
    member_lists = [Int[] for _ in 1:n_species]

    assignments = zeros(Int, n)
    for i in 1:n
        assigned = false
        for si in 1:length(species_list)
            if spec.distance_fn(fingerprints[i], species_list[si].representative) <= spec.threshold
                push!(member_lists[si], i)
                assignments[i] = si
                assigned = true
                break
            end
        end
        if !assigned
            push!(species_list, _SpeciesInfo(fingerprints[i], fitnesses[i], 0))
            push!(member_lists, [i])
            assignments[i] = length(species_list)
        end
    end

    # Update stagnation and representatives (using fingerprints, not genomes).
    for si in 1:length(species_list)
        members = member_lists[si]
        if isempty(members)
            species_list[si].stagnation += 1
            continue
        end
        species_best = minimum(fitnesses[j] for j in members)
        if species_list[si].best_fitness - species_best >= 1e-6
            species_list[si].best_fitness = species_best
            species_list[si].stagnation = 0
        else
            species_list[si].stagnation += 1
        end
        best_member = members[argmin([fitnesses[j] for j in members])]
        species_list[si].representative = fingerprints[best_member]
    end

    global_best_species = assignments[1]

    to_cull = Int[]
    for si in 1:length(species_list)
        si == global_best_species && continue
        if species_list[si].stagnation > spec.stagnation_limit &&
           length(member_lists[si]) < spec.min_species_size
            push!(to_cull, si)
        end
    end

    if !isempty(to_cull)
        orphans = Int[]
        for si in to_cull
            append!(orphans, member_lists[si])
        end
        remaining = [si for si in 1:length(species_list) if si ∉ to_cull]
        for oi in orphans
            best_si_idx = 1
            best_dist = Inf
            for (ri, si) in enumerate(remaining)
                d = spec.distance_fn(fingerprints[oi], species_list[si].representative)
                if d < best_dist
                    best_dist = d
                    best_si_idx = ri
                end
            end
            push!(member_lists[remaining[best_si_idx]], oi)
            assignments[oi] = remaining[best_si_idx]
        end
        sort!(to_cull, rev=true)
        for si in to_cull
            deleteat!(species_list, si)
            deleteat!(member_lists, si)
        end
    end

    _remove_empty_species!(species_list, member_lists)

    return _compute_shared_fitnesses(fitnesses, member_lists, spec.sharing_formula)
end

function _update_stagnation!(species_list, member_lists, fitnesses, genomes)
    for si in 1:length(species_list)
        members = member_lists[si]
        if isempty(members)
            species_list[si].stagnation += 1
            continue
        end
        species_best = minimum(fitnesses[j] for j in members)
        if species_list[si].best_fitness - species_best >= 1e-6
            species_list[si].best_fitness = species_best
            species_list[si].stagnation = 0
        else
            species_list[si].stagnation += 1
        end
        best_member = members[argmin([fitnesses[j] for j in members])]
        species_list[si].representative = deepcopy(genomes[best_member])
    end
end

function _cull_species!(species_list, member_lists, assignments, genomes, fitnesses,
                        stagnation_limit, min_species_size, global_best_species, n)
    to_cull = Int[]
    for si in 1:length(species_list)
        si == global_best_species && continue
        if species_list[si].stagnation > stagnation_limit &&
           length(member_lists[si]) < min_species_size
            push!(to_cull, si)
        end
    end
    isempty(to_cull) && return

    orphans = Int[]
    for si in to_cull
        append!(orphans, member_lists[si])
    end
    remaining = [si for si in 1:length(species_list) if si ∉ to_cull]
    for oi in orphans
        best_si_idx = 1
        best_dist = Inf
        for (ri, si) in enumerate(remaining)
            d = distance(genomes[oi], species_list[si].representative)
            if d < best_dist
                best_dist = d
                best_si_idx = ri
            end
        end
        push!(member_lists[remaining[best_si_idx]], oi)
        assignments[oi] = remaining[best_si_idx]
    end
    sort!(to_cull, rev=true)
    for si in to_cull
        deleteat!(species_list, si)
        deleteat!(member_lists, si)
    end
end

function _remove_empty_species!(species_list, member_lists)
    empty_species = [si for si in 1:length(species_list) if isempty(member_lists[si])]
    isempty(empty_species) && return
    sort!(empty_species, rev=true)
    for si in empty_species
        deleteat!(species_list, si)
        deleteat!(member_lists, si)
    end
end

function _compute_shared_fitnesses(fitnesses, member_lists, formula::Symbol)
    shared_fitnesses = copy(fitnesses)
    for (si, members) in enumerate(member_lists)
        species_size = length(members)
        for j in members
            shared_fitnesses[j] = apply_sharing(fitnesses[j], species_size, formula)
        end
    end
    return shared_fitnesses
end
```

### src/genome/codegen.jl — Code generation and tree manipulation

```julia
struct FunctionDetails
    name::Symbol
    args::Vector{DataType}
    return_type::DataType
end

struct FunctionSet
    funcs::Set{FunctionDetails}
end

struct GenState
    rng::AbstractRNG
    statement_types::Vector{Symbol}
    funcs::FunctionSet
    inputs::Dict{Symbol, DataType}
    outputs::Dict{Symbol, DataType}
    temps::Dict{Symbol, DataType}
    used_types::Set{DataType}
    all_vars::Dict{Symbol, DataType}
end

function GenState(rng::AbstractRNG, fset::FunctionSet, inputs::Dict{Symbol, DataType}, outputs::Dict{Symbol, DataType}, num_temps::Int)
    statement_types = [:(=), :call, :for, :while, :if, :block]
    used_types = Set([Bool])
    for (k, v) in union(inputs, outputs)
        push!(used_types, v)
    end
    used_types_vec = _sorted_types(used_types)
    temps = Dict([(Symbol("__temp_$i"), rand(rng, used_types_vec)) for i in 1:num_temps])
    all_vars = Dict(union(inputs, outputs, temps))
    return GenState(rng, statement_types, fset, inputs, outputs, temps, used_types, all_vars)
end

function create_random_assignment(s::GenState)
    while true
        v = rand(s.rng, get_lvalues(s))
        r = create_random_rvalue(s, v[2])
        if v[1] != r
            return :($(v[1]) = $r)
        end
    end
end

function mutate!(s::GenState, expr::Expr)
    if expr.head == :(=)
        mutate_assignment!(s, expr)
    elseif expr.head == :call
        mutate_function_call!(s, expr)
    elseif expr.head == :for
        mutate_for_loop!(s, expr)
    elseif expr.head == :while
        mutate_while_loop!(s, expr)
    elseif expr.head == :if
        mutate_if_statement!(s, expr)
    elseif expr.head == :block
        mutate_block!(s, expr)
    else
        throw(ErrorException("Unsupported expression type $(expr.head)"))
    end
    return expr
end

function add_loop_checks_expr(expr::Expr, limit::Int)
    if expr.head == :for || expr.head == :while
        counter = gensym("__loopcheck")
        increment = :($counter += 1)
        check = Expr(:if, Expr(:call, :>, counter, limit), Expr(:call, :throw, Expr(:call, :LoopLimitExceeded)))
        body_idx = 2
        processed_body = add_loop_checks_expr(expr.args[body_idx], limit)
        if processed_body isa Expr && processed_body.head == :block
            new_body = Expr(:block, increment, check, processed_body.args...)
        else
            new_body = Expr(:block, increment, check, processed_body)
        end
        if expr.head == :for
            new_loop = Expr(:for, expr.args[1], new_body)
        else
            processed_cond = add_loop_checks_expr(expr.args[1], limit)
            new_loop = Expr(:while, processed_cond, new_body)
        end
        return Expr(:block, :($counter = 0), new_loop)
    elseif expr.head == :block
        return Expr(:block, [add_loop_checks_expr(a, limit) for a in expr.args]...)
    else
        new_args = [a isa Expr ? add_loop_checks_expr(a, limit) : a for a in expr.args]
        return Expr(expr.head, new_args...)
    end
end

function unravel(tree, expressions=[])
    if tree isa Expr
        push!(expressions, tree)
        for arg in tree.args
            unravel(arg, expressions)
        end
    end
    return expressions
end
```

### src/genome/expr_genome.jl — ExprGenome type and evaluation

```julia
struct ExprGenome <: AbstractGenome
    body::Vector{Expr}
    state::GenState
end

struct GPProblem{G<:AbstractGenome, E<:AbstractEvaluator}
    evaluator::E
    genome_type::Type{G}
    function_set::FunctionSet
    num_temps::Int
    seed::Union{Int, Nothing}
end

function evaluate_genome(g::ExprGenome, evaluator::AbstractEvaluator)
    fname = gensym("evolved")
    try
        checked_body = add_loop_checks(g.body)
        harness = create_harness(g.state, checked_body, fname)
        f = @eval $harness
        return evaluate(evaluator, f)
    catch e
        e isa InterruptException && rethrow()
        return Inf
    end
end
```

### src/evaluators.jl — TableFitnessEvaluator

```julia
struct TableFitnessEvaluator <: AbstractEvaluator
    input_cols::Dict{Symbol, DataType}
    output_cols::Dict{Symbol, DataType}
    input_rows::Vector{Dict{Symbol, Any}}
    output_rows::Vector{Dict{Symbol, Any}}
    time_limit_ns::Int
end

function evaluate(fe::TableFitnessEvaluator, f::Function)
    n_rows = length(fe.input_rows)
    n_errors = 0
    total_se = 0.0

    sorted_inputs = sort(collect(fe.input_cols), by=first)
    sorted_outputs = sort(collect(fe.output_cols), by=first)
    n_outputs = length(sorted_outputs)

    for (in_row, out_row) in zip(fe.input_rows, fe.output_rows)
        t0 = time_ns()
        result = try
            args = [in_row[name] for (name, _) in sorted_inputs]
            Base.invokelatest(f, args...)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        elapsed_ns = time_ns() - t0

        if result === nothing || elapsed_ns > fe.time_limit_ns
            n_errors += 1
            continue
        end

        se = 0.0
        try
            if n_outputs == 1
                expected = out_row[sorted_outputs[1][1]]
                se = (Float64(result) - Float64(expected))^2
            else
                for (k, (name, _)) in enumerate(sorted_outputs)
                    expected = out_row[name]
                    se += (Float64(result[k]) - Float64(expected))^2
                end
            end
        catch e
            e isa InterruptException && rethrow()
            n_errors += 1
            continue
        end

        if !isfinite(se)
            n_errors += 1
            continue
        end

        total_se += se
    end

    if n_errors > n_rows / 2
        return Inf
    end

    n_success = n_rows - n_errors
    if n_success == 0
        return Inf
    end

    return total_se / n_success
end
```

### src/topology.jl — Migration topologies

```julia
struct RingTopology <: AbstractTopology end
struct CompleteTopology <: AbstractTopology end
struct RandomTopology <: AbstractTopology
    n_targets::Int
end

function migration_targets(::RingTopology, i::Int, n_islands::Int, ::AbstractRNG)
    return [(i % n_islands) + 1]
end

function migration_targets(::CompleteTopology, i::Int, n_islands::Int, ::AbstractRNG)
    return [j for j in 1:n_islands if j != i]
end

function migration_targets(t::RandomTopology, i::Int, n_islands::Int, rng::AbstractRNG)
    others = [j for j in 1:n_islands if j != i]
    k = min(t.n_targets, length(others))
    return others[randperm(rng, length(others))[1:k]]
end
```

### src/migration.jl — MigrantGenome

```julia
struct MigrantGenome
    data::Any
    fitness::Float64
    genome_type::Symbol
end

function to_migrant(g::ExprGenome, fitness::Float64)
    MigrantGenome(deepcopy(g.body), fitness, :ExprGenome)
end

function from_migrant(m::MigrantGenome, state::GenState)
    m.genome_type == :ExprGenome || throw(ArgumentError("Expected ExprGenome migrant, got $(m.genome_type)"))
    ExprGenome(deepcopy(m.data), state)
end
```

## Relevant paper observations (arborist_paper_observations.md excerpts)

### Section 1.2 — Problem/Algorithm/Solve pattern
> Arborist.jl adopts the Problem/Algorithm/Solve pattern established by SciML's DifferentialEquations.jl and Optimization.jl ecosystems.

### Section 1.6 — Reproducibility design
> All stochastic operations thread an explicit `rng::AbstractRNG` parameter derived from a user-specified seed. Collection sampling uses deterministic sorted ordering to counter Julia's non-deterministic Dict/Set iteration. Temp variable names use `__temp_$i` counters rather than `gensym()` to avoid session-dependent naming.
> Fixed seed + `parallel=false` gives fully reproducible runs.

### Section 2.5 — Speciation findings
> Genomic speciation (ThresholdSpeciation, symmetric-difference distance) is harmful for this problem. With threshold=10, the population splits into ~100-122 species for 200 individuals because syntactically trivial differences produce large structural distances.
> Root cause of the harm: the sharing formula `shared = raw * species_size` (correct for NEAT-style topology protection) inverts selection pressure when most species are singletons.

### Section 3.1 — Sharing formula tradeoff
> Three sharing formulas are now implemented: `:linear`, `:sqrt`, `:log2`.

### Section 3.3 — XOR NEAT benchmark
> The fitness sharing sign error (division instead of multiplication) in the initial GraphGenome implementation produced networks that learned nothing. After fixing to multiplication, XOR converged 4/5 seeds.

## Prior Findings (if any)

None — this is the first iteration.

## Output Format

Produce numbered findings. For each:

```
### Finding N: [short title]

**Tag:** [STRUCTURAL | PARAMETER | UNTESTED | CIRCULAR | SCALING | SILENT_FAILURE | MISSING_MECHANISM]
**Confidence:** [HIGH | MEDIUM | LOW]
**Severity:** [CRITICAL — breaks a core claim | SIGNIFICANT — wrong but fixable | MINOR — edge case or refinement]

**Claim being challenged:**
[The specific claim or assumption you think is wrong or undertested]

**Why it might be wrong:**
[Your argument. Be specific. Reference the source material.]

**Suggested test:**
[A concrete investigation that would determine whether you're right.
Include what the expected result would be if the project is correct
vs. if your concern is valid.]
```

Aim for 5-15 findings. Quality over quantity — one well-argued structural finding is worth more than ten vague concerns. If you find fewer than 5 things worth reporting, that's fine.
