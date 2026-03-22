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
    _tournament_select(genomes, fitnesses, tournament_size, rng) -> Int

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
    _run_evolution!(pop, problem, algorithm, rng; verbose, callback) -> GPResult

The internal evolution loop. Not part of the public API.
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

    # Evaluate initial population.
    for i in 1:pop_size
        fitnesses[i] = evaluate_genome(genomes[i], problem.evaluator)
    end

    fitness_history = Float64[]
    mean_history = Float64[]

    t0 = time()

    for gen in 1:algorithm.generations
        # Sort by fitness ascending (best first).
        order = sortperm(fitnesses)
        genomes = genomes[order]
        fitnesses = fitnesses[order]

        # Record history.
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

        # Build next generation.
        next_genomes = Vector{G}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        # Elitism: carry top individuals forward.
        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = deepcopy(genomes[i])
            next_fitnesses[i] = fitnesses[i]
        end

        # Determine tournament size from selection strategy.
        t_size = algorithm.selection isa TournamentSelection ?
                 algorithm.selection.tournament_size : algorithm.tournament_size

        # Fill the rest via tournament selection + genetic operators.
        idx = algorithm.elitism + 1
        while idx <= pop_size
            r = rand(rng)
            if r < algorithm.crossover_rate && idx + 1 <= pop_size
                p1_idx = _tournament_select(fitnesses, t_size, rng)
                p2_idx = _tournament_select(fitnesses, t_size, rng)
                op = rand(rng, algorithm.crossover_ops)
                (c1, c2) = crossover(op, genomes[p1_idx], genomes[p2_idx], rng)
                next_genomes[idx] = c1
                next_genomes[idx + 1] = c2
                idx += 2
            elseif r < algorithm.crossover_rate + algorithm.mutation_rate
                p_idx = _tournament_select(fitnesses, t_size, rng)
                op = rand(rng, algorithm.mutation_ops)
                child = mutate(op, genomes[p_idx], rng)
                next_genomes[idx] = child
                idx += 1
            else
                p_idx = _tournament_select(fitnesses, t_size, rng)
                next_genomes[idx] = deepcopy(genomes[p_idx])
                idx += 1
            end
        end

        # Evaluate new individuals (skip elites which already have fitness).
        for i in (algorithm.elitism + 1):pop_size
            next_fitnesses[i] = evaluate_genome(next_genomes[i], problem.evaluator)
        end

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
        fitnesses[1] < 1.0
    )
end
