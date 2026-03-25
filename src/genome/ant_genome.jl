# ant_genome.jl — Genome for imperative, side-effectful programs.

# Module-level ant simulator reference for @eval'd programs.
const _ant_sim_ref = Ref{Any}(nothing)

# =============================================================================
# Ant simulator
# =============================================================================

"""
    AntSimulator

Mutable state for ant simulation on a toroidal grid.
"""
mutable struct AntSimulator
    grid::Matrix{Bool}
    row::Int
    col::Int
    direction::Int   # 0=east, 1=south, 2=west, 3=north
    moves::Int
    food_eaten::Int
    max_moves::Int
end

function AntSimulator(food_positions::Vector{Tuple{Int,Int}}, max_moves::Int;
                      grid_size::Int=32)
    grid = zeros(Bool, grid_size, grid_size)
    for (r, c) in food_positions
        grid[r, c] = true
    end
    AntSimulator(grid, 1, 1, 0, 0, 0, max_moves)
end

const _ANT_DR = [0, 1, 0, -1]  # east, south, west, north
const _ANT_DC = [1, 0, -1, 0]

function _ant_ahead(ant::AntSimulator)
    gs = size(ant.grid, 1)
    r = mod1(ant.row + _ANT_DR[ant.direction + 1], gs)
    c = mod1(ant.col + _ANT_DC[ant.direction + 1], gs)
    return (r, c)
end

# Primitives callable from @eval'd evolved programs
"""Move the ant forward. Eats food if present."""
function gp_ant_move(::Bool)::Bool
    ant = _ant_sim_ref[]
    ant === nothing && return false
    ant.moves >= ant.max_moves && return false
    r, c = _ant_ahead(ant)
    ant.row = r; ant.col = c; ant.moves += 1
    if ant.grid[r, c]
        ant.food_eaten += 1; ant.grid[r, c] = false
    end
    return true
end

"""Turn the ant left."""
function gp_ant_left(::Bool)::Bool
    ant = _ant_sim_ref[]
    ant === nothing && return false
    ant.moves >= ant.max_moves && return false
    ant.direction = mod(ant.direction + 3, 4)
    ant.moves += 1
    return true
end

"""Turn the ant right."""
function gp_ant_right(::Bool)::Bool
    ant = _ant_sim_ref[]
    ant === nothing && return false
    ant.moves >= ant.max_moves && return false
    ant.direction = mod(ant.direction + 1, 4)
    ant.moves += 1
    return true
end

"""Check if food is ahead (sensor, does not consume a move)."""
function gp_ant_food_ahead(::Bool)::Bool
    ant = _ant_sim_ref[]
    ant === nothing && return false
    r, c = _ant_ahead(ant)
    return ant.grid[r, c]
end

# =============================================================================
# AntGenome struct
# =============================================================================

"""
    AntGenome <: AbstractGenome

A genome for evolving programs that control an agent via side-effectful
primitives. Unlike `ExprGenome`, AntGenome does not require a typed
input/output signature — the program operates on implicit agent state
via a module-level simulator reference.

Suitable for the Santa Fe Ant Trail and similar control problems.

# Fields
- `program::Expr`: a `:block` expression of nested primitive calls and control flow
- `primitives::Vector{Symbol}`: action primitives (consume moves)
- `conditions::Vector{Symbol}`: condition primitives (sensors)
- `max_depth::Int`: maximum program depth
"""
struct AntGenome <: AbstractGenome
    program::Expr
    primitives::Vector{Symbol}
    conditions::Vector{Symbol}
    max_depth::Int
end

# =============================================================================
# Random program generation
# =============================================================================

function _random_ant_program(primitives::Vector{Symbol},
                              conditions::Vector{Symbol},
                              max_depth::Int, rng::AbstractRNG)
    if max_depth <= 0
        return Expr(:call, rand(rng, primitives), true)
    end

    choice = rand(rng, 1:3)
    if choice == 1
        # Action primitive
        return Expr(:call, rand(rng, primitives), true)
    elseif choice == 2 && !isempty(conditions)
        # If-then-else with condition
        cond = Expr(:call, rand(rng, conditions), true)
        then_b = _random_ant_program(primitives, conditions, max_depth - 1, rng)
        else_b = _random_ant_program(primitives, conditions, max_depth - 1, rng)
        return Expr(:if, cond, then_b, else_b)
    else
        # Sequential block
        n = rand(rng, 2:3)
        stmts = [_random_ant_program(primitives, conditions, max_depth - 1, rng)
                 for _ in 1:n]
        return Expr(:block, stmts...)
    end
end

# =============================================================================
# AbstractGenome interface
# =============================================================================

function initialize(::Type{AntGenome}, primitives::Vector{Symbol},
                    conditions::Vector{Symbol}, max_depth::Int,
                    rng::AbstractRNG)
    program = _random_ant_program(primitives, conditions, max_depth, rng)
    AntGenome(program, primitives, conditions, max_depth)
end

function mutate(g::AntGenome, rng::AbstractRNG)
    new_prog = deepcopy(g.program)
    nodes = unravel(new_prog)
    if isempty(nodes)
        return AntGenome(_random_ant_program(g.primitives, g.conditions,
                                             g.max_depth, rng),
                         g.primitives, g.conditions, g.max_depth)
    end

    # Replace a random subtree with a new random subtree
    target_idx = rand(rng, 1:length(nodes))
    replacement = _random_ant_program(g.primitives, g.conditions,
                                      max(1, g.max_depth - 2), rng)

    if target_idx == 1
        return AntGenome(replacement, g.primitives, g.conditions, g.max_depth)
    end

    # Try to replace in the tree
    for i in 1:length(new_prog.args)
        if new_prog.args[i] isa Expr && new_prog.args[i] === nodes[target_idx]
            new_prog.args[i] = replacement
            return AntGenome(new_prog, g.primitives, g.conditions, g.max_depth)
        end
    end

    # Nested replacement
    for node in nodes
        if node !== nodes[target_idx] && node isa Expr
            for i in 1:length(node.args)
                if node.args[i] isa Expr && node.args[i] === nodes[target_idx]
                    node.args[i] = replacement
                    return AntGenome(new_prog, g.primitives, g.conditions, g.max_depth)
                end
            end
        end
    end

    return AntGenome(new_prog, g.primitives, g.conditions, g.max_depth)
end

function crossover(g1::AntGenome, g2::AntGenome, rng::AbstractRNG)
    p1 = deepcopy(g1.program)
    p2 = deepcopy(g2.program)
    nodes1 = unravel(p1)
    nodes2 = unravel(p2)

    if isempty(nodes1) || isempty(nodes2)
        return (g1, g2)
    end

    # Swap random subtrees
    idx1 = rand(rng, 1:length(nodes1))
    idx2 = rand(rng, 1:length(nodes2))
    sub1 = deepcopy(nodes1[idx1])
    sub2 = deepcopy(nodes2[idx2])

    c1_prog = idx1 == 1 ? sub2 : begin
        replace_subtree!(p1, nodes1[idx1], sub2); p1
    end
    c2_prog = idx2 == 1 ? sub1 : begin
        replace_subtree!(p2, nodes2[idx2], sub1); p2
    end

    return (AntGenome(c1_prog, g1.primitives, g1.conditions, g1.max_depth),
            AntGenome(c2_prog, g1.primitives, g1.conditions, g1.max_depth))
end

function distance(g1::AntGenome, g2::AntGenome)
    Float64(abs(length(unravel(g1.program)) - length(unravel(g2.program))))
end

function complexity(g::AntGenome)
    Float64(length(unravel(g.program)))
end

function serialize(g::AntGenome)
    repr(g.program)
end

function deserialize(::Type{AntGenome}, s::String,
                     primitives::Vector{Symbol}, conditions::Vector{Symbol},
                     max_depth::Int)
    try
        expr = Meta.parse(s)
        expr isa Expr || return nothing
        if expr.head == :quote && length(expr.args) == 1 && expr.args[1] isa Expr
            expr = expr.args[1]
        end
        return AntGenome(expr, primitives, conditions, max_depth)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

# =============================================================================
# AntEvaluator
# =============================================================================

"""
    AntEvaluator <: AbstractEvaluator

Fitness evaluator for `AntGenome`. Compiles and executes the evolved
program with an AntSimulator. Returns the number of uneaten food
pellets as fitness (lower is better, 0 = perfect).
"""
struct AntEvaluator <: AbstractEvaluator
    food_positions::Vector{Tuple{Int,Int}}
    max_moves::Int
end

input_signature(::AntEvaluator) = Dict(:dummy => Bool)
output_signature(::AntEvaluator) = Dict(:result => Bool)

function evaluate(e::AntEvaluator, f::Function)
    ant = AntSimulator(e.food_positions, e.max_moves)
    _ant_sim_ref[] = ant
    n_food = length(e.food_positions)
    max_calls = e.max_moves * 2
    calls = 0
    while ant.moves < ant.max_moves && ant.food_eaten < n_food
        moves_before = ant.moves
        try
            Base.invokelatest(f)
        catch e
            e isa InterruptException && rethrow()
            break
        end
        calls += 1
        if ant.moves == moves_before || calls >= max_calls
            break
        end
    end
    return Float64(n_food - ant.food_eaten)
end

"""
    evaluate_genome(g::AntGenome, e::AntEvaluator) -> Float64

Compile and evaluate an AntGenome against the ant trail evaluator.
"""
function evaluate_genome(g::AntGenome, e::AntEvaluator)
    fname = gensym("ant_evolved")
    try
        func_expr = Expr(:function, Expr(:call, fname), g.program)
        f = @eval $func_expr
        return evaluate(e, f)
    catch e
        e isa InterruptException && rethrow()
        return Inf
    end
end

# =============================================================================
# AntGenome solve method
# =============================================================================

"""
    solve(problem::GPProblem{AntGenome}, algorithm::GeneticProgramming; ...) -> GPResult

Run GP evolution with AntGenome for side-effectful program synthesis.

!!! warning
    AntGenome uses a module-level simulator reference (`_ant_sim_ref`) that is
    not thread-safe. The `parallel` field on `algorithm` must be `false`.
    For parallel side-effectful evaluation, use thread-local state as
    demonstrated in the bin packing example.
"""
function solve(problem::GPProblem{AntGenome, E},
               algorithm::GeneticProgramming;
               verbose::Bool = false,
               callback = nothing) where {E<:AntEvaluator}
    if algorithm.parallel
        error("AntGenome uses module-level simulator state (_ant_sim_ref) that is " *
              "not thread-safe. Set parallel=false in GeneticProgramming. " *
              "For parallel side-effectful evaluation, use thread-local state " *
              "as demonstrated in examples/bin_packing.jl.")
    end
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    evaluator = problem.evaluator
    pop_size = algorithm.pop_size

    # Default ant primitives
    primitives = [:gp_ant_move, :gp_ant_left, :gp_ant_right]
    conditions = [:gp_ant_food_ahead]

    genomes = [initialize(AntGenome, primitives, conditions, 4, rng)
               for _ in 1:pop_size]
    fitnesses = fill(Inf, pop_size)

    for i in 1:pop_size
        fitnesses[i] = evaluate_genome(genomes[i], evaluator)
    end

    species_state = _init_species_state(algorithm.speciation)
    fitness_history = Float64[]
    mean_history = Float64[]
    t0 = time()

    for gen in 1:algorithm.generations
        order = sortperm(fitnesses)
        genomes = genomes[order]
        fitnesses = fitnesses[order]

        push!(fitness_history, fitnesses[1])
        finite_fits = filter(isfinite, fitnesses)
        mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
        push!(mean_history, mean_fit)

        if verbose
            println("Generation $gen: best=$(round(fitnesses[1], digits=6))")
            flush(stdout)
        end

        callback !== nothing && callback(gen, fitnesses[1], genomes[1])

        selection_fitnesses = _apply_speciation!(genomes, fitnesses,
                                                  algorithm.speciation, species_state, rng)

        next_genomes = Vector{AntGenome}(undef, pop_size)
        next_fitnesses = fill(Inf, pop_size)

        for i in 1:min(algorithm.elitism, pop_size)
            next_genomes[i] = AntGenome(deepcopy(genomes[i].program),
                                        genomes[i].primitives, genomes[i].conditions,
                                        genomes[i].max_depth)
            next_fitnesses[i] = fitnesses[i]
        end

        t_size = algorithm.selection.tournament_size

        idx = algorithm.elitism + 1
        while idx <= pop_size
            r = rand(rng)
            if r < algorithm.crossover_rate && idx + 1 <= pop_size
                p1 = _tournament_select(selection_fitnesses, t_size, rng)
                p2 = _tournament_select(selection_fitnesses, t_size, rng)
                (c1, c2) = crossover(genomes[p1], genomes[p2], rng)
                next_genomes[idx] = c1
                next_genomes[idx + 1] = c2
                idx += 2
            elseif r < algorithm.crossover_rate + algorithm.mutation_rate
                p_idx = _tournament_select(selection_fitnesses, t_size, rng)
                next_genomes[idx] = mutate(genomes[p_idx], rng)
                idx += 1
            else
                p_idx = _tournament_select(selection_fitnesses, t_size, rng)
                next_genomes[idx] = AntGenome(deepcopy(genomes[p_idx].program),
                                              genomes[p_idx].primitives,
                                              genomes[p_idx].conditions,
                                              genomes[p_idx].max_depth)
                idx += 1
            end
        end

        for i in (algorithm.elitism + 1):pop_size
            next_fitnesses[i] = evaluate_genome(next_genomes[i], evaluator)
        end

        genomes = next_genomes
        fitnesses = next_fitnesses
    end

    order = sortperm(fitnesses)
    genomes = genomes[order]
    fitnesses = fitnesses[order]

    return GPResult{AntGenome}(
        genomes[1], fitnesses[1], genomes,
        fitness_history, mean_history,
        algorithm.generations, time() - t0,
        fitnesses[1] < 1.0
    )
end
