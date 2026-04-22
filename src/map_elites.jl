# map_elites.jl — MAP-Elites (Mouret & Clune 2015) quality-diversity algorithm.
#
# Quality-Diversity search: instead of finding a single best solution, maintain
# an ARCHIVE of the best solution found in each cell of a discretized feature
# space. Over the run, the archive both improves (each cell's best gets better)
# and illuminates (more cells filled). Useful for:
# - Discovering diverse strategies in control tasks.
# - Exploring the structure-behavior relationship (e.g. hidden_nodes ×
#   n_connections for NEAT genomes).
# - Robust solutions: a filled archive gives you many candidates, not one
#   brittle optimum.
#
# Design (per Phase F D4):
# - Grid-based archive: user specifies per-feature bounds + bin counts.
# - Each iteration: sample a parent from the archive, mutate/crossover,
#   evaluate, place into the corresponding bin if it beats the incumbent.
# - Initial `n_init` genomes are randomly generated to seed the archive.

"""
    MAPElitesArchive{G}

Grid-based archive for MAP-Elites. Maps a cell index (an `N`-tuple of
integers) to a `(genome, fitness)` pair. Fitness is lower-is-better,
matching the framework convention.

Not thread-safe — MAP-Elites runs its own evaluation batches and
serializes archive updates; parallel evaluation happens within a batch,
insertion after the batch synchronizes.
"""
mutable struct MAPElitesArchive{G}
    grid::Dict{Tuple, Pair{G, Float64}}
    n_dims::Int
    n_bins::Vector{Int}
end

MAPElitesArchive{G}(n_bins::Vector{Int}) where G =
    MAPElitesArchive{G}(Dict{Tuple, Pair{G, Float64}}(), length(n_bins), copy(n_bins))

Base.length(a::MAPElitesArchive) = length(a.grid)
Base.isempty(a::MAPElitesArchive) = isempty(a.grid)

"""
    coverage(a::MAPElitesArchive) -> Float64

Fraction of bins in the grid that hold a genome. Ranges [0, 1].
"""
coverage(a::MAPElitesArchive) = length(a) / prod(a.n_bins)

"""
    qd_score(a::MAPElitesArchive; worst_fitness=0.0) -> Float64

Quality-Diversity score: sum over filled cells of `(worst_fitness - f_cell)`.
Rewards archives that are both deeply filled and broad. Since the framework
uses lower-is-better fitness, larger QD score means the archive holds
better solutions across more bins. Pass `worst_fitness = maximum` of the
task's plausible loss range for a meaningful score.
"""
function qd_score(a::MAPElitesArchive; worst_fitness::Float64 = 0.0)
    s = 0.0
    for (_, pair) in a.grid
        f = pair.second
        isfinite(f) || continue
        s += (worst_fitness - f)
    end
    return s
end

# ---------------------------------------------------------------------------
# Algorithm config
# ---------------------------------------------------------------------------

"""
    MAPElites{F, FB, MO, CO} <: AbstractEvolutionaryAlgorithm

Illuminates a user-defined feature space by evolving a grid-based archive.

# Fields (constructor kwargs with defaults)
- `generations::Int = 100`: number of outer iterations (each generation
  samples and evaluates `batch_size` children from the archive).
- `batch_size::Int = 50`: mutations produced per generation.
- `n_init::Int = 50`: random genomes generated at generation 0 to seed
  the archive before any sampling happens.
- `feature_fn::F`: genome -> `NTuple{N, Float64}`. Behavior descriptor
  used to place genomes into the discretized grid.
- `feature_bounds::FB`: `Vector{Tuple{Float64,Float64}}` of per-dimension
  `(low, high)` intervals. Features outside the range clamp to the
  extreme bin.
- `n_bins::Vector{Int}`: per-dimension bin count. Total archive capacity
  is `prod(n_bins)`.
- `mutation_ops::MO`: vector of mutation operators (as in `GeneticProgramming`).
- `crossover_ops::CO`: optional vector of crossover operators. If empty,
  every child is produced by mutation only.
- `crossover_rate::Float64 = 0.0`: probability of picking crossover over
  mutation when both are available.
- `parallel::Bool = true`: threaded evaluation of a batch.
- `elitism::Int = 0`: placeholder for interface parity; MAP-Elites has
  archive-based elitism intrinsically, so this is unused.
"""
struct MAPElites{F, FB, MO, CO} <: AbstractEvolutionaryAlgorithm
    generations::Int
    batch_size::Int
    n_init::Int
    feature_fn::F
    feature_bounds::FB
    n_bins::Vector{Int}
    mutation_ops::MO
    crossover_ops::CO
    crossover_rate::Float64
    parallel::Bool
end

function MAPElites(;
    feature_fn::F,
    feature_bounds::FB,
    n_bins::Vector{Int},
    mutation_ops::MO,
    crossover_ops::CO = AbstractCrossoverOperator[],
    generations::Int = 100,
    batch_size::Int = 50,
    n_init::Int = 50,
    crossover_rate::Float64 = 0.0,
    parallel::Bool = true,
) where {F, FB, MO, CO}
    length(feature_bounds) == length(n_bins) || throw(ArgumentError(
        "feature_bounds and n_bins must have the same length (got " *
        "$(length(feature_bounds)) and $(length(n_bins)))"))
    all(>(0), n_bins) || throw(ArgumentError("n_bins must be all positive"))
    0.0 <= crossover_rate <= 1.0 || throw(ArgumentError(
        "crossover_rate must be in [0, 1] (got $crossover_rate)"))
    generations >= 1 || throw(ArgumentError("generations must be >= 1"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be >= 1"))
    n_init >= 1 || throw(ArgumentError("n_init must be >= 1"))
    MAPElites{F, FB, MO, CO}(generations, batch_size, n_init,
                              feature_fn, feature_bounds, n_bins,
                              mutation_ops, crossover_ops, crossover_rate,
                              parallel)
end

"""
    MAPElitesResult{G}

Result returned by `solve(problem, ::MAPElites)`. Carries the final archive
and per-generation coverage/QD-score trajectories.

# Fields
- `archive::MAPElitesArchive{G}`: final grid.
- `coverage_history::Vector{Float64}`: coverage fraction per generation.
- `qd_score_history::Vector{Float64}`: qd_score per generation.
- `best_genome::G`: the single best (lowest-fitness) genome in the archive.
- `best_fitness::Float64`: its fitness.
- `wall_time::Float64`: seconds elapsed.
- `generations_run::Int`: outer iterations completed.
"""
struct MAPElitesResult{G} <: AbstractEvolutionResult
    archive::MAPElitesArchive{G}
    coverage_history::Vector{Float64}
    qd_score_history::Vector{Float64}
    best_genome::G
    best_fitness::Float64
    wall_time::Float64
    generations_run::Int
end

# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

"""
    _cell_index(features, bounds, n_bins) -> Tuple{Int...}

Map a feature vector to its grid cell. Features below `low` clamp to bin 1;
features at or above `high` clamp to bin `n_bins[d]`.
"""
function _cell_index(features, bounds, n_bins)
    n = length(features)
    idx = Vector{Int}(undef, n)
    @inbounds for d in 1:n
        lo, hi = bounds[d]
        b = n_bins[d]
        x = Float64(features[d])
        if x <= lo
            idx[d] = 1
        elseif x >= hi
            idx[d] = b
        else
            # Linear binning. Uses floor after rescaling to [0, b).
            pos = (x - lo) / (hi - lo) * b
            cell = clamp(Int(floor(pos)) + 1, 1, b)
            idx[d] = cell
        end
    end
    return Tuple(idx)
end

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

"""
    solve(problem::GPProblem{G,E}, alg::MAPElites; verbose=false, log=nothing) -> MAPElitesResult{G}

Run MAP-Elites. Requires `problem.evaluator` to implement
`evaluate_genome(g, e)`. The feature function and grid are supplied via
`alg`. Seeds the archive with `alg.n_init` random individuals via the
problem's `_initialize_population` path, then alternates generations of
(sample parent from archive → mutate/crossover → evaluate → maybe insert).

`log::Union{Nothing, RunLog}` records per-generation coverage and qd_score
alongside the standard fitness stats (best from archive as a scalar).
"""
function solve(problem::GPProblem{G,E}, alg::MAPElites;
               verbose::Bool = false,
               log::Union{Nothing, RunLog} = nothing) where {G, E}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    # Use the same init path as GeneticProgramming so custom genome types work.
    # We piggyback on `_initialize_population` which takes a GeneticProgramming;
    # build a minimal throwaway to harvest the population.
    throwaway_alg = GeneticProgramming(
        pop_size = alg.n_init,
        generations = 1,
        mutation_rate = 0.0, crossover_rate = 0.0, elitism = 0,
        parallel = alg.parallel,
        mutation_ops = convert(Vector{AbstractMutationOperator}, alg.mutation_ops),
        crossover_ops = isempty(alg.crossover_ops) ?
            AbstractCrossoverOperator[SubtreeCrossover()] :
            convert(Vector{AbstractCrossoverOperator}, alg.crossover_ops),
        selection = TournamentSelection(2),
    )
    pop_tuple = _initialize_population(problem, throwaway_alg, rng)
    init_genomes = pop_tuple[1]

    archive = MAPElitesArchive{G}(alg.n_bins)
    coverage_history = Float64[]
    qd_score_history = Float64[]
    t0 = time()

    # Evaluate and place the seed population.
    init_fits = fill(Inf, length(init_genomes))
    _parallel_evaluate_genome!(init_fits, init_genomes, problem.evaluator,
                               1:length(init_genomes), alg.parallel)
    for (g, f) in zip(init_genomes, init_fits)
        _try_insert!(archive, g, f, alg.feature_fn, alg.feature_bounds, alg.n_bins)
    end
    push!(coverage_history, coverage(archive))
    push!(qd_score_history, qd_score(archive))

    best_genome = init_genomes[argmin(init_fits)]
    best_fitness = minimum(init_fits)

    # Main loop.
    for gen in 1:alg.generations
        isempty(archive) && break  # nothing to sample from

        # Sample `batch_size` parents (uniformly from filled cells) and mutate.
        parents_keys = collect(keys(archive.grid))
        children = Vector{G}(undef, alg.batch_size)
        for i in 1:alg.batch_size
            kp1 = rand(rng, parents_keys)
            p1 = archive.grid[kp1].first
            children[i] = if !isempty(alg.crossover_ops) && rand(rng) < alg.crossover_rate
                kp2 = rand(rng, parents_keys)
                p2 = archive.grid[kp2].first
                op = rand(rng, alg.crossover_ops)
                c1, _ = crossover(op, p1, p2, rng)
                c1
            else
                op = rand(rng, alg.mutation_ops)
                mutate(op, p1, rng)
            end
        end

        # Batch-evaluate.
        child_fits = fill(Inf, alg.batch_size)
        _parallel_evaluate_genome!(child_fits, children, problem.evaluator,
                                   1:alg.batch_size, alg.parallel)

        # Serialized inserts.
        for (g, f) in zip(children, child_fits)
            _try_insert!(archive, g, f, alg.feature_fn, alg.feature_bounds, alg.n_bins)
            if f < best_fitness
                best_fitness = f
                best_genome = g
            end
        end

        push!(coverage_history, coverage(archive))
        push!(qd_score_history, qd_score(archive))

        if verbose
            println("MAP-Elites gen=$gen: coverage=$(round(coverage(archive), digits=3)), " *
                    "qd=$(round(qd_score(archive), digits=3)), " *
                    "best=$(round(best_fitness, digits=6))")
            flush(stdout)
        end

        if log !== nothing
            record!(log, gen, child_fits, children, time() - t0)
        end
    end

    wall_time = time() - t0
    return MAPElitesResult{G}(archive, coverage_history, qd_score_history,
                              best_genome, best_fitness, wall_time, alg.generations)
end

# Evaluate a batch using `evaluate_genome`. No bloat penalty — MAP-Elites uses
# raw fitness; diversity pressure is the grid, not a penalty.
function _parallel_evaluate_genome!(fitnesses::Vector{Float64},
                                    genomes::AbstractVector,
                                    evaluator::AbstractEvaluator,
                                    indices, parallel::Bool)
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in collect(indices)
            fitnesses[i] = try
                evaluate_genome(genomes[i], evaluator)
            catch err
                err isa InterruptException && rethrow()
                Inf
            end
        end
    else
        for i in indices
            fitnesses[i] = try
                evaluate_genome(genomes[i], evaluator)
            catch err
                err isa InterruptException && rethrow()
                Inf
            end
        end
    end
end

function _try_insert!(archive::MAPElitesArchive{G}, g::G, f::Float64,
                      feature_fn, feature_bounds, n_bins) where G
    isfinite(f) || return false
    features = try
        feature_fn(g)
    catch err
        err isa InterruptException && rethrow()
        return false
    end
    features === nothing && return false
    key = _cell_index(features, feature_bounds, n_bins)
    incumbent = get(archive.grid, key, nothing)
    if incumbent === nothing || f < incumbent.second
        archive.grid[key] = Pair{G, Float64}(deepcopy(g), f)
        return true
    end
    return false
end
