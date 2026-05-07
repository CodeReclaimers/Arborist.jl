# cmaes.jl — Covariance Matrix Adaptation Evolution Strategy (Hansen 2016).
#
# In-tree implementation. Closes the F-survey gap "no continuous-policy
# optimizer for fixed-topology neuroevolution / weight-only search" without
# pulling Evolutionary.jl or CMAEvolutionStrategy.jl as deps.
#
# Operates on real-valued vectors. To use with `GraphGenome`, freeze the
# topology and let CMA-ES optimize the connection weights via the
# `flatten_weights` / `unflatten_weights!` interface defined in this file.
# For other genome types, define those two methods to opt in.
#
# Algorithm: standard (μ_w/μ, λ)-CMA-ES with rank-1 + rank-μ updates,
# evolution-path step-size adaptation. Reference: Hansen 2016 tutorial.
# Defaults follow the standard recommendations:
#   λ = 4 + floor(3 ln n)   (population size)
#   μ = floor(λ/2)
#   weights = log(μ + 1/2) - log(1..μ) (then normalized)

using LinearAlgebra: I, Diagonal, Symmetric, eigen, norm, mul!

# ---------------------------------------------------------------------------
# Algorithm config
# ---------------------------------------------------------------------------

"""
    CMAES <: AbstractEvolutionaryAlgorithm

Covariance Matrix Adaptation Evolution Strategy. Suitable for continuous
parameter optimization where the search dimension is fixed at solve time.
Pairs with `GraphGenome` via `flatten_weights` / `unflatten_weights!` to
optimize connection weights under a frozen topology.

# Fields (all kwargs, sensible defaults)
- `generations::Int = 100`: outer iteration count.
- `pop_size::Int = 0`: population size λ. `0` means auto: `4 + floor(3 ln n)`,
  computed at solve time once n (the parameter count) is known.
- `sigma0::Float64 = 1.0`: initial step size.
- `parallel::Bool = true`: thread the per-generation evaluations.
- `convergence_threshold::Float64 = -Inf`: stop early when best fitness
  drops below this (lower-is-better convention; default never triggers).
- `seed_genome::Bool = true`: when true (default), the starting mean is
  taken from the problem's initial genome; when false, drawn from N(0, I)
  scaled by sigma0.
"""
struct CMAES <: AbstractEvolutionaryAlgorithm
    generations::Int
    pop_size::Int
    sigma0::Float64
    parallel::Bool
    convergence_threshold::Float64
    seed_genome::Bool
end

function CMAES(; generations::Int = 100,
                pop_size::Int = 0,
                sigma0::Float64 = 1.0,
                parallel::Bool = true,
                convergence_threshold::Float64 = -Inf,
                seed_genome::Bool = true)
    generations >= 1 || throw(ArgumentError("generations must be >= 1"))
    pop_size >= 0 || throw(ArgumentError("pop_size must be >= 0"))
    sigma0 > 0.0 || throw(ArgumentError("sigma0 must be positive"))
    CMAES(generations, pop_size, sigma0, parallel, convergence_threshold, seed_genome)
end

# ---------------------------------------------------------------------------
# Genome <-> real vector adaptors
# ---------------------------------------------------------------------------

"""
    flatten_weights(g::AbstractGenome) -> Vector{Float64}

Return the genome's continuous parameters as a Float64 vector. Defined
for `GraphGenome` over its enabled connection weights, sorted by
innovation number for determinism.

Raises MethodError for genome types that don't expose a continuous
parameter vector — those genomes can't be searched by CMA-ES directly.
"""
function flatten_weights end

"""
    unflatten_weights!(g::AbstractGenome, w::Vector{Float64}) -> AbstractGenome

Inverse of `flatten_weights`: writes `w` back into `g`'s continuous
parameters. Mutates `g` and returns it. Length of `w` must match
`length(flatten_weights(g))`.
"""
function unflatten_weights! end

# GraphGenome adaptor: connection weights, deterministic order.
function flatten_weights(g::GraphGenome)
    sorted_innovs = sort!(collect(keys(g.connections)))
    return Float64[g.connections[i].weight for i in sorted_innovs]
end

function unflatten_weights!(g::GraphGenome, w::Vector{Float64})
    sorted_innovs = sort!(collect(keys(g.connections)))
    length(w) == length(sorted_innovs) || throw(ArgumentError(
        "weight vector length ($(length(w))) does not match connection count " *
        "($(length(sorted_innovs)))"))
    for (i, innov) in enumerate(sorted_innovs)
        g.connections[innov].weight = w[i]
    end
    return g
end

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

"""
    solve(problem::GPProblem{G,E}, alg::CMAES; verbose=false, log=nothing) -> GPResult{G}

Run CMA-ES against the problem's evaluator on a fixed-topology genome.
Requires `flatten_weights(g)` / `unflatten_weights!(g, w)` to be defined
for `G`. The initial genome (constructed via `_initialize_population`)
provides both the topology and, when `alg.seed_genome=true`, the starting
mean of the search distribution.

Returns a `GPResult{G}` with the best genome found.
"""
function solve(problem::GPProblem{G,E}, alg::CMAES;
               verbose::Bool = false,
               callback = nothing,
               log::Union{Nothing, RunLog} = nothing) where {G, E<:AbstractEvaluator}
    rng = problem.seed === nothing ? Random.default_rng() :
          Random.MersenneTwister(problem.seed)

    # Use a throwaway GP config to reach _initialize_population; we just
    # want one genome to provide the topology.
    init_gp = GeneticProgramming(
        pop_size=1, generations=1, elitism=0, parallel=false,
        mutation_ops=AbstractMutationOperator[],
        crossover_ops=AbstractCrossoverOperator[],
        selection=TournamentSelection(1),
    )
    pop_tuple = try
        _initialize_population(problem, init_gp, rng)
    catch err
        # Some _initialize_population paths require non-empty operator lists
        # for validation. Fall back to a direct initialize for GraphGenome.
        if G === GraphGenome
            n_in = length(input_signature(problem.evaluator))
            n_out = length(output_signature(problem.evaluator))
            ([initialize(GraphGenome, n_in, n_out, rng)], nothing)
        else
            rethrow(err)
        end
    end
    seed_genome = pop_tuple[1][1]

    # Determine search dimension and initial mean.
    initial_w = flatten_weights(seed_genome)
    n = length(initial_w)
    n >= 1 || throw(ArgumentError(
        "CMA-ES needs at least 1 parameter; flatten_weights returned an empty vector"))

    mean_x = alg.seed_genome ? copy(initial_w) : alg.sigma0 .* randn(rng, n)
    sigma = alg.sigma0

    # Population size (Hansen default: 4 + floor(3 ln n)).
    # NB: `log` is reserved as a kwarg in this function — use Base.log here.
    lambda = alg.pop_size > 0 ? alg.pop_size : 4 + Int(floor(3.0 * Base.log(n)))
    mu = max(1, lambda ÷ 2)

    # Recombination weights (positive, normalized).
    raw_w = [Base.log(mu + 0.5) - Base.log(Float64(i)) for i in 1:mu]
    weights = raw_w ./ sum(raw_w)
    mueff = 1.0 / sum(weights .^ 2)

    # Strategy parameters (Hansen 2016).
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    cc = (4.0 + mueff/n) / (n + 4.0 + 2.0 * mueff/n)
    c1 = 2.0 / ((n + 1.3)^2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0/mueff) / ((n + 2.0)^2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, sqrt((mueff - 1.0)/(n + 1.0)) - 1.0) + cs

    # Evolution paths and covariance.
    p_sigma = zeros(n)
    p_c = zeros(n)
    C = Matrix{Float64}(I(n))
    chi_n = sqrt(Float64(n)) * (1.0 - 1.0/(4.0 * n) + 1.0/(21.0 * n * n))

    # Bookkeeping.
    best_x = copy(mean_x)
    best_f = Inf
    fitness_history = Float64[]
    mean_history = Float64[]
    t0 = time()

    # Workspace.
    pop_genomes = Vector{G}(undef, lambda)
    pop_fits = fill(Inf, lambda)

    # Helper: write w into a copy of seed_genome and evaluate.
    function _eval_at(w::Vector{Float64})
        g = deepcopy(seed_genome)
        unflatten_weights!(g, w)
        return g, evaluate_genome(g, problem.evaluator)
    end

    for gen in 1:alg.generations
        # --- Sample λ candidates from N(mean, sigma^2 * C) ---
        # Eigendecompose C; ensure symmetric (numerical drift safeguard).
        Csym = Symmetric(C)
        eig = eigen(Csym)
        # Clamp tiny negative eigenvalues that arise from rounding.
        eigvals = max.(eig.values, 0.0)
        D = sqrt.(eigvals)
        BD = eig.vectors * Diagonal(D)

        zs = [randn(rng, n) for _ in 1:lambda]
        ys = [BD * z for z in zs]
        xs = [mean_x .+ sigma .* y for y in ys]

        # Evaluate; serialize per-individual to avoid contention on BD/Csym.
        if alg.parallel && Threads.nthreads() > 1
            Threads.@threads for i in 1:lambda
                g, f = _eval_at(xs[i])
                pop_genomes[i] = g
                pop_fits[i] = f
            end
        else
            for i in 1:lambda
                g, f = _eval_at(xs[i])
                pop_genomes[i] = g
                pop_fits[i] = f
            end
        end

        # --- Selection: best mu by fitness (lower is better) ---
        order = sortperm(pop_fits)

        # Track best ever.
        for i in 1:lambda
            if pop_fits[i] < best_f
                best_f = pop_fits[i]
                best_x = copy(xs[i])
            end
        end
        push!(fitness_history, best_f)
        finite_fits = filter(isfinite, pop_fits)
        mean_fit = isempty(finite_fits) ? Inf : sum(finite_fits) / length(finite_fits)
        push!(mean_history, mean_fit)

        if verbose
            println("CMA-ES gen=$gen: best=$(round(best_f, digits=6)), " *
                    "mean=$(round(mean_fit, digits=6)), sigma=$(round(sigma, digits=4))")
            flush(stdout)
        end

        if callback !== nothing
            callback(gen, best_f, pop_genomes[order[1]])
        end

        if log !== nothing
            record!(log, gen, pop_fits, pop_genomes, time() - t0)
        end

        # --- Recombination: weighted mean of best mu ---
        old_mean = copy(mean_x)
        new_mean = zeros(n)
        for i in 1:mu
            new_mean .+= weights[i] .* xs[order[i]]
        end

        # --- Step-size update via evolution path ---
        # C^(-1/2) * (m_new - m_old)/sigma  =  B * D^(-1) * B^T * (m_new - m_old)/sigma
        Dinv_safe = [d > 1e-12 ? 1.0/d : 0.0 for d in D]
        BDinv = eig.vectors * Diagonal(Dinv_safe)
        # Compute z = B * D^(-1) * B^T * v
        diff_norm = (new_mean .- old_mean) ./ sigma
        z = BDinv * (eig.vectors' * diff_norm)
        p_sigma = (1.0 - cs) .* p_sigma .+ sqrt(cs * (2.0 - cs) * mueff) .* z

        sigma *= exp((cs / damps) * (norm(p_sigma) / chi_n - 1.0))
        # Cap sigma against runaway growth.
        sigma = min(sigma, 1e6)

        # --- Covariance update ---
        # Heaviside step in p_c update.
        hsig = norm(p_sigma) / sqrt(1.0 - (1.0 - cs)^(2.0 * gen)) <
               (1.4 + 2.0 / (n + 1.0)) * chi_n
        p_c = (1.0 - cc) .* p_c .+ (hsig ? sqrt(cc * (2.0 - cc) * mueff) : 0.0) .* (new_mean .- old_mean) ./ sigma

        # Rank-1 update.
        rank1 = c1 .* (p_c * p_c')
        # Rank-μ update.
        artmp = zeros(n, mu)
        @inbounds for i in 1:mu
            artmp[:, i] = (xs[order[i]] .- old_mean) ./ sigma
        end
        rankmu = cmu .* artmp * Diagonal(weights) * artmp'
        # Compensate for rank-1 path attenuation when hsig=false.
        delta_hsig = (1.0 - Float64(hsig)) * cc * (2.0 - cc)

        C = (1.0 - c1 - cmu + delta_hsig * c1) .* C .+ rank1 .+ rankmu
        # Symmetrize to guard against numerical asymmetry drift.
        C = 0.5 .* (C .+ C')

        mean_x = new_mean

        # Convergence check.
        if best_f < alg.convergence_threshold
            break
        end
    end

    # Final result genome at best_x.
    final_g = deepcopy(seed_genome)
    unflatten_weights!(final_g, best_x)
    wall_time = time() - t0

    # Synthesize a final population from the last generation's order.
    final_population = pop_genomes[sortperm(pop_fits)]
    final_fits = sort(pop_fits)

    return GPResult{G}(
        final_g,
        best_f,
        final_population,
        fitness_history,
        mean_history,
        alg.generations,
        wall_time,
        best_f < alg.convergence_threshold,
    )
end
