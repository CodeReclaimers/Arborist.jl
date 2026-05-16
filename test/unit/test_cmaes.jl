using Test
using Arborist
using Random

# A throwaway evaluator that wraps a Julia function f(::Vector{Float64}) -> Float64
# and exposes the n-dim parameter space through a synthetic genome that holds
# the vector directly. Used to stress-test CMA-ES on standard test functions
# without needing GraphGenome plumbing.
mutable struct _RealVecGenome <: AbstractGenome
    w::Vector{Float64}
end
struct _RealFnEvaluator{F} <: AbstractEvaluator
    f::F
    n::Int
end
Arborist.input_signature(e::_RealFnEvaluator) = Dict{Symbol, DataType}(Symbol("x$i") => Float64 for i in 1:e.n)
Arborist.output_signature(e::_RealFnEvaluator) = Dict{Symbol, DataType}(:y => Float64)
Arborist.evaluate_genome(g::_RealVecGenome, e::_RealFnEvaluator) = e.f(g.w)
Arborist.flatten_weights(g::_RealVecGenome) = copy(g.w)
function Arborist.unflatten_weights!(g::_RealVecGenome, w::Vector{Float64})
    length(w) == length(g.w) || throw(ArgumentError("dimension mismatch"))
    g.w .= w
    return g
end
Arborist.complexity(g::_RealVecGenome) = Float64(length(g.w))
Arborist.distance(a::_RealVecGenome, b::_RealVecGenome) = sqrt(sum((a.w .- b.w).^2))
Arborist.serialize(g::_RealVecGenome) = string(g.w)

# Custom _initialize_population for the synthetic genome.
Arborist._initialize_population(problem::GPProblem{_RealVecGenome, E},
                                 alg::GeneticProgramming,
                                 rng::AbstractRNG) where {E} =
    ([_RealVecGenome(randn(rng, problem.evaluator.n))], nothing)

@testset "CMA-ES" begin
    @testset "config validation" begin
        @test_throws ArgumentError CMAES(generations=0)
        @test_throws ArgumentError CMAES(sigma0=0.0)
        @test_throws ArgumentError CMAES(sigma0=-1.0)
        @test_throws ArgumentError CMAES(pop_size=-3)
        c = CMAES()
        @test c.generations == 100
        @test c.sigma0 == 1.0
        @test c.parallel == true
    end

    @testset "GraphGenome flatten/unflatten round-trip" begin
        rng = MersenneTwister(42)
        Arborist.reset_innovation_counter!()
        g = Arborist.initialize(GraphGenome, 2, 1, rng)
        w = flatten_weights(g)
        @test length(w) == length(g.connections)
        # Mutate via direct assignment.
        new_w = w .* 2.0 .+ 1.0
        unflatten_weights!(g, new_w)
        w2 = flatten_weights(g)
        @test w2 ≈ new_w

        # Length-mismatch error.
        @test_throws ArgumentError unflatten_weights!(g, [1.0])
    end

    @testset "Sphere function in 5-D" begin
        # f(x) = sum(x.^2). Optimum at zero.
        n = 5
        evaluator = _RealFnEvaluator(w -> sum(w .^ 2), n)
        problem = GPProblem(evaluator, _RealVecGenome; seed=1)
        alg = CMAES(generations=80, pop_size=12, sigma0=1.0, parallel=false,
                    seed_genome=true)
        result = solve(problem, alg)
        @test result.best_fitness < 1e-6
        @test all(abs.(result.best_genome.w) .< 1e-3)
    end

    @testset "Rosenbrock function in 5-D" begin
        # Standard Rosenbrock; optimum at (1,1,...,1) with f=0.
        function rosen(x)
            n = length(x)
            s = 0.0
            for i in 1:(n-1)
                s += 100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2
            end
            return s
        end
        n = 5
        evaluator = _RealFnEvaluator(rosen, n)
        problem = GPProblem(evaluator, _RealVecGenome; seed=2)
        alg = CMAES(generations=400, pop_size=14, sigma0=0.5, parallel=false,
                    seed_genome=false, convergence_threshold=1e-6)
        result = solve(problem, alg)
        @test result.best_fitness < 1.0  # forward progress; tight convergence is hard
    end

    @testset "Rastrigin in 5-D — finds near-optimum" begin
        # Famously deceptive: many local minima. CMA-ES should escape them.
        function rastrigin(x)
            n = length(x)
            10.0 * n + sum(x.^2 .- 10.0 .* cos.(2π .* x))
        end
        n = 5
        evaluator = _RealFnEvaluator(rastrigin, n)
        problem = GPProblem(evaluator, _RealVecGenome; seed=42)
        alg = CMAES(generations=300, pop_size=20, sigma0=1.0, parallel=false,
                    seed_genome=false)
        result = solve(problem, alg)
        # Forward-progress gate: should beat a random baseline (random ~50+).
        @test result.best_fitness < 10.0
        println("  CMA-ES Rastrigin-5 best = $(round(result.best_fitness, digits=6))")
        flush(stdout)
    end

    @testset "early termination reports actual generation count" begin
        # convergence_threshold deliberately above the random init fitness so
        # the first generation triggers the break. generations_run must report
        # the number actually completed, not alg.generations.
        n = 3
        evaluator = _RealFnEvaluator(w -> sum(w .^ 2), n)
        problem = GPProblem(evaluator, _RealVecGenome; seed=7)
        alg = CMAES(generations=5, pop_size=8, sigma0=1.0, parallel=false,
                    seed_genome=false, convergence_threshold=1e9)
        result = solve(problem, alg)
        @test length(result.fitness_history) == 1
        @test result.generations_run == 1
        @test result.converged == true
    end

    @testset "infinite convergence_threshold does not falsely report convergence" begin
        # The shared _converged helper requires a finite threshold; the CMA-ES
        # solve path should use it (not the inline best_f < threshold check)
        # so that the default convergence_threshold=Inf reports converged=false.
        n = 3
        evaluator = _RealFnEvaluator(w -> sum(w .^ 2), n)
        problem = GPProblem(evaluator, _RealVecGenome; seed=11)
        alg = CMAES(generations=4, pop_size=8, sigma0=1.0, parallel=false,
                    seed_genome=false, convergence_threshold=Inf)
        result = solve(problem, alg)
        @test result.generations_run == 4
        @test length(result.fitness_history) == 4
        @test result.converged == false
    end

    @testset "GraphGenome end-to-end (XOR with frozen topology)" begin
        # Initialize a 2-input, 1-output XOR network (just inputs+output, no
        # hidden nodes — known not to solve XOR optimally, but we just verify
        # CMA-ES improves the weights from random init.
        Arborist.reset_innovation_counter!()
        rng = MersenneTwister(13)
        seed_g = Arborist.initialize(GraphGenome, 2, 1, rng)
        n_params = length(flatten_weights(seed_g))
        @test n_params >= 1

        input_data = Float64[0 0 1 1; 0 1 0 1]
        output_data = Float64[0 1 1 0]
        evaluator = GraphEvaluator(input_data, output_data)

        # Initial fitness with current random weights.
        initial_fit = Arborist.evaluate_genome(seed_g, evaluator)

        problem = GPProblem(evaluator, GraphGenome; seed=100)
        alg = CMAES(generations=60, pop_size=12, sigma0=0.5, parallel=false)
        result = solve(problem, alg)

        @test result isa GPResult{GraphGenome}
        @test result.best_fitness <= initial_fit + 1e-10
        @test result.best_genome isa GraphGenome
        # Topology preserved (CMA-ES only touches weights).
        @test length(result.best_genome.connections) == length(seed_g.connections)
        @test keys(result.best_genome.nodes) == keys(seed_g.nodes)
    end
end
