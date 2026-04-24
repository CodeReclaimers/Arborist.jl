using Test
using Arborist
using Random
using DynamicExpressions

@testset "Ephemeral Random Constants (ERC)" begin
    @testset "erc_uniform basics" begin
        rng = MersenneTwister(0)
        sampler = erc_uniform(-5.0f0, 5.0f0)
        # Draw many samples; all must be in range and type.
        for _ in 1:200
            v = sampler(rng)
            @test v isa Float32
            @test -5.0f0 <= v <= 5.0f0
        end

        # Degenerate range (lo == hi) is allowed; produces a point mass.
        point = erc_uniform(3.5f0, 3.5f0)
        @test point(rng) == 3.5f0

        # Inverted range rejected.
        @test_throws ArgumentError erc_uniform(2.0f0, 1.0f0)
    end

    @testset "sampler is consulted by _random_terminal" begin
        # Install a sentinel sampler via task_local_storage and confirm the
        # sampler path fires (checks the task-local plumbing independent
        # of solve()).
        rng = MersenneTwister(42)
        hits = Ref(0)
        task_local_storage(Arborist._ERC_TLS_KEY,
            (r) -> (hits[] += 1; 42.0f0))
        try
            # Build a constant-only tree by passing n_features=0 (forces
            # the constant branch in _random_terminal).
            node = Arborist._random_terminal(rng, 0, Float32)
            @test node.degree == 0
            @test node.val == 42.0f0
            @test hits[] == 1
        finally
            # Reset so later tests see the default randn path.
            task_local_storage(Arborist._ERC_TLS_KEY, nothing)
        end
    end

    @testset "solve() wires constant_sampler through initial population" begin
        # Use a deterministic sampler that always produces the same value.
        # After one gen of evolution with a linear dataset, at least one
        # constant leaf in some genome must equal the sentinel — confirming
        # the sampler fired during initial population creation.
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        xs = collect(Float32, -1.0f0:0.2f0:1.0f0)
        X  = reshape(xs, 1, :)
        y  = Float32.(2.0f0 .* xs .+ 1.0f0)
        eval_ = TreeFitnessEvaluator(X, y, ops)

        sentinel = 7.777f0
        problem = GPProblem(eval_, TreeGenome{Float32}; seed=123)
        alg = GeneticProgramming(;
            pop_size=30, generations=1,
            mutation_rate=0.0, crossover_rate=0.0,  # keep initial pop verbatim
            elitism=30,                             # disable breeding
            parallel=false,
            constant_sampler=(rng) -> sentinel,
        )
        result = solve(problem, alg)

        # Scan the final population for a constant equal to the sentinel.
        found = false
        for g in result.population
            constants, _ = DynamicExpressions.get_scalar_constants(g.tree)
            if any(c -> c == sentinel, constants)
                found = true
                break
            end
        end
        @test found
    end

    @testset "erc_uniform range respected end-to-end" begin
        # After a short solve with erc_uniform(100, 200), every literal
        # in the initial population lies in that range.
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        xs = collect(Float32, 0.0f0:0.1f0:1.0f0)
        X  = reshape(xs, 1, :)
        y  = xs
        eval_ = TreeFitnessEvaluator(X, y, ops)

        problem = GPProblem(eval_, TreeGenome{Float32}; seed=7)
        alg = GeneticProgramming(;
            pop_size=20, generations=1,
            mutation_rate=0.0, crossover_rate=0.0,
            elitism=20,
            parallel=false,
            constant_sampler=erc_uniform(100.0f0, 200.0f0),
        )
        result = solve(problem, alg)

        # Every numeric literal across the whole population must be in range.
        any_constant = false
        for g in result.population
            cs, _ = DynamicExpressions.get_scalar_constants(g.tree)
            for c in cs
                any_constant = true
                @test 100.0f0 <= c <= 200.0f0
            end
        end
        @test any_constant  # smoke: at least one literal appeared in the pop
    end

    @testset "default (sampler=nothing) preserves randn behavior" begin
        # With sampler=nothing, constants should be in a roughly ±5 window
        # (standard normal tail beyond that is unlikely for small populations).
        # We don't assert distribution shape — only that it's NOT in the
        # erc_uniform sentinel range above.
        ops = OperatorEnum(binary_operators=[+, *], unary_operators=[])
        xs = collect(Float32, 0.0f0:0.1f0:1.0f0)
        X  = reshape(xs, 1, :)
        y  = xs
        eval_ = TreeFitnessEvaluator(X, y, ops)

        problem = GPProblem(eval_, TreeGenome{Float32}; seed=11)
        alg = GeneticProgramming(;
            pop_size=50, generations=1,
            mutation_rate=0.0, crossover_rate=0.0,
            elitism=50,
            parallel=false,
            # constant_sampler left default (nothing)
        )
        result = solve(problem, alg)

        in_sentinel_range = 0
        total = 0
        for g in result.population
            cs, _ = DynamicExpressions.get_scalar_constants(g.tree)
            for c in cs
                total += 1
                if 100.0f0 <= c <= 200.0f0
                    in_sentinel_range += 1
                end
            end
        end
        # Standard-normal draws are essentially never in [100, 200].
        if total > 0
            @test in_sentinel_range == 0
        end
    end
end
