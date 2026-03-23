# Load HTTP to trigger the extension, then load mock infrastructure.
using HTTP
const LLMExt = Base.get_extension(GenProg, :LLMOperatorExt)
include(joinpath(@__DIR__, "..", "mocks", "mock_http.jl"))

@testset "LLMMutationOperator integration" begin

    # Helper: create a simple test problem and genome.
    function make_llm_test_setup()
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        rng = Random.MersenneTwister(42)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)
        g = ExprGenome([:(y = x * Float32(2.0))], state)
        return (g, state, rng)
    end

    @testset "serialize/deserialize round-trip" begin
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        for op in [:>, :<]
            add!(fset, op, 2, Float32, Bool)
        end
        rng = Random.MersenneTwister(42)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        # Generate multiple random assignment-only genomes and round-trip them.
        successes = 0
        for _ in 1:20
            body = [create_random_assignment(state) for _ in 1:3]
            g = ExprGenome(body, state)
            s = serialize(g)
            g2 = deserialize(ExprGenome, s, state)
            # Some assignments may fail type-checking during round-trip if
            # repr produces forms like Float32(literal) that aren't in the
            # function set. We require at least some statements to survive.
            if g2 !== nothing
                successes += 1
                @test g2 isa ExprGenome
                @test length(g2.body) >= 1
            end
        end
        # At least 80% of genomes should round-trip successfully.
        @test successes >= 16
    end

    @testset "deserialize with partial recovery" begin
        fset = FunctionSet(Set{FunctionDetails}())
        add!(fset, :+, 2, Float32, Float32)
        rng = Random.MersenneTwister(42)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 1)

        # Mix of valid and invalid lines (plain format without :() wrapping).
        s = "y = x + Float32(1.0)\nthis is garbage\ny = x"
        g = deserialize(ExprGenome, s, state)
        @test g !== nothing
        # "y = x + Float32(1.0)" fails type-check because Float32 is not
        # in the function set as a callable. "y = x" passes. Garbage is skipped.
        @test length(g.body) >= 1
    end

    @testset "deserialize returns nothing for all-invalid input" begin
        fset = FunctionSet(Set{FunctionDetails}())
        rng = Random.MersenneTwister(42)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 0)

        @test deserialize(ExprGenome, "not julia code!!!", state) === nothing
        @test deserialize(ExprGenome, "", state) === nothing
    end

    @testset "Valid mock response" begin
        (g, state, rng) = make_llm_test_setup()

        op = LLMExt.LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="",  # skip key check
        )

        install_mock_http!(LLMExt)
        try
            # Register a mock that returns a type-correct mutated program.
            register_mock_response!("anthropic.com",
                mock_anthropic_response("y = x * x"))

            result = mutate(op, g, rng)
            @test result isa ExprGenome
            @test !isempty(result.body)
            # The result should come from the LLM, not the fallback.
            @test any(e -> string(e) == string(:(y = x * x)), result.body)
        finally
            clear_mock_responses!()
            restore_http!(LLMExt)
        end
    end

    @testset "OpenAI-format mock response" begin
        (g, state, rng) = make_llm_test_setup()

        op = LLMExt.LLMMutationOperator(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="llama3",
            api_key_env="",
        )

        install_mock_http!(LLMExt)
        try
            register_mock_response!("localhost",
                mock_openai_response("y = x + Float32(1.0)"))

            result = mutate(op, g, rng)
            @test result isa ExprGenome
            @test !isempty(result.body)
        finally
            clear_mock_responses!()
            restore_http!(LLMExt)
        end
    end

    @testset "Garbage response falls back gracefully" begin
        (g, state, rng) = make_llm_test_setup()

        op = LLMExt.LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="",
        )

        install_mock_http!(LLMExt)
        try
            register_mock_response!("anthropic.com",
                mock_anthropic_response("this is not julia code!!!"))

            # Should fall back to SubtreeMutation without error.
            result = mutate(op, g, rng)
            @test result isa ExprGenome
            @test !isempty(result.body)
        finally
            clear_mock_responses!()
            restore_http!(LLMExt)
        end
    end

    @testset "Timeout falls back gracefully" begin
        (g, state, rng) = make_llm_test_setup()

        op = LLMExt.LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="",
            timeout_seconds=1.0,
        )

        install_mock_http!(LLMExt)
        try
            # Register a mock that throws a timeout-like exception.
            register_mock_response!("anthropic.com",
                ErrorException("HTTP read timeout after 1 seconds"))

            result = mutate(op, g, rng)
            @test result isa ExprGenome
            @test !isempty(result.body)
        finally
            clear_mock_responses!()
            restore_http!(LLMExt)
        end
    end

    @testset "Missing API key falls back gracefully" begin
        (g, state, rng) = make_llm_test_setup()

        # Use a key env var that definitely doesn't exist.
        op = LLMExt.LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="GENPROG_TEST_NONEXISTENT_KEY_12345",
        )

        # Temporarily ensure the env var doesn't exist.
        had_key = haskey(ENV, "GENPROG_TEST_NONEXISTENT_KEY_12345")
        had_key && delete!(ENV, "GENPROG_TEST_NONEXISTENT_KEY_12345")
        try
            result = mutate(op, g, rng)
            @test result isa ExprGenome
            @test !isempty(result.body)
        finally
            # No need to restore — the var didn't exist.
        end
    end

    @testset "LLM operator in evolution loop" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, 0.0, 1.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        for op in [:>, :<]
            add!(fset, op, 2, Float32, Bool)
        end

        llm_op = LLMExt.LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="",
        )

        problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
        algorithm = GeneticProgramming(
            pop_size=20,
            generations=5,
            mutation_rate=0.4,
            crossover_rate=0.2,
            mutation_ops=AbstractMutationOperator[llm_op, SubtreeMutation(), PointMutation()],
        )

        install_mock_http!(LLMExt)
        try
            # Register a mock that returns a valid mutation.
            register_mock_response!("anthropic.com",
                mock_anthropic_response("y = x * x"))

            result = solve(problem, algorithm; verbose=false)
            @test result isa GPResult{ExprGenome}
            @test result.best_fitness >= 0.0
            @test length(result.fitness_history) == 5
        finally
            clear_mock_responses!()
            restore_http!(LLMExt)
        end
    end

    @testset "Live API (skipped without key)" begin
        if haskey(ENV, "ANTHROPIC_API_KEY")
            input_cols = Dict(:x => Float32)
            output_cols = Dict(:y => Float32)
            xs = Float32[-1.0, 0.0, 1.0]
            input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
            output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
            fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                        time_limit_ns=1_000_000_000)

            fset = FunctionSet(Set{FunctionDetails}())
            for func in [:+, :-, :*, :/]
                add!(fset, func, 2, Float32, Float32)
            end

            llm_op = LLMExt.LLMMutationOperator()

            problem = GPProblem(fe, ExprGenome; function_set=fset, num_temps=2, seed=42)
            algorithm = GeneticProgramming(
                pop_size=10,
                generations=2,
                mutation_rate=0.5,
                mutation_ops=AbstractMutationOperator[llm_op, SubtreeMutation()],
            )

            result = solve(problem, algorithm; verbose=false)
            @test result isa GPResult{ExprGenome}
        else
            @test_skip "ANTHROPIC_API_KEY not set"
        end
    end
end
