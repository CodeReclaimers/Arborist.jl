# LLMMutationOperator is now in core (no extension needed).
include(joinpath(@__DIR__, "..", "mocks", "mock_http.jl"))

@testset "LLMMutationOperator integration" begin

    @testset "JSON escape handles control characters" begin
        @test Arborist._json_escape("hello\nworld") == "hello\\nworld"
        @test Arborist._json_escape("tab\there") == "tab\\there"
        @test Arborist._json_escape("quote\"mark") == "quote\\\"mark"
        @test Arborist._json_escape("back\\slash") == "back\\\\slash"
        @test Arborist._json_escape("bs\b") == "bs\\b"
        @test Arborist._json_escape("ff\f") == "ff\\f"
        @test Arborist._json_escape("cr\r") == "cr\\r"
        println("  _json_escape: all control characters escaped correctly")
        flush(stdout)
    end

    @testset "JSON unescape order is correct" begin
        # The key test: a literal backslash followed by 'n' should NOT become a newline.
        # In JSON this is encoded as "\\n" (escaped backslash + n).
        # After regex extraction, the raw captured string contains: \\n
        # This should unescape to: \n (literal backslash + n), NOT a newline.
        json = """{"text": "line1\\\\nline2"}"""
        result = Arborist._find_last_json_string(json, "text")
        @test result == "line1\\nline2"  # literal \n, not newline
        @test !contains(result, "\n")     # should NOT contain actual newline

        # A real newline is encoded as \n in JSON:
        json2 = """{"text": "line1\\nline2"}"""
        result2 = Arborist._find_last_json_string(json2, "text")
        @test result2 == "line1\nline2"  # actual newline

        # Additional escapes
        json3 = """{"text": "a\\tb\\bc\\/d"}"""
        result3 = Arborist._find_last_json_string(json3, "text")
        @test result3 == "a\tb\bc/d"

        println("  JSON unescape order: backslash-n vs newline correctly distinguished")
        flush(stdout)
    end

    @testset "JSON unescape walker preserves NUL-byte sentinel sequences" begin
        # The previous unescape implementation used "\x00BACKSLASH\x00" as a
        # round-trip placeholder while decoding \\. If a captured JSON string
        # ever contained that exact byte sequence (highly unlikely in source
        # code but possible in arbitrary LLM output), the placeholder would
        # be mangled on the way out. The single-pass walker has no such
        # placeholder, so the sequence round-trips byte-for-byte.
        sentinel = "\x00BACKSLASH\x00"
        @test Arborist._json_unescape(sentinel) == sentinel
        # Same sentinel embedded next to a real escape; only the \n should decode.
        @test Arborist._json_unescape("\x00BACKSLASH\x00\\n") == "\x00BACKSLASH\x00\n"
        # Sentinel surviving \\ decoding (which used to overwrite it).
        @test Arborist._json_unescape("\\\\$sentinel") == "\\$sentinel"
    end

    @testset "JSON unescape walker handles malformed escapes gracefully" begin
        # Trailing lone backslash — emit literally rather than dropping or crashing.
        @test Arborist._json_unescape("abc\\") == "abc\\"
        # Unknown escape — pass through unchanged.
        @test Arborist._json_unescape("a\\zb") == "a\\zb"
        # Malformed \u (fewer than 4 hex digits) — pass through unchanged.
        @test Arborist._json_unescape("a\\u12") == "a\\u12"
        @test Arborist._json_unescape("a\\uZZZZ") == "a\\uZZZZ"
        # Well-formed unicode still decodes.
        @test Arborist._json_unescape("a\\u003cb") == "a<b"
    end


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

    @testset "deserialize accepts control flow" begin
        fset = FunctionSet(Set{FunctionDetails}())
        for func in [:+, :-, :*, :/]
            add!(fset, func, 2, Float32, Float32)
        end
        for op in [:>, :<]
            add!(fset, op, 2, Float32, Bool)
        end
        rng = Random.MersenneTwister(42)
        # Use 2 temps so t1, t2 are available as Float32 variables
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        # while loop with Bool condition (using variables, not Float32() calls)
        s_while = """
        while x > y
            y = x - y
        end
        """
        g = deserialize(ExprGenome, s_while, state)
        @test g !== nothing
        @test any(e -> e isa Expr && e.head == :while, g.body)
        println("  deserialize while: $(length(g.body)) statements")

        # if-else
        s_if = """
        if x > y
            y = x + y
        else
            y = x - y
        end
        """
        g2 = deserialize(ExprGenome, s_if, state)
        @test g2 !== nothing
        @test any(e -> e isa Expr && e.head == :if, g2.body)
        println("  deserialize if-else: $(length(g2.body)) statements")

        # for loop
        s_for = """
        for i = 1:10
            y = x + y
        end
        """
        g3 = deserialize(ExprGenome, s_for, state)
        @test g3 !== nothing
        @test any(e -> e isa Expr && e.head == :for, g3.body)
        println("  deserialize for: $(length(g3.body)) statements")

        # Mixed: assignments + control flow
        s_mixed = """
        y = x * x
        while x > y
            y = y + x
        end
        """
        g4 = deserialize(ExprGenome, s_mixed, state)
        @test g4 !== nothing
        @test length(g4.body) == 2
        println("  deserialize mixed: $(length(g4.body)) statements")

        # Pure assignment still works (backward compat)
        g5 = deserialize(ExprGenome, "y = x + x", state)
        @test g5 !== nothing
        @test length(g5.body) >= 1

        # Full serialize → deserialize round-trip with control flow
        g_rt_in = ExprGenome([
            :(y = x * x),
            Expr(:while, :(x > y), Expr(:block, :(y = x + y)))
        ], state)
        s_rt = serialize(g_rt_in)
        g_rt_out = deserialize(ExprGenome, s_rt, state)
        @test g_rt_out !== nothing
        @test length(g_rt_out.body) == 2
        @test any(e -> e isa Expr && e.head == :(=), g_rt_out.body)
        @test any(e -> e isa Expr && e.head == :while, g_rt_out.body)
        println("  serialize round-trip with control flow: $(length(g_rt_out.body)) statements")
    end

    @testset "Valid mock response" begin
        (g, state, rng) = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="",  # skip key check
        )

        install_mock_http!()
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
            restore_http!()
        end
    end

    @testset "OpenAI-format mock response" begin
        (g, state, rng) = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="llama3",
            api_key_env="",
        )

        install_mock_http!()
        try
            register_mock_response!("localhost",
                mock_openai_response("y = x + Float32(1.0)"))

            result = mutate(op, g, rng)
            @test result isa ExprGenome
            @test !isempty(result.body)
        finally
            clear_mock_responses!()
            restore_http!()
        end
    end

    @testset "Garbage response falls back gracefully" begin
        (g, state, rng) = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="",
        )

        install_mock_http!()
        try
            register_mock_response!("anthropic.com",
                mock_anthropic_response("this is not julia code!!!"))

            # Should fall back to SubtreeMutation without error.
            result = mutate(op, g, rng)
            @test result isa ExprGenome
            @test !isempty(result.body)
        finally
            clear_mock_responses!()
            restore_http!()
        end
    end

    @testset "Timeout falls back gracefully" begin
        (g, state, rng) = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            api_key_env="",
            timeout_seconds=1.0,
        )

        install_mock_http!()
        try
            # Register a mock that throws a timeout-like exception.
            register_mock_response!("anthropic.com",
                ErrorException("HTTP read timeout after 1 seconds"))

            result = mutate(op, g, rng)
            @test result isa ExprGenome
            @test !isempty(result.body)
        finally
            clear_mock_responses!()
            restore_http!()
        end
    end

    @testset "Missing API key falls back gracefully" begin
        (g, state, rng) = make_llm_test_setup()

        # Use a key env var that definitely doesn't exist.
        op = LLMMutationOperator(
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

        llm_op = LLMMutationOperator(
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

        install_mock_http!()
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
            restore_http!()
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

            llm_op = LLMMutationOperator()

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

    # =========================================================================
    # Prompt enrichment integration tests
    # =========================================================================

    @testset "Prompt enrichment appears in request body" begin
        genome, state, rng = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="test",
            api_key_env="",
            sections=[FitnessSection(), GenerationSection()],
        )

        # Manually set context (normally done by solve loop).
        op.context = MutationContext(
            50, 100,
            [0.5, 0.8, 1.0],
            ["elite_prog_1", "elite_prog_2"],
            0.8, 2
        )

        clear_mock_responses!()
        register_mock_response!("localhost", mock_openai_response("y = x * x"))
        install_mock_http!()

        try
            result = mutate(op, genome, rng)
            body = LAST_REQUEST_BODY[]

            # Enrichment sections should appear in the request body.
            @test occursin("Fitness Context", body)
            @test occursin("Generation Progress", body)
            @test occursin("50/100", body)
            @test occursin("Program to mutate", body)

            # The serialized genome should also be present.
            source = serialize(genome)
            @test occursin(Arborist._json_escape(source), body)

            println("  Enrichment present in request body: fitness + generation + genome")
        finally
            restore_http!()
        end
    end

    @testset "No enrichment when sections empty" begin
        genome, state, rng = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="test",
            api_key_env="",
        )

        clear_mock_responses!()
        register_mock_response!("localhost", mock_openai_response("y = x * x"))
        install_mock_http!()

        try
            result = mutate(op, genome, rng)
            body = LAST_REQUEST_BODY[]

            # No enrichment prefix.
            @test !occursin("Fitness Context", body)
            @test !occursin("Program to mutate", body)

            # Genome source should be present directly.
            source = serialize(genome)
            @test occursin(Arborist._json_escape(source), body)

            println("  No enrichment: body contains only genome source")
        finally
            restore_http!()
        end
    end

    @testset "ElitesSection in request body" begin
        genome, state, rng = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="test",
            api_key_env="",
            sections=[ElitesSection(2)],
        )

        op.context = MutationContext(
            10, 100,
            [0.1, 0.5, 1.0],
            ["elite_alpha", "elite_beta", "elite_gamma"],
            0.5, 2
        )

        clear_mock_responses!()
        register_mock_response!("localhost", mock_openai_response("y = x * x"))
        install_mock_http!()

        try
            result = mutate(op, genome, rng)
            body = LAST_REQUEST_BODY[]

            @test occursin("Top 2 Programs", body)
            @test occursin("elite_alpha", body)
            @test occursin("elite_beta", body)
            @test !occursin("elite_gamma", body)  # k=2, third excluded

            println("  ElitesSection(2): top 2 elites in body, 3rd excluded")
        finally
            restore_http!()
        end
    end

    # =========================================================================
    # Token extraction and LLMCallStats tests
    # =========================================================================

    @testset "_extract_usage Anthropic format" begin
        resp = mock_anthropic_response("y = x"; input_tokens=150, output_tokens=42)
        in_tok, out_tok = Arborist._extract_usage(resp, true)
        @test in_tok == 150
        @test out_tok == 42
        println("  _extract_usage Anthropic: in=150, out=42")
    end

    @testset "_extract_usage OpenAI format" begin
        resp = mock_openai_response("y = x"; prompt_tokens=200, completion_tokens=60)
        in_tok, out_tok = Arborist._extract_usage(resp, false)
        @test in_tok == 200
        @test out_tok == 60
        println("  _extract_usage OpenAI: in=200, out=60")
    end

    @testset "_extract_usage missing fields" begin
        in_tok, out_tok = Arborist._extract_usage("{\"id\":\"test\"}", true)
        @test in_tok == 0
        @test out_tok == 0
        println("  _extract_usage missing: in=0, out=0")
    end

    @testset "LLMCallStats accumulation on success" begin
        genome, state, rng = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="test",
            api_key_env="",
        )

        clear_mock_responses!()
        register_mock_response!("localhost",
            mock_openai_response("y = x * x";
                                 prompt_tokens=120, completion_tokens=30))
        install_mock_http!()

        try
            result = mutate(op, genome, rng)
            s = op.stats
            @test s.total_calls == 1
            @test s.llm_successes == 1
            @test s.llm_failures == 0
            @test s.fallback_skips == 0
            @test s.input_tokens == 120
            @test s.output_tokens == 30
            @test s.input_chars > 0
            @test s.output_chars > 0
            @test s.total_latency > 0.0
            println("  Stats on success: calls=1, successes=1, in_tok=120, out_tok=30, " *
                    "in_chars=$(s.input_chars), out_chars=$(s.output_chars)")
        finally
            restore_http!()
        end
    end

    @testset "LLMCallStats accumulation on parse failure" begin
        genome, state, rng = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="test",
            api_key_env="",
        )

        clear_mock_responses!()
        register_mock_response!("localhost",
            mock_openai_response("this is not valid julia code at all!!!"))
        install_mock_http!()

        try
            result = mutate(op, genome, rng)
            s = op.stats
            @test s.total_calls == 1
            @test s.llm_successes == 0
            @test s.llm_failures == 1
            @test s.fallback_skips == 0
            @test s.total_latency >= 0.0  # mock is instant; real calls take seconds
            println("  Stats on parse failure: calls=1, failures=1")
        finally
            restore_http!()
        end
    end

    @testset "LLMCallStats accumulation on API key skip" begin
        genome, state, rng = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="https://api.anthropic.com/v1/messages",
            model="test",
            api_key_env="GENPROG_TEST_NONEXISTENT_KEY_STATS",
        )

        result = mutate(op, genome, rng)
        s = op.stats
        @test s.total_calls == 1
        @test s.llm_successes == 0
        @test s.llm_failures == 0
        @test s.fallback_skips == 1
        println("  Stats on API key skip: calls=1, skips=1")
    end

    @testset "LLMCallStats accumulates across multiple calls" begin
        genome, state, rng = make_llm_test_setup()

        op = LLMMutationOperator(
            endpoint="http://localhost:11434/v1/chat/completions",
            model="test",
            api_key_env="",
        )

        clear_mock_responses!()
        register_mock_response!("localhost",
            mock_openai_response("y = x * x";
                                 prompt_tokens=100, completion_tokens=25))
        install_mock_http!()

        try
            mutate(op, genome, rng)
            mutate(op, genome, rng)
            mutate(op, genome, rng)
            s = op.stats
            @test s.total_calls == 3
            @test s.llm_successes == 3
            @test s.input_tokens == 300   # 3 × 100
            @test s.output_tokens == 75   # 3 × 25
            println("  Stats accumulate: 3 calls, in_tok=300, out_tok=75")
        finally
            restore_http!()
        end
    end
end
