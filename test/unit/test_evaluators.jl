@testset "Evaluators" begin

    @testset "TableFitnessEvaluator basic evaluation" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[1.0, 2.0, 3.0, 4.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v) for v in xs]
        # Use generous time limit to avoid JIT warmup failures on CI.
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        # A perfect function.
        perfect(x::Float32) = x
        @test evaluate(fe, perfect) == 0.0

        # A function that returns constant 0 — MSE should be mean of x^2.
        zero_func(x::Float32) = Float32(0.0)
        expected_mse = sum(xs .^ 2) / length(xs)
        @test evaluate(fe, zero_func) ≈ expected_mse

        # A function that always throws.
        bad_func(x::Float32) = error("boom")
        @test evaluate(fe, bad_func) == Inf
    end

    @testset "TableFitnessEvaluator handles partial failures" begin
        input_cols = Dict(:x => Float32)
        output_cols = Dict(:y => Float32)
        xs = Float32[-1.0, -2.0, 1.0, 2.0]
        input_rows = [Dict{Symbol,Any}(:x => v) for v in xs]
        output_rows = [Dict{Symbol,Any}(:y => v^2) for v in xs]
        fe = TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                    time_limit_ns=1_000_000_000)

        function half_fail(x::Float32)
            x < 0 && error("negative")
            return x^2
        end
        result = evaluate(fe, half_fail)
        @test result == 0.0  # Successful rows have perfect fitness.
    end

    @testset "input_signature and output_signature" begin
        input_cols = Dict(:x => Float32, :y => Int32)
        output_cols = Dict(:z => Float32)
        fe = TableFitnessEvaluator(input_cols, output_cols,
            [Dict{Symbol,Any}(:x => 1.0f0, :y => Int32(1))],
            [Dict{Symbol,Any}(:z => 1.0f0)])
        @test input_signature(fe) == input_cols
        @test output_signature(fe) == output_cols
    end

end
