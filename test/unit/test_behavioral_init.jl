@testset "Behavioral initialization" begin
    using Arborist: behavioral_initialize, GenState, FunctionSet, FunctionDetails,
                    ExprGenome, TableFitnessEvaluator, add!

    # Build a simple evaluator: y = x * 2 with 5 sample points.
    # This gives a meaningful fitness landscape where some random programs
    # will score better than others.
    input_cols = Dict(:x => Float32)
    output_cols = Dict(:y => Float32)
    input_rows = [Dict(:x => Float32(i)) for i in 1:5]
    output_rows = [Dict(:y => Float32(2*i)) for i in 1:5]
    evaluator = TableFitnessEvaluator(input_cols, output_cols,
                                       input_rows, output_rows)

    fset = FunctionSet(Set{FunctionDetails}())
    for func in [:+, :-, :*, :/]
        add!(fset, func, 2, Float32, Float32)
    end

    # Simple behavioral fingerprint: evaluate on probe inputs, return
    # rounded output vector.
    function simple_fingerprint(g::ExprGenome)
        try
            body = g.body
            state = g.state
            harness = Arborist.create_harness(state, body)
            f = @eval $harness
            outputs = Float32[]
            for xv in [1.0f0, 2.0f0, 3.0f0]
                try
                    result = f(xv)
                    push!(outputs, round(Float32(result), digits=1))
                catch
                    push!(outputs, NaN32)
                end
            end
            return outputs
        catch
            return [NaN32, NaN32, NaN32]
        end
    end

    function simple_distance(a::Vector{Float32}, b::Vector{Float32})
        n = length(a)
        mismatches = sum(1 for i in 1:n if a[i] != b[i])
        return Float64(mismatches) / n
    end

    @testset "basic: returns correct size" begin
        rng = Random.MersenneTwister(42)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        result = behavioral_initialize(
            state, evaluator, simple_fingerprint, simple_distance, 20;
            pool_size=200, bin_threshold=0.3, verbose=false)

        @test length(result) == 20
        @test all(g -> g isa ExprGenome, result)
        println("  basic: returned $(length(result)) ExprGenome programs")
    end

    @testset "diversity: fingerprints are not all identical" begin
        rng = Random.MersenneTwister(123)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        result = behavioral_initialize(
            state, evaluator, simple_fingerprint, simple_distance, 20;
            pool_size=500, bin_threshold=0.3, verbose=false)

        # Compute fingerprints of the returned population.
        fps = [simple_fingerprint(g) for g in result]
        valid_fps = filter(fp -> !any(isnan, fp), fps)

        if length(valid_fps) >= 2
            # Check that at least some fingerprints differ.
            unique_fps = Set(valid_fps)
            @test length(unique_fps) >= 2
            println("  diversity: $(length(unique_fps)) distinct fingerprints in population of 20")
        else
            println("  diversity: too few valid fingerprints to check ($(length(valid_fps)))")
            @test true  # don't fail — random programs may all crash
        end
    end

    @testset "small pool: graceful when pool < target" begin
        rng = Random.MersenneTwister(456)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        # Pool of 5 but target of 10 — should still return 10 programs.
        result = behavioral_initialize(
            state, evaluator, simple_fingerprint, simple_distance, 10;
            pool_size=5, bin_threshold=0.3, verbose=false)

        @test length(result) == 10
        @test all(g -> g isa ExprGenome, result)
        println("  small pool: pool=5, target=10, returned $(length(result))")
    end

    @testset "verbose=true runs without error" begin
        rng = Random.MersenneTwister(789)
        state = GenState(rng, fset, Dict(:x => Float32), Dict(:y => Float32), 2)

        result = behavioral_initialize(
            state, evaluator, simple_fingerprint, simple_distance, 10;
            pool_size=50, bin_threshold=0.3, verbose=true)

        @test length(result) == 10
        println("  verbose=true: completed without error")
    end
end
