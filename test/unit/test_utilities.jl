using Test
using Arborist
using Random

@testset "Utilities" begin
    @testset "train_test_split: basic shape" begin
        X = reshape(Float64.(1:40), 4, 10)
        y = collect(1:10)
        rng = MersenneTwister(0)
        Xtr, ytr, Xte, yte = train_test_split(X, y; test_size=0.3, rng=rng)
        @test size(Xtr, 1) == 4 && size(Xte, 1) == 4
        @test size(Xtr, 2) + size(Xte, 2) == 10
        @test length(ytr) + length(yte) == 10
        # Default test_size=0.3 on 10 samples rounds to 3 test, 7 train.
        @test size(Xte, 2) == 3
        @test size(Xtr, 2) == 7
    end

    @testset "train_test_split: columns track labels" begin
        X = reshape(Float64.(1:80), 4, 20)
        y = collect(1:20)
        rng = MersenneTwister(0)
        Xtr, ytr, Xte, yte = train_test_split(X, y; test_size=0.2, rng=rng)
        # Each sample's column is uniquely identified by its first-row value
        # equal to 4 * (col_idx - 1) + 1. Check X[1, j] == 4 * (y[j] - 1) + 1
        # to confirm alignment survives the permutation.
        for j in 1:length(ytr)
            @test Xtr[1, j] == 4 * (ytr[j] - 1) + 1
        end
        for j in 1:length(yte)
            @test Xte[1, j] == 4 * (yte[j] - 1) + 1
        end
    end

    @testset "train_test_split: stratified preserves class proportions" begin
        X = reshape(Float64.(1:120), 4, 30)
        y = collect(1:30)
        classes = vcat(fill(1, 10), fill(2, 10), fill(3, 10))
        rng = MersenneTwister(42)
        Xtr, ytr, Xte, yte = train_test_split(X, y; test_size=0.3, rng=rng,
                                               stratify=classes)
        # Under test_size=0.3, each class (10 rows) yields 3 test / 7 train.
        # Check the class labels corresponding to the y values split that way.
        cls_tr = classes[ytr]
        cls_te = classes[yte]
        for c in 1:3
            @test count(==(c), cls_tr) == 7
            @test count(==(c), cls_te) == 3
        end
    end

    @testset "train_test_split: validation" begin
        X = rand(2, 10); y = collect(1:10)
        @test_throws ArgumentError train_test_split(X, y; test_size=0.0)
        @test_throws ArgumentError train_test_split(X, y; test_size=1.0)
        @test_throws ArgumentError train_test_split(X, y; test_size=-0.1)
        # Mismatched X / y sizes.
        X2 = rand(2, 5)
        @test_throws ArgumentError train_test_split(X2, y; test_size=0.2)
        # Stratify length mismatch.
        @test_throws ArgumentError train_test_split(
            X, y; test_size=0.2, stratify=collect(1:5))
    end

    @testset "train_test_split: determinism under same RNG" begin
        X = rand(3, 50); y = rand(50)
        rng1 = MersenneTwister(7)
        rng2 = MersenneTwister(7)
        a = train_test_split(X, y; test_size=0.4, rng=rng1)
        b = train_test_split(X, y; test_size=0.4, rng=rng2)
        @test a == b
    end

    @testset "summarize: known values" begin
        s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        @test s.n == 5
        @test s.mean ≈ 3.0
        @test s.median ≈ 3.0
        @test s.min == 1.0
        @test s.max == 5.0
        @test s.q25 ≈ 2.0
        @test s.q75 ≈ 4.0
        # Sample std of 1..5: sqrt(10/4) = sqrt(2.5) ≈ 1.5811.
        @test isapprox(s.std, sqrt(2.5); atol=1e-6)
    end

    @testset "summarize: excludes non-finite" begin
        s = summarize([1.0, Inf, 2.0, NaN, 3.0, -Inf])
        @test s.n == 3
        @test s.mean ≈ 2.0
        @test s.min == 1.0
        @test s.max == 3.0
    end

    @testset "summarize: all non-finite returns NaN summary" begin
        s = summarize([Inf, NaN])
        @test s.n == 0
        @test isnan(s.mean)
        @test isnan(s.std)
    end

    @testset "summarize: single value" begin
        s = summarize([42.0])
        @test s.n == 1
        @test s.mean == 42.0
        @test s.std == 0.0
        @test s.median == 42.0
    end

    @testset "run_multi_seed: serial" begin
        results = run_multi_seed([1, 2, 3, 4, 5]) do seed
            MersenneTwister(seed) |> rng -> rand(rng)
        end
        @test length(results) == 5
        # Deterministic w/ same seed sequence.
        results2 = run_multi_seed([1, 2, 3, 4, 5]) do seed
            MersenneTwister(seed) |> rng -> rand(rng)
        end
        @test results == results2
    end

    @testset "run_multi_seed: parallel returns same set" begin
        if Threads.nthreads() >= 2
            par = run_multi_seed([1, 2, 3, 4, 5]; parallel=true) do seed
                MersenneTwister(seed) |> rng -> rand(rng)
            end
            ser = run_multi_seed([1, 2, 3, 4, 5]) do seed
                MersenneTwister(seed) |> rng -> rand(rng)
            end
            # Order matches because each task's RNG is seeded deterministically.
            @test par == ser
        end
    end

    @testset "run_multi_seed: empty seeds" begin
        @test isempty(run_multi_seed(x -> x, Int[]))
    end

    @testset "train_test_split drives Benchmarks.iris stratified split" begin
        # Confirms the refactor: Benchmarks.iris now calls train_test_split.
        ds = Arborist.Benchmarks.iris(; rng=MersenneTwister(20260421),
                                        test_ratio=0.2)
        @test size(ds.X_train) == (4, 120)
        @test size(ds.X_test) == (4, 30)
        for cls in 1:3
            @test count(==(cls), ds.y_train) == 40
            @test count(==(cls), ds.y_test) == 10
        end
    end
end
