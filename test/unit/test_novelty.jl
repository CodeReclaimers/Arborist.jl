using Test
using Arborist
using Random

@testset "Novelty Search" begin
    @testset "NoveltyArchive basics" begin
        archive = NoveltyArchive(Vector{Float64}; max_size=10, add_threshold=0.0)
        @test isempty(archive)
        @test length(archive) == 0

        @test_throws ArgumentError NoveltyArchive(Vector{Float64}; max_size=0)
        @test_throws ArgumentError NoveltyArchive(Vector{Float64}; add_threshold=-0.1)
    end

    @testset "novelty score: empty archive" begin
        archive = NoveltyArchive(Vector{Float64}; max_size=10)
        # Use a dummy genome that wraps a Float64 vector fingerprint via a
        # minimal AbstractGenome. Simpler: drive the score function directly.
        fp = [0.5, 0.5]
        # Empty archive → novelty_score returns 0.0.
        score = Arborist._novelty_score(fp, archive, 3,
                                         (a, b) -> sqrt(sum((a .- b).^2)))
        @test score == 0.0
    end

    @testset "novelty score: k-NN of stored points" begin
        archive = NoveltyArchive(Vector{Float64}; max_size=10)
        push!(archive.entries, [0.0, 0.0])
        push!(archive.entries, [1.0, 0.0])
        push!(archive.entries, [0.0, 1.0])
        push!(archive.entries, [5.0, 5.0])

        dist = (a, b) -> sqrt(sum((a .- b).^2))

        # Query at origin: 3 nearest are {origin (dist 0), (1,0), (0,1)}
        # with dists 0, 1, 1 → mean 2/3.
        s = Arborist._novelty_score([0.0, 0.0], archive, 3, dist)
        @test s ≈ 2.0 / 3.0

        # Query at (10, 10): distances to {(0,0), (1,0), (0,1), (5,5)} are
        # {sqrt(200), sqrt(181), sqrt(181), sqrt(50)}. Top 3 nearest:
        # sqrt(50), sqrt(181), sqrt(181).
        s2 = Arborist._novelty_score([10.0, 10.0], archive, 3, dist)
        @test s2 ≈ (sqrt(50.0) + 2 * sqrt(181.0)) / 3.0
    end

    @testset "archive respects max_size" begin
        archive = NoveltyArchive(Vector{Float64}; max_size=3)
        Arborist._maybe_add_to_archive!(archive, [0.0], 0.1)
        Arborist._maybe_add_to_archive!(archive, [1.0], 0.1)
        Arborist._maybe_add_to_archive!(archive, [2.0], 0.1)
        @test length(archive) == 3
        # Full → drop further adds silently.
        added = Arborist._maybe_add_to_archive!(archive, [3.0], 0.1)
        @test added == false
        @test length(archive) == 3
    end

    @testset "archive respects add_threshold" begin
        archive = NoveltyArchive(Vector{Float64}; max_size=10, add_threshold=0.5)
        # Score below threshold: rejected.
        @test Arborist._maybe_add_to_archive!(archive, [0.0], 0.4) == false
        @test length(archive) == 0
        # Score at threshold: accepted.
        @test Arborist._maybe_add_to_archive!(archive, [1.0], 0.5) == true
        @test length(archive) == 1
    end

    @testset "NoveltySearchEvaluator end-to-end on TreeGenome" begin
        using DynamicExpressions
        # Use tree complexity (depth, count_nodes) as a behavioral fingerprint.
        # NoveltySearch will prefer structurally diverse trees — this doesn't
        # solve an objective, just exercises the plumbing end-to-end.
        ops = OperatorEnum(binary_operators=[+, -, *], unary_operators=[sin])
        X = reshape(collect(Float32, 0.0:0.1:1.0), 1, :)
        y = X[1, :] .* X[1, :]
        # Base evaluator is just for the solve path's input/output signatures.
        # The novelty evaluator ignores it — it uses only the fingerprint.
        archive = NoveltyArchive(Vector{Float64}; max_size=50, add_threshold=0.0)
        fp_fn = g -> [Float64(count_nodes(g.tree)), Float64(count_depth(g.tree))]
        dist_fn = (a, b) -> sqrt(sum((a .- b).^2))
        nov_eval = NoveltySearchEvaluator(fp_fn, dist_fn, archive; k=5)

        # Need the solve path to init a TreeGenome population. The init path
        # reads input/output signatures from the evaluator. NoveltySearch's
        # empty signatures break that for TreeGenome. Workaround: wrap a
        # real TreeFitnessEvaluator for init; then use novelty for eval.
        # Simpler: build a few TreeGenomes directly and call evaluate_genome.
        rng = MersenneTwister(42)
        g1 = TreeGenome{Float32}(Node{Float32}(; val=Float32(1.0)), ops, 1)
        g2 = TreeGenome{Float32}(
            Node{Float32}(; op=UInt8(3),
                           l=Node{Float32}(; feature=UInt16(1)),
                           r=Node{Float32}(; val=Float32(2.0))),
            ops, 1)
        s1 = Arborist.evaluate_genome(g1, nov_eval)
        s2 = Arborist.evaluate_genome(g2, nov_eval)
        @test isfinite(s1)
        @test isfinite(s2)
        # Both fingerprints should have made it into the archive (empty
        # add_threshold).
        @test length(archive) >= 1
    end

    @testset "fingerprint_fn failure returns Inf" begin
        fp_fn = g -> error("boom")
        archive = NoveltyArchive(Int; max_size=5)
        nov_eval = NoveltySearchEvaluator(fp_fn, (a, b) -> 0.0, archive; k=1)
        # Use a tiny ad-hoc genome — an AbstractGenome subtype we'll construct
        # inline isn't needed; use any existing concrete AbstractGenome.
        # Build a minimal TreeGenome.
        using DynamicExpressions
        ops = OperatorEnum(binary_operators=[+], unary_operators=[])
        g = TreeGenome{Float32}(Node{Float32}(; val=Float32(0.0)), ops, 0)
        @test Arborist.evaluate_genome(g, nov_eval) == Inf
    end
end
