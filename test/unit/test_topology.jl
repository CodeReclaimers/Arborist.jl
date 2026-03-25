@testset "Topology" begin
    rng = Random.MersenneTwister(42)

    @testset "RingTopology" begin
        t = RingTopology()
        @test migration_targets(t, 1, 4, rng) == [2]
        @test migration_targets(t, 2, 4, rng) == [3]
        @test migration_targets(t, 3, 4, rng) == [4]
        @test migration_targets(t, 4, 4, rng) == [1]  # wrap around
        # Edge case: 2 islands
        @test migration_targets(t, 1, 2, rng) == [2]
        @test migration_targets(t, 2, 2, rng) == [1]
    end

    @testset "CompleteTopology" begin
        t = CompleteTopology()
        @test sort(migration_targets(t, 1, 4, rng)) == [2, 3, 4]
        @test sort(migration_targets(t, 3, 4, rng)) == [1, 2, 4]
        @test sort(migration_targets(t, 2, 3, rng)) == [1, 3]
    end

    @testset "RandomTopology" begin
        t = RandomTopology(2)
        targets = migration_targets(t, 1, 5, rng)
        @test length(targets) == 2
        @test all(x -> x != 1, targets)
        @test all(x -> 1 <= x <= 5, targets)
        @test length(unique(targets)) == 2  # no duplicates

        # n_targets > n_islands - 1 should clamp
        t2 = RandomTopology(10)
        targets2 = migration_targets(t2, 1, 3, rng)
        @test length(targets2) == 2  # only 2 others available
        @test sort(targets2) == [2, 3]
    end

    @testset "IslandModel construction with topology" begin
        alg = GeneticProgramming(pop_size=10)
        # Default topology
        im = IslandModel(island_algorithm=alg)
        @test im.topology isa RingTopology
        @test im.distributed == false
        @test im.async == false

        # Custom topology
        im2 = IslandModel(island_algorithm=alg, topology=CompleteTopology())
        @test im2.topology isa CompleteTopology

        # async requires distributed
        @test_throws ArgumentError IslandModel(island_algorithm=alg, async=true)
        @test_throws ArgumentError IslandModel(island_algorithm=alg, async=true, distributed=false)

        # async with distributed is fine
        im3 = IslandModel(island_algorithm=alg, async=true, distributed=true)
        @test im3.async == true
        @test im3.distributed == true
    end
end
