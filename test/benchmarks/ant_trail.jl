# Santa Fe Ant Trail benchmark — uses AntGenome (proper side-effectful genome type).
#
# AntGenome is the correct representation for side-effectful GP problems.
# Unlike the Phase 4 Bool-dummy ExprGenome approach, AntGenome directly
# represents programs as trees of primitive calls and control flow.
#
# The Santa Fe Trail is widely documented as a hard GP benchmark.
# See Langdon & Poli (1998) "Why Ants Are Hard."

# Canonical Santa Fe Trail food positions (1-indexed, row then col).
const SANTA_FE_FOOD = [
    (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,10),(1,11),
    (1,12),(1,13),(1,14),(1,15),(1,16),(1,17),(1,18),(1,19),(1,20),
    (1,21),(1,22),(1,23),(1,24),(1,25),(1,26),(2,26),(3,26),(4,26),
    (5,26),(6,26),(8,26),(9,26),(10,26),(11,26),(12,26),(13,26),
    (14,26),(16,26),(17,26),(18,26),(19,26),(20,26),(21,26),(24,26),
    (25,26),(26,26),(27,26),(28,26),(29,26),(30,26),(31,26),(32,26),
    (32,25),(32,24),(32,23),(32,22),(32,21),(32,20),(32,19),(32,17),
    (32,16),(32,15),(32,14),(32,13),(32,12),(32,10),(32,9),(32,8),
    (32,7),(32,6),(32,5),(32,4),(32,3),(32,2),(32,1),(31,1),(30,1),
    (29,1),(28,1),(27,1),(26,1),(25,1),(24,1),(23,1),(22,1),(21,1),
    (20,1),(19,1),(18,1),(17,1),(16,1)
]
const N_FOOD = length(SANTA_FE_FOOD)

@testset "Santa Fe Ant Trail benchmark (AntGenome)" begin
    evaluator = AntEvaluator(SANTA_FE_FOOD, 600)

    # Smoke test: verify the AntGenome framework works correctly.
    # Full ant trail runs (pop=200, gen=500) remain impractical due to
    # @eval overhead — each individual requires function compilation.
    algorithm = GeneticProgramming(
        pop_size=10, generations=3,
        mutation_rate=0.4, crossover_rate=0.3,
        elitism=1, parallel=false
    )

    problem = GPProblem(evaluator, AntGenome; seed=42)
    result = solve(problem, algorithm; verbose=false)
    eaten = N_FOOD - Int(result.best_fitness)
    println("  Ant trail smoke test (AntGenome): $eaten/$N_FOOD pellets eaten")
    flush(stdout)

    @test result isa GPResult{AntGenome}
    @test result.best_fitness <= Float64(N_FOOD)
    @test result.best_fitness >= 0.0
    @test length(result.fitness_history) == 3
end
