using Test
using GenProg
using Random

@testset "GenProg.jl" begin
    # Unit tests
    include("unit/test_evaluators.jl")
    include("unit/test_genome.jl")
    include("unit/test_operators.jl")
    include("unit/test_speciation.jl")
    include("unit/test_bloat_penalty.jl")
    include("unit/test_island_model.jl")

    # Benchmarks
    include("benchmarks/max_ones.jl")
    include("benchmarks/symbolic_regression.jl")
    include("benchmarks/boolean_parity.jl")
    include("benchmarks/koza_regression.jl")
end
