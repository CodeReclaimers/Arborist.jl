using Test
using GenProg
using Random

@testset "GenProg.jl" begin
    include("unit/test_evaluators.jl")
    include("unit/test_genome.jl")
    include("unit/test_operators.jl")
    include("unit/test_speciation.jl")
    include("benchmarks/max_ones.jl")
    include("benchmarks/symbolic_regression.jl")
end
