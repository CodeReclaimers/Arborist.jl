@testset "Speciation" begin
    @testset "NoSpeciation" begin
        ns = NoSpeciation()
        @test ns isa AbstractSpeciation
    end
end
