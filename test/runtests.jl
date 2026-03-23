using Test
using Arborist
using Random

# Benchmarks are opt-in due to long runtime.
# Run with: GENPROG_RUN_BENCHMARKS=true julia -e 'using Pkg; Pkg.test("Arborist")'
const RUN_BENCHMARKS = get(ENV, "GENPROG_RUN_BENCHMARKS", "false") == "true"

@testset "Arborist.jl" begin
    # Unit tests — always run, target < 30s total
    include("unit/test_evaluators.jl")
    include("unit/test_genome.jl")
    include("unit/test_operators.jl")
    include("unit/test_speciation.jl")
    include("unit/test_bloat_penalty.jl")
    include("unit/test_island_model.jl")
    include("unit/test_sanitizer.jl")
    include("unit/test_ant_genome.jl")
    include("unit/test_graph_genome.jl")

    # Integration tests — always run, target < 60s total
    include("integration/test_llm_operator.jl")
    include("integration/test_tree_genome.jl")

    if RUN_BENCHMARKS
        @info "Running full benchmark suite (GENPROG_RUN_BENCHMARKS=true)"
        include("benchmarks/max_ones.jl")
        include("benchmarks/symbolic_regression.jl")
        include("benchmarks/koza_regression.jl")
        include("benchmarks/boolean_parity.jl")
        include("benchmarks/ant_trail.jl")
        include("benchmarks/speedup_benchmark.jl")
        include("benchmarks/xor_neat.jl")
    else
        @info "Skipping benchmarks (set GENPROG_RUN_BENCHMARKS=true to run)"
    end
end
