using Test
using Arborist
using Random

# Benchmarks are opt-in due to long runtime.
# Run with: ARBORIST_RUN_BENCHMARKS=true julia -e 'using Pkg; Pkg.test("Arborist")'
const RUN_BENCHMARKS = get(ENV, "ARBORIST_RUN_BENCHMARKS",
                           get(ENV, "GENPROG_RUN_BENCHMARKS", "false")) == "true"

@testset "Arborist.jl" begin
    # Unit tests — always run, target < 30s total
    include("unit/test_evaluators.jl")
    include("unit/test_protected_operators.jl")
    include("unit/test_genome.jl")
    include("unit/test_operators.jl")
    include("unit/test_tree_caps.jl")
    include("unit/test_show_methods.jl")
    include("unit/test_dot_export.jl")
    include("unit/test_speciation.jl")
    include("unit/test_bloat_penalty.jl")
    include("unit/test_island_model.jl")
    include("unit/test_topology.jl")
    include("unit/test_migrant_genome.jl")
    include("unit/test_sanitizer.jl")
    include("unit/test_ant_genome.jl")
    include("unit/test_graph_genome.jl")
    include("unit/test_episodic_evaluator.jl")
    include("unit/test_coverage_gaps.jl")
    include("unit/test_nsga2.jl")
    include("unit/test_prompt_context.jl")
    include("unit/test_behavioral_init.jl")
    include("unit/test_run_log.jl")
    include("unit/test_evaluate_cases.jl")
    include("unit/test_lexicase.jl")
    include("unit/test_constant_optimization.jl")
    include("unit/test_novelty.jl")
    include("unit/test_map_elites.jl")
    include("unit/test_checkpoint.jl")
    include("unit/test_plots_ext.jl")
    include("unit/test_adf_genome.jl")
    include("unit/test_cmaes.jl")

    # Integration tests — always run, target < 60s total
    include("integration/test_llm_operator.jl")
    include("integration/test_tree_genome.jl")
    include("integration/test_distributed_islands.jl")
    include("integration/test_island_tree_genome.jl")

    if RUN_BENCHMARKS
        @info "Running full benchmark suite (ARBORIST_RUN_BENCHMARKS=true)"
        include("benchmarks/max_ones.jl")
        include("benchmarks/symbolic_regression.jl")
        include("benchmarks/koza_regression.jl")
        include("benchmarks/nguyen_regression.jl")
        include("benchmarks/keijzer_extrapolation.jl")
        include("benchmarks/boolean_parity.jl")
        include("benchmarks/multiplexer.jl")
        include("benchmarks/iris_classification.jl")
        include("benchmarks/ant_trail.jl")
        include("benchmarks/speedup_benchmark.jl")
        include("benchmarks/xor_neat.jl")
        include("benchmarks/xor_nsga2_neat.jl")
        include("benchmarks/parity_neat.jl")
        include("benchmarks/two_spirals_neat.jl")
        include("benchmarks/two_spirals_nsga2_neat.jl")
        include("benchmarks/cartpole_neat.jl")
        include("benchmarks/double_pole_neat.jl")
        include("benchmarks/mountain_car_neat.jl")
        include("benchmarks/acrobot_neat.jl")
        include("benchmarks/sequence_memory_neat.jl")
        include("benchmarks/sequence_recall_neat.jl")
        include("benchmarks/retina_neat.jl")
        include("benchmarks/mackey_glass_neat.jl")
        include("benchmarks/retina_nsga2_neat.jl")
        include("benchmarks/lorenz_recovery.jl")
        include("benchmarks/lexicase_modal_regression.jl")
    else
        @info "Skipping benchmarks (set ARBORIST_RUN_BENCHMARKS=true to run)"
    end
end
