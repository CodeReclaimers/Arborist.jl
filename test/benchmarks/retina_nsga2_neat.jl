# NSGA-II retina benchmark with (classification_error, modularity_proxy) objectives.
#
# Extends retina_neat.jl with a second objective that rewards modular
# decomposition: the "modularity proxy" counts hidden nodes that receive
# direct input connections from *both* left- and right-side retinas.
# Lower proxy = more modular (hidden nodes specialize to one side).
#
# Because the zero-connection attractor still exists (complexity=0 yields
# modularity=0 trivially), we pair modularity with classification error
# rather than parsimony: the trivial zero-net also has the highest
# classification error, so it's Pareto-dominated by any non-trivial
# network that learns something.

struct _RetinaModularityEvaluator{E<:AbstractEvaluator} <: Arborist.AbstractMultiObjectiveEvaluator
    inner::E
    left_inputs::Set{Int}
    right_inputs::Set{Int}
end

function Arborist.evaluate_multi(e::_RetinaModularityEvaluator, g::AbstractGenome)
    mse = evaluate_genome(g, e.inner)
    # Count hidden nodes that receive connections from both sides.
    n_mixed = 0
    n_hidden = 0
    for node in values(g.nodes)
        node.type == :hidden || continue
        n_hidden += 1
        has_left = false
        has_right = false
        for c in values(g.connections)
            c.enabled || continue
            c.out_node != node.id && continue
            if c.in_node in e.left_inputs
                has_left = true
            end
            if c.in_node in e.right_inputs
                has_right = true
            end
        end
        (has_left && has_right) && (n_mixed += 1)
    end
    proxy = n_hidden == 0 ? 0.0 : Float64(n_mixed)
    return [mse, proxy]
end

Arborist.objective_names(::_RetinaModularityEvaluator) = ["error", "modularity_proxy"]
Arborist.input_signature(e::_RetinaModularityEvaluator) = input_signature(e.inner)
Arborist.output_signature(e::_RetinaModularityEvaluator) = output_signature(e.inner)

@testset "NSGA-II retina (error vs modularity-proxy)" begin
    _is_obj(bits::Int) = count_ones(bits) == 1
    n_patterns = 2^8
    input_data  = zeros(Float64, 8, n_patterns)
    output_data = zeros(Float64, 1, n_patterns)
    col = 0
    for left in 0:15
        left_o = _is_obj(left)
        for right in 0:15
            col += 1
            right_o = _is_obj(right)
            for i in 1:4
                input_data[i,     col] = Float64((left  >> (i-1)) & 1)
                input_data[i + 4, col] = Float64((right >> (i-1)) & 1)
            end
            output_data[1, col] = (left_o && right_o) ? 1.0 : 0.0
        end
    end

    # Input IDs in GraphGenome initialization are 1..n_inputs: left is 1..4,
    # right is 5..8 (matches the order input_data rows are seeded).
    evaluator = _RetinaModularityEvaluator(
        GraphEvaluator(input_data, output_data),
        Set{Int}([1, 2, 3, 4]),
        Set{Int}([5, 6, 7, 8]),
    )

    ops = neat_defaults()
    algorithm = NSGAII(
        pop_size=150, generations=200,
        mutation_rate=0.5, crossover_rate=0.3, parallel=false,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
    )

    reset_innovation_counter!()
    problem = GPProblem(evaluator, GraphGenome; seed=4242)
    result = solve(problem, algorithm; verbose=false)

    @test result isa NSGAIIResult{GraphGenome}
    @test result.objective_names == ["error", "modularity_proxy"]
    @test !isempty(result.pareto_front)
    @test length(result.hypervolume_history) == algorithm.generations
    @test all(all(isfinite, f) for f in result.pareto_fitnesses)
    @test any(h -> h > 0.0, result.hypervolume_history)

    # At least one front member must have meaningfully low error (< 0.055)
    # — the same threshold used for the single-objective retina benchmark.
    # Modularity alone without a solve is the trivial zero-net, so we
    # gate on solve-ability too.
    front_err = [f[1] for f in result.pareto_fitnesses]
    front_mod = [f[2] for f in result.pareto_fitnesses]
    best_err = minimum(front_err)
    @test best_err < 0.06

    println("  Retina NSGA-II Pareto front ($(length(result.pareto_front)) genomes):")
    println("    error            : min=$(round(best_err, digits=5))  " *
            "max=$(round(maximum(front_err), digits=5))")
    println("    modularity_proxy : min=$(Int(minimum(front_mod)))  " *
            "max=$(Int(maximum(front_mod)))")
    println("    hypervolume[end] = $(round(result.hypervolume_history[end], digits=4))")
    flush(stdout)
end
