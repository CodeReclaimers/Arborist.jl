# UCI Iris classification benchmark (Fisher 1936).
#
# Dataset + stratified 80/20 split sourced from
# `Arborist.Benchmarks.iris(; rng=MersenneTwister(20260421))`, which
# preserves the historical split used before the Benchmarks submodule.
# Strategy: one-vs-rest with three independently-evolved TreeGenomes
# (one per class).  Predict by argmax across the three tree outputs.

using DynamicExpressions
using Random

# Local mean helper — Pkg.test runs in a sandboxed env where stdlib
# Statistics is not available.
_mean_iris(x) = sum(x) / length(x)

@testset "UCI Iris classification (TreeGenome, one-vs-rest)" begin
    iris_ds = Arborist.Benchmarks.iris(; rng=MersenneTwister(20260421),
                                         test_ratio=0.2)

    X_train = iris_ds.X_train
    y_labels_train = iris_ds.y_train
    X_test  = iris_ds.X_test
    y_labels_test  = iris_ds.y_test

    @test size(X_train) == (4, 120)
    @test size(X_test)  == (4, 30)

    operators = OperatorEnum(;
        binary_operators=[+, -, *, /],
        unary_operators=[tanh]
    )

    algorithm = GeneticProgramming(
        pop_size=100,
        generations=100,
        mutation_rate=0.4,
        crossover_rate=0.2,
        elitism=2
    )

    # For each seed, evolve three trees (one per class) and classify
    # test points by argmax of the three tree outputs.
    accuracies = map(1:5) do seed
        trees = Vector{Any}(undef, 3)
        for cls in 1:3
            y_ovr = Float32[(y_labels_train[i] == cls) ? 1f0 : 0f0
                            for i in eachindex(y_labels_train)]
            evaluator = TreeFitnessEvaluator(X_train, y_ovr, operators)
            # Offset seed per class so the three trees aren't identical.
            problem = GPProblem(evaluator, TreeGenome{Float32};
                                seed=seed * 100 + cls)
            result = solve(problem, algorithm; verbose=false)
            trees[cls] = result.best_genome.tree
        end

        test_scores = zeros(Float32, 3, size(X_test, 2))
        for cls in 1:3
            test_scores[cls, :] = trees[cls](X_test, operators)
        end

        n_correct = 0
        for col in 1:size(X_test, 2)
            pred = argmax(view(test_scores, :, col))
            if pred == y_labels_test[col]
                n_correct += 1
            end
        end
        acc = n_correct / length(y_labels_test)
        println("    Iris seed=$seed: test accuracy = $(round(acc * 100, digits=1))%")
        flush(stdout)
        acc
    end

    mean_acc = _mean_iris(accuracies)
    n_above_90 = count(a -> a >= 0.90, accuracies)
    println("  Iris: $n_above_90/5 seeds >= 90% test accuracy " *
            "(mean = $(round(mean_acc * 100, digits=1))%)")
    flush(stdout)
    @test n_above_90 >= 4
end
