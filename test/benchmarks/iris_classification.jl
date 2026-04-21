# UCI Iris classification benchmark (Fisher 1936).
#
# 150 flowers × 3 classes (setosa, versicolor, virginica) × 4 features
# (sepal length/width, petal length/width).  The canonical real-world
# classification benchmark cited in essentially every ML tutorial.
# Well within GP's capabilities — a single threshold on petal_length
# separates setosa perfectly, and a diagonal line in petal-feature
# space separates versicolor from virginica with ~95% accuracy.
#
# Strategy: one-vs-rest with three independently-evolved TreeGenomes
# (one per class).  Predict by argmax across the three tree outputs.
# Each tree sees the same training features but with y=1.0 for its
# class and y=0.0 otherwise.  tanh is included in the operator set so
# outputs stay bounded.
#
# The dataset is embedded inline to avoid network dependency during
# tests.  Fisher's Iris is public domain.

using DynamicExpressions
using Random
using Statistics: mean

# Each row: (sepal_length, sepal_width, petal_length, petal_width, class_index)
# class: 1 = setosa, 2 = versicolor, 3 = virginica
const _IRIS_DATA = Tuple{Float32, Float32, Float32, Float32, Int}[
    # --- setosa (class 1) ---
    (5.1f0, 3.5f0, 1.4f0, 0.2f0, 1), (4.9f0, 3.0f0, 1.4f0, 0.2f0, 1),
    (4.7f0, 3.2f0, 1.3f0, 0.2f0, 1), (4.6f0, 3.1f0, 1.5f0, 0.2f0, 1),
    (5.0f0, 3.6f0, 1.4f0, 0.2f0, 1), (5.4f0, 3.9f0, 1.7f0, 0.4f0, 1),
    (4.6f0, 3.4f0, 1.4f0, 0.3f0, 1), (5.0f0, 3.4f0, 1.5f0, 0.2f0, 1),
    (4.4f0, 2.9f0, 1.4f0, 0.2f0, 1), (4.9f0, 3.1f0, 1.5f0, 0.1f0, 1),
    (5.4f0, 3.7f0, 1.5f0, 0.2f0, 1), (4.8f0, 3.4f0, 1.6f0, 0.2f0, 1),
    (4.8f0, 3.0f0, 1.4f0, 0.1f0, 1), (4.3f0, 3.0f0, 1.1f0, 0.1f0, 1),
    (5.8f0, 4.0f0, 1.2f0, 0.2f0, 1), (5.7f0, 4.4f0, 1.5f0, 0.4f0, 1),
    (5.4f0, 3.9f0, 1.3f0, 0.4f0, 1), (5.1f0, 3.5f0, 1.4f0, 0.3f0, 1),
    (5.7f0, 3.8f0, 1.7f0, 0.3f0, 1), (5.1f0, 3.8f0, 1.5f0, 0.3f0, 1),
    (5.4f0, 3.4f0, 1.7f0, 0.2f0, 1), (5.1f0, 3.7f0, 1.5f0, 0.4f0, 1),
    (4.6f0, 3.6f0, 1.0f0, 0.2f0, 1), (5.1f0, 3.3f0, 1.7f0, 0.5f0, 1),
    (4.8f0, 3.4f0, 1.9f0, 0.2f0, 1), (5.0f0, 3.0f0, 1.6f0, 0.2f0, 1),
    (5.0f0, 3.4f0, 1.6f0, 0.4f0, 1), (5.2f0, 3.5f0, 1.5f0, 0.2f0, 1),
    (5.2f0, 3.4f0, 1.4f0, 0.2f0, 1), (4.7f0, 3.2f0, 1.6f0, 0.2f0, 1),
    (4.8f0, 3.1f0, 1.6f0, 0.2f0, 1), (5.4f0, 3.4f0, 1.5f0, 0.4f0, 1),
    (5.2f0, 4.1f0, 1.5f0, 0.1f0, 1), (5.5f0, 4.2f0, 1.4f0, 0.2f0, 1),
    (4.9f0, 3.1f0, 1.5f0, 0.2f0, 1), (5.0f0, 3.2f0, 1.2f0, 0.2f0, 1),
    (5.5f0, 3.5f0, 1.3f0, 0.2f0, 1), (4.9f0, 3.6f0, 1.4f0, 0.1f0, 1),
    (4.4f0, 3.0f0, 1.3f0, 0.2f0, 1), (5.1f0, 3.4f0, 1.5f0, 0.2f0, 1),
    (5.0f0, 3.5f0, 1.3f0, 0.3f0, 1), (4.5f0, 2.3f0, 1.3f0, 0.3f0, 1),
    (4.4f0, 3.2f0, 1.3f0, 0.2f0, 1), (5.0f0, 3.5f0, 1.6f0, 0.6f0, 1),
    (5.1f0, 3.8f0, 1.9f0, 0.4f0, 1), (4.8f0, 3.0f0, 1.4f0, 0.3f0, 1),
    (5.1f0, 3.8f0, 1.6f0, 0.2f0, 1), (4.6f0, 3.2f0, 1.4f0, 0.2f0, 1),
    (5.3f0, 3.7f0, 1.5f0, 0.2f0, 1), (5.0f0, 3.3f0, 1.4f0, 0.2f0, 1),
    # --- versicolor (class 2) ---
    (7.0f0, 3.2f0, 4.7f0, 1.4f0, 2), (6.4f0, 3.2f0, 4.5f0, 1.5f0, 2),
    (6.9f0, 3.1f0, 4.9f0, 1.5f0, 2), (5.5f0, 2.3f0, 4.0f0, 1.3f0, 2),
    (6.5f0, 2.8f0, 4.6f0, 1.5f0, 2), (5.7f0, 2.8f0, 4.5f0, 1.3f0, 2),
    (6.3f0, 3.3f0, 4.7f0, 1.6f0, 2), (4.9f0, 2.4f0, 3.3f0, 1.0f0, 2),
    (6.6f0, 2.9f0, 4.6f0, 1.3f0, 2), (5.2f0, 2.7f0, 3.9f0, 1.4f0, 2),
    (5.0f0, 2.0f0, 3.5f0, 1.0f0, 2), (5.9f0, 3.0f0, 4.2f0, 1.5f0, 2),
    (6.0f0, 2.2f0, 4.0f0, 1.0f0, 2), (6.1f0, 2.9f0, 4.7f0, 1.4f0, 2),
    (5.6f0, 2.9f0, 3.6f0, 1.3f0, 2), (6.7f0, 3.1f0, 4.4f0, 1.4f0, 2),
    (5.6f0, 3.0f0, 4.5f0, 1.5f0, 2), (5.8f0, 2.7f0, 4.1f0, 1.0f0, 2),
    (6.2f0, 2.2f0, 4.5f0, 1.5f0, 2), (5.6f0, 2.5f0, 3.9f0, 1.1f0, 2),
    (5.9f0, 3.2f0, 4.8f0, 1.8f0, 2), (6.1f0, 2.8f0, 4.0f0, 1.3f0, 2),
    (6.3f0, 2.5f0, 4.9f0, 1.5f0, 2), (6.1f0, 2.8f0, 4.7f0, 1.2f0, 2),
    (6.4f0, 2.9f0, 4.3f0, 1.3f0, 2), (6.6f0, 3.0f0, 4.4f0, 1.4f0, 2),
    (6.8f0, 2.8f0, 4.8f0, 1.4f0, 2), (6.7f0, 3.0f0, 5.0f0, 1.7f0, 2),
    (6.0f0, 2.9f0, 4.5f0, 1.5f0, 2), (5.7f0, 2.6f0, 3.5f0, 1.0f0, 2),
    (5.5f0, 2.4f0, 3.8f0, 1.1f0, 2), (5.5f0, 2.4f0, 3.7f0, 1.0f0, 2),
    (5.8f0, 2.7f0, 3.9f0, 1.2f0, 2), (6.0f0, 2.7f0, 5.1f0, 1.6f0, 2),
    (5.4f0, 3.0f0, 4.5f0, 1.5f0, 2), (6.0f0, 3.4f0, 4.5f0, 1.6f0, 2),
    (6.7f0, 3.1f0, 4.7f0, 1.5f0, 2), (6.3f0, 2.3f0, 4.4f0, 1.3f0, 2),
    (5.6f0, 3.0f0, 4.1f0, 1.3f0, 2), (5.5f0, 2.5f0, 4.0f0, 1.3f0, 2),
    (5.5f0, 2.6f0, 4.4f0, 1.2f0, 2), (6.1f0, 3.0f0, 4.6f0, 1.4f0, 2),
    (5.8f0, 2.6f0, 4.0f0, 1.2f0, 2), (5.0f0, 2.3f0, 3.3f0, 1.0f0, 2),
    (5.6f0, 2.7f0, 4.2f0, 1.3f0, 2), (5.7f0, 3.0f0, 4.2f0, 1.2f0, 2),
    (5.7f0, 2.9f0, 4.2f0, 1.3f0, 2), (6.2f0, 2.9f0, 4.3f0, 1.3f0, 2),
    (5.1f0, 2.5f0, 3.0f0, 1.1f0, 2), (5.7f0, 2.8f0, 4.1f0, 1.3f0, 2),
    # --- virginica (class 3) ---
    (6.3f0, 3.3f0, 6.0f0, 2.5f0, 3), (5.8f0, 2.7f0, 5.1f0, 1.9f0, 3),
    (7.1f0, 3.0f0, 5.9f0, 2.1f0, 3), (6.3f0, 2.9f0, 5.6f0, 1.8f0, 3),
    (6.5f0, 3.0f0, 5.8f0, 2.2f0, 3), (7.6f0, 3.0f0, 6.6f0, 2.1f0, 3),
    (4.9f0, 2.5f0, 4.5f0, 1.7f0, 3), (7.3f0, 2.9f0, 6.3f0, 1.8f0, 3),
    (6.7f0, 2.5f0, 5.8f0, 1.8f0, 3), (7.2f0, 3.6f0, 6.1f0, 2.5f0, 3),
    (6.5f0, 3.2f0, 5.1f0, 2.0f0, 3), (6.4f0, 2.7f0, 5.3f0, 1.9f0, 3),
    (6.8f0, 3.0f0, 5.5f0, 2.1f0, 3), (5.7f0, 2.5f0, 5.0f0, 2.0f0, 3),
    (5.8f0, 2.8f0, 5.1f0, 2.4f0, 3), (6.4f0, 3.2f0, 5.3f0, 2.3f0, 3),
    (6.5f0, 3.0f0, 5.5f0, 1.8f0, 3), (7.7f0, 3.8f0, 6.7f0, 2.2f0, 3),
    (7.7f0, 2.6f0, 6.9f0, 2.3f0, 3), (6.0f0, 2.2f0, 5.0f0, 1.5f0, 3),
    (6.9f0, 3.2f0, 5.7f0, 2.3f0, 3), (5.6f0, 2.8f0, 4.9f0, 2.0f0, 3),
    (7.7f0, 2.8f0, 6.7f0, 2.0f0, 3), (6.3f0, 2.7f0, 4.9f0, 1.8f0, 3),
    (6.7f0, 3.3f0, 5.7f0, 2.1f0, 3), (7.2f0, 3.2f0, 6.0f0, 1.8f0, 3),
    (6.2f0, 2.8f0, 4.8f0, 1.8f0, 3), (6.1f0, 3.0f0, 4.9f0, 1.8f0, 3),
    (6.4f0, 2.8f0, 5.6f0, 2.1f0, 3), (7.2f0, 3.0f0, 5.8f0, 1.6f0, 3),
    (7.4f0, 2.8f0, 6.1f0, 1.9f0, 3), (7.9f0, 3.8f0, 6.4f0, 2.0f0, 3),
    (6.4f0, 2.8f0, 5.6f0, 2.2f0, 3), (6.3f0, 2.8f0, 5.1f0, 1.5f0, 3),
    (6.1f0, 2.6f0, 5.6f0, 1.4f0, 3), (7.7f0, 3.0f0, 6.1f0, 2.3f0, 3),
    (6.3f0, 3.4f0, 5.6f0, 2.4f0, 3), (6.4f0, 3.1f0, 5.5f0, 1.8f0, 3),
    (6.0f0, 3.0f0, 4.8f0, 1.8f0, 3), (6.9f0, 3.1f0, 5.4f0, 2.1f0, 3),
    (6.7f0, 3.1f0, 5.6f0, 2.4f0, 3), (6.9f0, 3.1f0, 5.1f0, 2.3f0, 3),
    (5.8f0, 2.7f0, 5.1f0, 1.9f0, 3), (6.8f0, 3.2f0, 5.9f0, 2.3f0, 3),
    (6.7f0, 3.3f0, 5.7f0, 2.5f0, 3), (6.7f0, 3.0f0, 5.2f0, 2.3f0, 3),
    (6.3f0, 2.5f0, 5.0f0, 1.9f0, 3), (6.5f0, 3.0f0, 5.2f0, 2.0f0, 3),
    (6.2f0, 3.4f0, 5.4f0, 2.3f0, 3), (5.9f0, 3.0f0, 5.1f0, 1.8f0, 3),
]

@testset "UCI Iris classification (TreeGenome, one-vs-rest)" begin
    @test length(_IRIS_DATA) == 150
    @test count(r -> r[5] == 1, _IRIS_DATA) == 50
    @test count(r -> r[5] == 2, _IRIS_DATA) == 50
    @test count(r -> r[5] == 3, _IRIS_DATA) == 50

    # --- Stratified 80/20 split (40 train + 10 test per class) ---
    # Fixed seed so the split is identical across runs.
    split_rng = MersenneTwister(20260421)
    train_idx = Int[]
    test_idx  = Int[]
    for cls in 1:3
        class_rows = findall(r -> r[5] == cls, _IRIS_DATA)
        perm = class_rows[randperm(split_rng, length(class_rows))]
        append!(train_idx, perm[1:40])
        append!(test_idx,  perm[41:50])
    end

    function _features_matrix(idxs)
        X = Matrix{Float32}(undef, 4, length(idxs))
        for (col, i) in enumerate(idxs)
            r = _IRIS_DATA[i]
            X[1, col] = r[1]
            X[2, col] = r[2]
            X[3, col] = r[3]
            X[4, col] = r[4]
        end
        return X
    end
    _labels(idxs) = Int[_IRIS_DATA[i][5] for i in idxs]

    X_train = _features_matrix(train_idx)
    y_labels_train = _labels(train_idx)
    X_test  = _features_matrix(test_idx)
    y_labels_test  = _labels(test_idx)

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
            y_ovr = Float32[(_IRIS_DATA[i][5] == cls) ? 1f0 : 0f0 for i in train_idx]
            evaluator = TreeFitnessEvaluator(X_train, y_ovr, operators)
            # Offset seed per class so the three trees aren't identical.
            problem = GPProblem(evaluator, TreeGenome{Float32};
                                seed=seed * 100 + cls)
            result = solve(problem, algorithm; verbose=false)
            trees[cls] = result.best_genome.tree
        end

        # Predict on test set.
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

    mean_acc = mean(accuracies)
    n_above_90 = count(a -> a >= 0.90, accuracies)
    println("  Iris: $n_above_90/5 seeds >= 90% test accuracy " *
            "(mean = $(round(mean_acc * 100, digits=1))%)")
    flush(stdout)
    @test n_above_90 >= 4
end
