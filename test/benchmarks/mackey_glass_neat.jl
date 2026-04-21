# Mackey-Glass (τ=17) time-series prediction benchmark with recurrent GraphGenome.
#
# The canonical chaotic-dynamics test for recurrent networks. We generate a
# τ=17 Mackey-Glass series (`dx/dt = 0.2·x(t-τ)/(1 + x(t-τ)^10) - 0.1·x(t)`)
# via Runge-Kutta-4, use the first N samples for one-step-ahead prediction.
# GraphEvaluator in recurrent mode treats the sequence as a time series with
# persistent node state, so the network can build up internal memory.

function _mackey_glass_series(n::Int; τ::Int = 17,
                               β::Float64 = 0.2, γ::Float64 = 0.1, p::Int = 10,
                               dt::Float64 = 1.0, burn_in::Int = 200)
    # History buffer length covers the delay.
    total = n + burn_in
    x = zeros(Float64, total + τ + 1)
    # Seed with a small constant; delay region uses the same.
    for i in 1:(τ + 1)
        x[i] = 1.2
    end
    function dxdt(xt, xt_delay)
        return β * xt_delay / (1 + xt_delay^p) - γ * xt
    end
    for t in (τ + 1):(total + τ)
        xt = x[t]
        xt_delay = x[t - τ]
        # RK4
        k1 = dxdt(xt, xt_delay)
        k2 = dxdt(xt + 0.5 * dt * k1, xt_delay)
        k3 = dxdt(xt + 0.5 * dt * k2, xt_delay)
        k4 = dxdt(xt + dt * k3,        xt_delay)
        x[t + 1] = xt + (dt / 6.0) * (k1 + 2k2 + 2k3 + k4)
    end
    # Return the post-burn-in segment.
    return x[(τ + burn_in + 1):(τ + burn_in + n + 1)]
end

@testset "Mackey-Glass τ=17 prediction (recurrent GraphGenome)" begin
    # 500 training points + 1-step-ahead prediction target.
    n_train = 500
    series = _mackey_glass_series(n_train; τ=17, burn_in=500)  # length n_train+1
    inputs  = series[1:n_train]
    targets = series[2:(n_train + 1)]

    input_data  = reshape(inputs,  1, :)   # 1 × n_train
    output_data = reshape(targets, 1, :)

    # Feedforward baseline: cannot exploit temporal structure.
    ff_eval = GraphEvaluator(input_data, output_data; allow_recurrent=false)
    # Recurrent solver.
    rec_eval = GraphEvaluator(input_data, output_data;
                              allow_recurrent=true, relaxation_passes=2)

    ops = neat_defaults()
    algorithm = GeneticProgramming(
        pop_size=150, generations=100,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0, min_species_size=2,
                                       stagnation_limit=20),
    )

    # Run 3 seeds each, compare rec vs ff.
    ff_fits  = Float64[]
    rec_fits = Float64[]
    for seed in 1:3
        reset_innovation_counter!()
        ff_problem = GPProblem(ff_eval, GraphGenome; seed=seed)
        ff_result  = solve(ff_problem, algorithm; verbose=false)
        push!(ff_fits, ff_result.best_fitness)

        reset_innovation_counter!()
        rec_problem = GPProblem(rec_eval, GraphGenome; seed=seed)
        rec_result  = solve(rec_problem, algorithm; verbose=false)
        push!(rec_fits, rec_result.best_fitness)

        println("    Mackey-Glass seed=$seed: ff_mse=$(round(ff_result.best_fitness, digits=5)), " *
                "rec_mse=$(round(rec_result.best_fitness, digits=5))")
        flush(stdout)
    end

    ff_best  = minimum(ff_fits)
    rec_best = minimum(rec_fits)
    println("  Mackey-Glass best: ff=$(round(ff_best, digits=5)), rec=$(round(rec_best, digits=5))")
    flush(stdout)

    # Infrastructure check: both produce finite fitness.
    @test isfinite(ff_best)
    @test isfinite(rec_best)

    # One-step-ahead MG prediction is easy (serial correlation is strong).
    # Both modes should reach very low MSE. The recurrent variant should
    # not regress; we allow equality with a small epsilon for seed noise.
    @test rec_best <= ff_best + 1e-3
end
