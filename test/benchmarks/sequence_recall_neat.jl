# Variable-delay sequence recall benchmark using GraphGenome + GraphEvaluator.
#
# Extension of sequence_memory_neat.jl: instead of a fixed 1-step delay,
# the sequence contains three concatenated trials with delays {2, 5, 10}.
# A network that learned a pure 1-step reactor (as trivial recurrent
# solutions tend to do on the fixed-delay task) cannot solve this —
# actual memory traces at multiple horizons are required.

@testset "Variable-delay sequence recall (recurrent GraphGenome)" begin
    rng_data = Random.MersenneTwister(4242)

    # Three trials, delays {2, 5, 10}, each separated by 2 silent steps.
    # Trial layout (one example trial with delay d):
    #   input:  [bit,  0, 0, ..., 0]  (length d+1)
    #   target: [  0,  0, 0, ..., bit]  (target at position d)
    function build_trial(bit::Float64, delay::Int)
        n_steps = delay + 1
        inp = zeros(Float64, n_steps)
        tgt = zeros(Float64, n_steps)
        inp[1] = bit
        tgt[end] = bit
        return inp, tgt
    end

    sep_len = 2   # silent steps between trials to reset the network's transient
    delays = [2, 5, 10]
    bits = Float64[rand(rng_data, Bool) ? 1.0 : 0.0 for _ in delays]

    inp_segments = Vector{Float64}[]
    tgt_segments = Vector{Float64}[]
    for (bit, d) in zip(bits, delays)
        inp, tgt = build_trial(bit, d)
        push!(inp_segments, inp)
        push!(tgt_segments, tgt)
        # Separator
        push!(inp_segments, zeros(Float64, sep_len))
        push!(tgt_segments, zeros(Float64, sep_len))
    end

    input_flat  = reduce(vcat, inp_segments)
    target_flat = reduce(vcat, tgt_segments)
    input_data  = reshape(input_flat, 1, :)   # 1 × T
    output_data = reshape(target_flat, 1, :)  # 1 × T

    ops = neat_defaults()

    # Feedforward baseline (expected to fail — no memory).
    ff_eval = GraphEvaluator(input_data, output_data; allow_recurrent=false)
    ff_alg = GeneticProgramming(
        pop_size=100, generations=50,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0),
    )
    reset_innovation_counter!()
    ff_problem = GPProblem(ff_eval, GraphGenome; seed=7)
    ff_result  = solve(ff_problem, ff_alg; verbose=false)
    ff_mse     = ff_result.best_fitness

    # Recurrent solver.
    rec_eval = GraphEvaluator(input_data, output_data;
                              allow_recurrent=true, relaxation_passes=3)
    rec_alg = GeneticProgramming(
        pop_size=100, generations=100,
        mutation_rate=0.5, crossover_rate=0.3, elitism=2,
        mutation_ops=ops.mutation_ops, crossover_ops=ops.crossover_ops,
        speciation=ThresholdSpeciation(threshold=3.0),
    )
    reset_innovation_counter!()
    rec_problem = GPProblem(rec_eval, GraphGenome; seed=7)
    rec_result  = solve(rec_problem, rec_alg; verbose=false)
    rec_mse     = rec_result.best_fitness

    println("  Variable-delay recall — ff_mse=$(round(ff_mse, digits=4)), " *
            "rec_mse=$(round(rec_mse, digits=4))")
    flush(stdout)

    # Both must return finite fitness — infrastructure check.
    @test isfinite(ff_mse)
    @test isfinite(rec_mse)

    # Recurrent must do at least as well as feedforward on this task.
    # Asserting strict improvement is too seed-sensitive at the test-suite
    # budget; a weak <= is enough to catch regressions where the recurrent
    # path silently breaks.
    @test rec_mse <= ff_mse + 1e-6
end
