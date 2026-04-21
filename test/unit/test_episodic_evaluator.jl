@testset "EpisodicEvaluator" begin
    @testset "reproducibility: same seed yields same fitness" begin
        # Trivial 1-D accumulator dynamics: state is a scalar, action is a
        # scalar, next_state = state + action, reward = -abs(next_state).
        # initial_state draws state from N(0, 1) — seed-dependent, so if
        # the evaluator re-seeds per episode we should see identical fitness
        # across two evaluate_genome calls.
        reset_innovation_counter!()
        rng = Random.MersenneTwister(111)
        g = initialize(GraphGenome, 1, 1, rng)

        e = EpisodicEvaluator(
            1, 1,
            rng -> randn(rng),                    # initial_state
            (s, a) -> s + a,                      # dynamics
            (s, a, sp) -> -abs(sp),               # reward
            s -> false,                           # done (always run to max)
            s -> Float64[s],                      # observe
            y -> y[1],                            # decode_action
            max_steps=10, n_episodes=3,
            episode_seed_base=42,
        )

        fit1 = evaluate_genome(g, e)
        fit2 = evaluate_genome(g, e)
        @test fit1 == fit2
        @test isfinite(fit1)
    end

    @testset "n_episodes averaging" begin
        # Constant-reward environment: reward = 1 every step, no variance.
        # mean_reward across n_episodes should always be identical, so
        # fitness = -mean_reward should equal -max_steps regardless of
        # n_episodes.
        reset_innovation_counter!()
        rng = Random.MersenneTwister(222)
        g = initialize(GraphGenome, 1, 1, rng)

        make_eval(n_eps) = EpisodicEvaluator(
            1, 1,
            rng -> 0.0,
            (s, a) -> s,
            (s, a, sp) -> 1.0,
            s -> false,
            s -> Float64[0.0],
            y -> 0.0,
            max_steps=5, n_episodes=n_eps,
            episode_seed_base=0,
        )

        @test evaluate_genome(g, make_eval(1)) == -5.0
        @test evaluate_genome(g, make_eval(5)) == -5.0
        @test evaluate_genome(g, make_eval(10)) == -5.0
    end

    @testset "done terminates episode early" begin
        # Dynamics: state integer, increments each step, done when state >= 3.
        # Reward = 1/step. An episode should accumulate exactly 3 reward.
        reset_innovation_counter!()
        rng = Random.MersenneTwister(333)
        g = initialize(GraphGenome, 1, 1, rng)

        e = EpisodicEvaluator(
            1, 1,
            rng -> 0,
            (s, a) -> s + 1,
            (s, a, sp) -> 1.0,
            s -> s >= 3,
            s -> Float64[Float64(s)],
            y -> 0,
            max_steps=100, n_episodes=1,
            episode_seed_base=0,
        )

        # 3 steps until done (state goes 0 → 1 → 2 → 3; loop-check sees 3 ≥ 3).
        @test evaluate_genome(g, e) == -3.0
    end

    @testset "done-at-start returns zero reward" begin
        # If done(initial_state) is already true, the loop never runs and
        # reward totals zero.
        reset_innovation_counter!()
        rng = Random.MersenneTwister(444)
        g = initialize(GraphGenome, 1, 1, rng)

        e = EpisodicEvaluator(
            1, 1,
            rng -> 0,
            (s, a) -> s,
            (s, a, sp) -> 1.0,
            s -> true,              # done immediately
            s -> Float64[0.0],
            y -> 0,
            max_steps=100, n_episodes=2,
            episode_seed_base=0,
        )

        @test evaluate_genome(g, e) == 0.0
    end

    @testset "dimension mismatch returns Inf" begin
        # Genome has 1 input, but evaluator declares n_inputs=2. Should
        # return Inf rather than crash.
        reset_innovation_counter!()
        rng = Random.MersenneTwister(555)
        g = initialize(GraphGenome, 1, 1, rng)

        e = EpisodicEvaluator(
            2, 1,
            rng -> 0.0,
            (s, a) -> s,
            (s, a, sp) -> 1.0,
            s -> false,
            s -> Float64[0.0, 0.0],  # two-dim obs — but genome has only 1 input
            y -> 0.0,
            max_steps=5, n_episodes=1,
        )

        @test evaluate_genome(g, e) == Inf
    end

    @testset "recurrent mode accepts cyclic topology" begin
        # Force a cyclic topology by adding a self-connection and running
        # both allow_recurrent=true and allow_recurrent=false. Recurrent
        # mode must tolerate the cycle; feedforward must report Inf.
        reset_innovation_counter!()
        rng = Random.MersenneTwister(666)
        g = initialize(GraphGenome, 1, 1, rng)

        # Inject a self-loop on an output (or hidden) node. Pick any non-input
        # node and add a connection to itself.
        output_id = first(n.id for n in values(g.nodes) if n.type == :output)
        c = ConnectionGene(output_id, output_id, 0.5, true, 999_999)
        g.connections[c.innovation] = c

        rec_eval = EpisodicEvaluator(
            1, 1,
            rng -> 0.0, (s, a) -> s, (s, a, sp) -> 1.0,
            s -> false, s -> Float64[0.0], y -> 0.0;
            max_steps=3, n_episodes=1, allow_recurrent=true,
        )
        ff_eval = EpisodicEvaluator(
            1, 1,
            rng -> 0.0, (s, a) -> s, (s, a, sp) -> 1.0,
            s -> false, s -> Float64[0.0], y -> 0.0;
            max_steps=3, n_episodes=1, allow_recurrent=false,
        )

        @test isfinite(evaluate_genome(g, rec_eval))
        @test evaluate_genome(g, ff_eval) == Inf
    end

    @testset "recurrent hidden state persists across timesteps" begin
        # Build a minimal GraphGenome with a hidden node that has a
        # self-connection. Drive it with zero input. If hidden state
        # persists across timesteps under `allow_recurrent=true`, the
        # hidden value evolves over steps; if it were reset each step,
        # the accumulator would stay at zero forever.
        reset_innovation_counter!()

        # Hand-built: input, bias, hidden (self-loop), output. All :identity.
        nodes = Dict{Int,NodeGene}(
            1 => NodeGene(1, :input,  :identity),
            2 => NodeGene(2, :bias,   :identity),
            3 => NodeGene(3, :hidden, :identity),
            4 => NodeGene(4, :output, :identity),
        )
        connections = Dict{Int,ConnectionGene}(
            1 => ConnectionGene(2, 3, 1.0, true, 1),  # bias  -> hidden, w=1.0
            2 => ConnectionGene(3, 3, 0.5, true, 2),  # hidden-> hidden, w=0.5 (self-loop)
            3 => ConnectionGene(3, 4, 1.0, true, 3),  # hidden-> output, w=1.0
        )
        g = GraphGenome(nodes, connections, 1, 1, NaN)

        # Expose the network's per-step output via the reward channel:
        # dynamics accumulates action into state, reward = (s' - s) = action.
        # Sum of rewards over the episode = sum of per-step outputs.
        e_rec = EpisodicEvaluator(
            1, 1,
            rng -> 0.0, (s, a) -> s + a, (s, a, sp) -> sp - s,
            s -> false, s -> Float64[0.0], y -> y[1];
            max_steps=4, n_episodes=1,
            allow_recurrent=true, relaxation_passes=1,
        )
        e_ff = EpisodicEvaluator(
            1, 1,
            rng -> 0.0, (s, a) -> s + a, (s, a, sp) -> sp - s,
            s -> false, s -> Float64[0.0], y -> y[1];
            max_steps=4, n_episodes=1,
            allow_recurrent=false, relaxation_passes=1,
        )

        # Feedforward must refuse the cyclic topology.
        @test evaluate_genome(g, e_ff) == Inf

        # Recurrent trace with relaxation_passes=1. In one relaxation
        # pass per step, each node reads from `prev` (the snapshot taken
        # at the top of that pass), so the output lags the hidden update
        # by one pass. Tracing node_vals across the four steps:
        #
        #   state  node_vals[3]  node_vals[4]  output (=prev[3])
        #   init    0             0             —
        #   step1:  1.0           0.0           0.0    (prev[3]=0)
        #   step2:  1.5           1.0           1.0    (prev[3]=1.0)
        #   step3:  1.75          1.5           1.5    (prev[3]=1.5)
        #   step4:  1.875         1.75          1.75   (prev[3]=1.75)
        #
        # Sum of outputs = 0 + 1.0 + 1.5 + 1.75 = 4.25. Fitness is the
        # negated mean reward: -4.25 / 1 episode = -4.25.
        # If hidden state were reset every step, node_vals[3] would
        # always be 1.0 by step's end and output would always be 0
        # (prev[3] is zero at the top of every step), giving fitness 0.
        # Hitting -4.25 proves persistence, *and* confirms the one-pass
        # output lag which is the correct semantics of relaxation_passes=1.
        @test evaluate_genome(g, e_rec) ≈ -4.25
    end
end
