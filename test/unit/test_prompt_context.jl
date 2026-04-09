@testset "Prompt enrichment" begin
    using Arborist: MutationContext, FitnessSection, ElitesSection, GenerationSection,
                    render, render_enrichment, AbstractPromptSection,
                    _find_llm_operators, _update_llm_contexts!, _set_parent_context!

    # Helper: build a populated context for testing.
    function make_test_context(; gen=50, max_gen=100,
                                fitnesses=[0.5, 0.8, 1.0, 1.2, 1.5],
                                serialized=["prog_a", "prog_b", "prog_c"],
                                parent_fit=0.8, parent_rank=2)
        MutationContext(gen, max_gen, fitnesses, serialized, parent_fit, parent_rank)
    end

    @testset "MutationContext default constructor" begin
        ctx = MutationContext()
        @test ctx.generation == 0
        @test ctx.max_generations == 0
        @test isempty(ctx.fitnesses)
        @test isempty(ctx.genomes_serialized)
        @test ctx.parent_fitness == Inf
        @test ctx.parent_rank == 0
        println("  MutationContext(): defaults correct")
    end

    @testset "FitnessSection rendering" begin
        ctx = make_test_context()
        text = render(FitnessSection(), ctx)
        @test occursin("Fitness Context", text)
        @test occursin("0.8", text)       # parent fitness
        @test occursin("rank 2/5", text)  # parent rank / pop size
        @test occursin("0.5", text)       # population best
        @test occursin("Lower fitness is better", text)
        println("  FitnessSection: rendered with fitness=0.8, rank=2/5, best=0.5")
    end

    @testset "FitnessSection empty context" begin
        ctx = MutationContext()
        text = render(FitnessSection(), ctx)
        @test text == ""
        println("  FitnessSection: empty context -> empty string")
    end

    @testset "ElitesSection rendering" begin
        ctx = make_test_context()
        text = render(ElitesSection(2), ctx)
        @test occursin("Top 2 Programs", text)
        @test occursin("prog_a", text)
        @test occursin("prog_b", text)
        @test !occursin("prog_c", text)   # k=2, should not include 3rd
        @test occursin("0.5", text)       # fitness of first elite
        @test occursin("0.8", text)       # fitness of second elite
        println("  ElitesSection(2): includes 2 programs, not 3")
    end

    @testset "ElitesSection caps at available" begin
        ctx = make_test_context(serialized=["only_one"])
        text = render(ElitesSection(5), ctx)
        @test occursin("Top 1 Programs", text)
        @test occursin("only_one", text)
        println("  ElitesSection(5) with 1 available: caps at 1")
    end

    @testset "ElitesSection empty context" begin
        ctx = MutationContext()
        text = render(ElitesSection(3), ctx)
        @test text == ""
        println("  ElitesSection: empty context -> empty string")
    end

    @testset "GenerationSection early/mid/late" begin
        early = render(GenerationSection(), make_test_context(gen=10, max_gen=100))
        @test occursin("10/100", early)
        @test occursin("early", early)

        mid = render(GenerationSection(), make_test_context(gen=50, max_gen=100))
        @test occursin("50/100", mid)
        @test occursin("mid", mid)

        late = render(GenerationSection(), make_test_context(gen=90, max_gen=100))
        @test occursin("90/100", late)
        @test occursin("late", late)

        println("  GenerationSection: early=10/100, mid=50/100, late=90/100")
    end

    @testset "GenerationSection empty context" begin
        ctx = MutationContext()
        text = render(GenerationSection(), ctx)
        @test text == ""
        println("  GenerationSection: max_gen=0 -> empty string")
    end

    @testset "render_enrichment composition" begin
        ctx = make_test_context()
        sections = AbstractPromptSection[FitnessSection(), GenerationSection()]
        text = render_enrichment(sections, ctx)
        @test occursin("Fitness Context", text)
        @test occursin("Generation Progress", text)
        println("  render_enrichment: combines FitnessSection + GenerationSection")
    end

    @testset "render_enrichment empty sections" begin
        ctx = make_test_context()
        text = render_enrichment(AbstractPromptSection[], ctx)
        @test text == ""
        println("  render_enrichment([]): empty string")
    end

    @testset "render_enrichment skips empty sections" begin
        ctx = MutationContext()  # empty context
        sections = AbstractPromptSection[FitnessSection(), GenerationSection()]
        text = render_enrichment(sections, ctx)
        @test text == ""
        println("  render_enrichment: all sections empty -> empty string")
    end

    @testset "_find_llm_operators unwrapping" begin
        op = Arborist.LLMMutationOperator(
            endpoint="http://test", model="test", api_key_env="",
            system_prompt="test", temperature=0.5, max_tokens=100,
            timeout_seconds=1.0, fallback_op=Arborist.SubtreeMutation(),
            sections=AbstractPromptSection[])

        # Direct
        found = _find_llm_operators([op])
        @test length(found) == 1
        @test found[1] === op

        # Wrapped (simulate TrackedMutation-like wrapper)
        wrapper = (inner=op, n_calls=0)  # NamedTuple with .inner
        # _find_llm_operators checks hasproperty(:inner), so it needs an actual
        # mutable struct. Use a simple test wrapper.
        mutable struct _TestWrapper <: Arborist.AbstractMutationOperator
            inner::Arborist.LLMMutationOperator
        end
        Arborist.mutate(w::_TestWrapper, g, rng) = Arborist.mutate(w.inner, g, rng)

        wrapped = _TestWrapper(op)
        found2 = _find_llm_operators(Arborist.AbstractMutationOperator[wrapped])
        @test length(found2) == 1
        @test found2[1] === op

        println("  _find_llm_operators: finds direct and wrapped operators")
    end
end
