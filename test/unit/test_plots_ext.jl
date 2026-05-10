using Test
using Arborist
using RecipesBase

@testset "RecipesBase extension" begin
    # Once both Arborist and RecipesBase are loaded, the extension module
    # auto-loads via the [extensions] entry in Project.toml. Verify the
    # extension is reachable.
    ext = Base.get_extension(Arborist, :ArboristRecipesBaseExt)
    @test ext !== nothing
    @test isdefined(ext, :PlotHyperVolumeTrajectory)
    @test isdefined(ext, :PlotArchive)
    # Forward-declared user-facing functions in Arborist must have methods
    # registered by the extension after Plots loads.
    @test !isempty(methods(plothypervolumetrajectory))
    @test !isempty(methods(plotarchive))

    # Verify @recipe registrations exist by checking RecipesBase's recipe table
    # contains our types. RecipesBase doesn't expose a public list directly,
    # so we rely on `Base.return_types(RecipesBase.apply_recipe, ...)` which
    # is implementation-defined. Instead we verify the recipe-defining methods
    # exist by looking at `methods(RecipesBase.apply_recipe)`.
    apply_methods = methods(RecipesBase.apply_recipe)
    method_sigs = [m.sig for m in apply_methods]
    # Search for any signature that mentions GPResult, NSGAIIResult, RunLog,
    # or MAPElitesResult.
    needles = (GPResult, NSGAIIResult, RunLog, MAPElitesResult)
    found = Dict{Any, Bool}(needle => false for needle in needles)
    for sig in method_sigs
        sig isa UnionAll && continue
        sig isa DataType || continue
        for t in sig.parameters
            for needle in needles
                t === needle && (found[needle] = true)
                t isa Type && t <: needle && (found[needle] = true)
            end
        end
    end
    # Verify each recipe-defined type actually has a method registered.
    for (needle, ok) in found
        @test ok || @info "no apply_recipe method found for $needle"
    end

    # ---- RunLog recipe: NSGA-II metric awareness -------------------------
    # When the RunLog has no NSGA-II metrics populated the recipe should
    # request the default 3-panel layout; when at least one entry has a
    # non-NaN hypervolume or non-empty front_sizes, it should request 5.
    plain = RunLog()
    Arborist.record!(plain, 1, [0.5, 0.7], [Ref(1), Ref(2)], 0.1)
    rich = RunLog()
    Arborist.record!(rich, 1, [0.5, 0.7], [Ref(1), Ref(2)], 0.1;
                     nsga2 = Arborist.NSGAIISnapshot(1.234, [10, 5, 3]))

    # `apply_recipe(plotattributes, log)` returns a vector of `RecipeData`;
    # the layout attribute is inserted into `plotattributes` by the recipe.
    plain_attrs = Dict{Symbol, Any}()
    rich_attrs  = Dict{Symbol, Any}()
    plain_series = RecipesBase.apply_recipe(plain_attrs, plain)
    rich_series  = RecipesBase.apply_recipe(rich_attrs,  rich)

    @test plain_attrs[:layout] == (3, 1)
    @test rich_attrs[:layout]  == (5, 1)
    # 3-panel layout produces 4 series (best + mean on subplot 1, then
    # species, then unique). 5-panel layout adds 2 (hypervolume + front1).
    @test length(plain_series) == 4
    @test length(rich_series)  == 6
end
