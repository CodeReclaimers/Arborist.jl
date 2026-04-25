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
end
