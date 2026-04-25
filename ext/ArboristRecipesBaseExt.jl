# ArboristRecipesBaseExt.jl — Plots.jl recipes via RecipesBase weakdep.
#
# Activated automatically when both Arborist and a RecipesBase consumer
# (Plots.jl, Makie via RecipesBaseSupport, etc.) are loaded. RecipesBase
# is the lightweight (no transitive deps beyond stdlib) home for `@recipe`,
# so this extension imposes no runtime cost on users who don't load Plots.

module ArboristRecipesBaseExt

using RecipesBase
using Arborist: GPResult, NSGAIIResult, RunLog, MAPElitesResult,
                MAPElitesArchive, GenerationLog, entries, coverage, qd_score
import Arborist: plothypervolumetrajectory, plothypervolumetrajectory!,
                 plotarchive, plotarchive!

# ---------------------------------------------------------------------------
# Helper: cap a vector of values at `median + k * MAD`, replacing values
# above the cap with NaN (so Plots draws a gap rather than a spike).
#
# Float32 protected arithmetic in TreeGenome / ExprGenome can produce very
# large but finite mean values when a single pathological tree dominates
# the population's average. Linear y-axis scaling on those points hides
# every other generation's data. Clipping at median + k·MAD is the same
# robust-statistics approach epsilon-lexicase uses for its per-case
# tolerance band, and surfaces the typical trajectory without losing the
# information that "an outlier occurred at generation N" — the gap is
# visible in the plot.
# ---------------------------------------------------------------------------
function _clip_outliers(xs::AbstractVector{<:Real}; k::Real=10.0)
    finite = [Float64(v) for v in xs if isfinite(v)]
    n = length(finite)
    n < 4 && return [isfinite(v) ? Float64(v) : NaN for v in xs]
    sorted_vals = sort(finite)
    med = sorted_vals[div(n + 1, 2)]
    mad = let abs_devs = sort([abs(v - med) for v in finite])
        abs_devs[div(n + 1, 2)]
    end
    # Floor cap so flat data (mad ≈ 0) doesn't get clipped to median.
    cap = max(med + k * mad, 2.0 * abs(med) + 1e-12)
    return [(isfinite(v) && v <= cap) ? Float64(v) : NaN for v in xs]
end

# ---------------------------------------------------------------------------
# Single-objective fitness trajectory.
# ---------------------------------------------------------------------------

@recipe function f(r::GPResult)
    title  --> "GP fitness trajectory"
    xlabel --> "Generation"
    ylabel --> "Fitness (lower = better)"
    legend --> :topright

    n = length(r.fitness_history)
    xs = collect(1:n)

    @series begin
        label --> "best"
        linewidth --> 2
        xs, r.fitness_history
    end

    @series begin
        label --> "mean (clipped)"
        linestyle --> :dash
        seriescolor --> :gray
        xs, _clip_outliers(r.mean_history)
    end
end

# ---------------------------------------------------------------------------
# NSGA-II Pareto front (2D supported; 3+ → parallel-coordinates not yet implemented).
# ---------------------------------------------------------------------------

@recipe function f(r::NSGAIIResult)
    front_fits = r.pareto_fitnesses
    isempty(front_fits) && error(
        "NSGAIIResult has empty pareto_fitnesses; nothing to plot.")
    n_obj = length(first(front_fits))

    if n_obj == 2
        seriestype --> :scatter
        title --> "NSGA-II Pareto front (rank 1)"
        xlabel --> "Objective 1"
        ylabel --> "Objective 2"
        legend --> false
        xs = [f[1] for f in front_fits]
        ys = [f[2] for f in front_fits]
        # Sort for nicer connecting line if requested.
        order = sortperm(xs)
        xs[order], ys[order]
    elseif n_obj == 3
        seriestype --> :scatter
        title --> "NSGA-II Pareto front (rank 1)"
        xlabel --> "Objective 1"
        ylabel --> "Objective 2"
        zlabel --> "Objective 3"
        legend --> false
        xs = [f[1] for f in front_fits]
        ys = [f[2] for f in front_fits]
        zs = [f[3] for f in front_fits]
        xs, ys, zs
    else
        error("NSGAIIResult plotting only supports 2 or 3 objectives directly. " *
              "For higher dimensions, use parallel-coordinates manually with " *
              "`r.pareto_fitnesses`.")
    end
end

# ---------------------------------------------------------------------------
# Hypervolume convergence — auxiliary recipe accessed via `plot_hypervolume`.
# (Default `plot(::NSGAIIResult)` shows the front; this is the trajectory.)
# ---------------------------------------------------------------------------

"""
    plothypervolumetrajectory(r::NSGAIIResult; kwargs...)

Plot the hypervolume trajectory across generations. Activates only when
RecipesBase is loaded; the function itself is forward-declared in
`Arborist` so user code that does `using Arborist, Plots` resolves the
name in the `Arborist` namespace.
"""
mutable struct PlotHyperVolumeTrajectory
    args::Tuple
end
plothypervolumetrajectory(args...; kw...) =
    RecipesBase.plot(PlotHyperVolumeTrajectory(args); kw...)
plothypervolumetrajectory!(args...; kw...) =
    RecipesBase.plot!(PlotHyperVolumeTrajectory(args); kw...)
@recipe function f(h::PlotHyperVolumeTrajectory)
    r = h.args[1]::NSGAIIResult
    title  --> "NSGA-II hypervolume convergence"
    xlabel --> "Generation"
    ylabel --> "Hypervolume (log scale)"
    legend --> false
    # The hypervolume can span several orders of magnitude (the reference
    # point shrinks rapidly as the population improves), so a log scale
    # is the natural default. Override with `yscale=:identity` to see
    # the linear view. Replace any nonpositive HV with NaN so log10
    # never gets a zero / negative input.
    yscale --> :log10
    ys = [v > 0 ? Float64(v) : NaN for v in r.hypervolume_history]
    1:length(ys), ys
end

# ---------------------------------------------------------------------------
# RunLog: per-generation metrics.
# ---------------------------------------------------------------------------

@recipe function f(log::RunLog)
    title  --> "RunLog summary"
    xlabel --> "Generation"
    legend --> :topright
    layout --> (3, 1)

    es = entries(log)
    xs = [e.generation for e in es]

    @series begin
        subplot := 1
        ylabel := "Fitness"
        label --> "best"
        xs, [e.best_fitness for e in es]
    end
    @series begin
        subplot := 1
        label --> "mean (clipped)"
        linestyle --> :dash
        # Same outlier-clipping as the GPResult fitness recipe — Float32
        # protected arithmetic can produce extreme single-generation means.
        xs, _clip_outliers([e.mean_fitness for e in es])
    end
    @series begin
        subplot := 2
        ylabel := "Species"
        label --> "n_species"
        # Step plot reads better than `:sticks` over many generations and
        # is informative even when speciation is disabled (a flat line at 1).
        seriestype --> :steppost
        xs, [e.n_species for e in es]
    end
    @series begin
        subplot := 3
        ylabel := "Unique structures"
        label --> "unique"
        xs, [e.unique_structures for e in es]
    end
end

# ---------------------------------------------------------------------------
# MAP-Elites: coverage + QD-score history, and 2D archive heatmap.
# ---------------------------------------------------------------------------

@recipe function f(r::MAPElitesResult)
    title  --> "MAP-Elites coverage / QD-score"
    xlabel --> "Generation"
    legend --> :topright
    layout --> (2, 1)

    n = length(r.coverage_history)
    xs = collect(1:n)
    @series begin
        subplot := 1
        ylabel := "Coverage"
        label --> "coverage"
        linewidth --> 2
        xs, r.coverage_history
    end
    @series begin
        subplot := 2
        ylabel := "QD score"
        label --> "qd_score"
        linewidth --> 2
        xs, r.qd_score_history
    end
end

"""
    plotarchive(archive::MAPElitesArchive; kwargs...)

Plot a 2D MAP-Elites archive as a heatmap of best-cell fitness. For 1D
archives, plots a bar chart. Higher-dimensional archives raise.
Activates only when RecipesBase is loaded; the function itself is
forward-declared in `Arborist`.
"""
mutable struct PlotArchive
    args::Tuple
end
plotarchive(args...; kw...) = RecipesBase.plot(PlotArchive(args); kw...)
plotarchive!(args...; kw...) = RecipesBase.plot!(PlotArchive(args); kw...)
@recipe function f(p::PlotArchive)
    archive = p.args[1]::MAPElitesArchive
    n_dims = archive.n_dims
    n_bins = archive.n_bins

    if n_dims == 2
        # Build a grid; missing cells get NaN.
        grid = fill(NaN, n_bins[1], n_bins[2])
        for (key, pair) in archive.grid
            grid[key[1], key[2]] = pair.second
        end
        seriestype --> :heatmap
        title --> "MAP-Elites archive (best fitness per cell)"
        xlabel --> "Feature 1 bin"
        ylabel --> "Feature 2 bin"
        1:n_bins[1], 1:n_bins[2], grid
    elseif n_dims == 1
        bars = fill(NaN, n_bins[1])
        for (key, pair) in archive.grid
            bars[key[1]] = pair.second
        end
        seriestype --> :bar
        title --> "MAP-Elites archive (best fitness per cell)"
        xlabel --> "Feature 1 bin"
        ylabel --> "Best fitness"
        legend --> false
        1:n_bins[1], bars
    else
        error("plot_archive supports 1D or 2D archives directly (got $n_dims dims). " *
              "For 3+ dims, slice or marginalize before plotting.")
    end
end

end # module
