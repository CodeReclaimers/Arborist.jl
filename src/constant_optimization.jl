# constant_optimization.jl — configuration struct for the periodic constant-
# optimization pass. The actual optimizer (BFGS with FD gradients) and the
# TreeGenome-specific `optimize_constants!` / `_maybe_optimize_constants!`
# live in `src/tree_genome.jl` because they reference TreeGenome and
# TreeFitnessEvaluator, which are defined there.
#
# This file ships only the config struct so that `GeneticProgramming` in
# `src/algorithm.jl` (loaded before tree_genome.jl) can declare an optional
# `constant_optimization::Union{Nothing, ConstantOptimization}` field.
#
# Dependency story: DynamicExpressions' `eval_grad_tree_array` requires
# Zygote (too heavy for a required dep). `differentiable_eval_tree_array`
# returns only predictions, not gradients, despite the name. Central finite
# differences use two tree evaluations per constant per BFGS step: O(N_const)
# cost per gradient, O(BFGS_iter * N_const) per optimize pass. For typical
# SR (≤20 constants, ≤100 samples) at 25-gen frequency, top_k=5, this is
# low single-digit percent overhead on a full run.

"""
    ConstantOptimization(; frequency=25, top_k=5, max_iter=50, tol=1e-8, fd_step=1e-3)

Configuration for the periodic constant-optimization pass. Enable by passing
`GeneticProgramming(; constant_optimization=ConstantOptimization(), ...)`.

# Fields
- `frequency::Int`: generations between optimization passes (default: 25).
  The pass runs at the end of each Nth generation — gen 25, 50, 75, ...
- `top_k::Int`: number of top (lowest-fitness) individuals to optimize per
  pass (default: 5). Applying to all individuals would double evaluation cost
  every generation; applying only to elites refines the best candidates.
- `max_iter::Int`: maximum BFGS iterations per individual (default: 50).
- `tol::Float64`: gradient-norm convergence tolerance (default: 1e-8).
- `fd_step::Float64`: half-step size for central finite differences
  (default: 1e-3). Too small amplifies roundoff; too large linearizes too coarsely.
"""
struct ConstantOptimization
    frequency::Int
    top_k::Int
    max_iter::Int
    tol::Float64
    fd_step::Float64
end

function ConstantOptimization(; frequency::Int = 25,
                                top_k::Int = 5,
                                max_iter::Int = 50,
                                tol::Float64 = 1e-8,
                                fd_step::Float64 = 1e-3)
    frequency >= 1 || throw(ArgumentError("frequency must be >= 1 (got $frequency)"))
    top_k >= 1 || throw(ArgumentError("top_k must be >= 1 (got $top_k)"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1 (got $max_iter)"))
    tol > 0 || throw(ArgumentError("tol must be positive (got $tol)"))
    fd_step > 0 || throw(ArgumentError("fd_step must be positive (got $fd_step)"))
    ConstantOptimization(frequency, top_k, max_iter, tol, fd_step)
end
