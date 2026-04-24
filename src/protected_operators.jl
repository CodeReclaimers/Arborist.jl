# Protected arithmetic / unary operators for symbolic regression.
#
# Koza-canonical "protected" versions of division, logarithm, square root,
# exponential, and inverse. Each guards against domain violations
# (divide-by-zero, log of a non-positive, sqrt of a negative, exp overflow)
# that would otherwise produce `NaN` or `Inf` and terminate a fitness
# evaluation early. Using these in a GP function set keeps evolved programs
# evaluable over the full input domain without hand-written exception
# handling in every evaluator.

const PROTECTED_EPS = 1e-10

"""
    pdiv(a, b)

Protected division. Returns `one(a)` when `|b| < $(PROTECTED_EPS)`;
otherwise `a / b`. The canonical Koza-style guard against division by zero.
"""
pdiv(a, b) = abs(b) < PROTECTED_EPS ? one(a) : a / b

"""
    plog(x)

Protected natural logarithm, `log(|x| + $(PROTECTED_EPS))`. Always finite
and real-valued; tracks `log|x|` away from zero and saturates at
`log(PROTECTED_EPS)` near zero.
"""
plog(x) = log(abs(x) + PROTECTED_EPS)

"""
    psqrt(x)

Protected square root, `sqrt(|x|)`. Always finite and real-valued.
"""
psqrt(x) = sqrt(abs(x))

"""
    pexp(x)

Protected exponential, `exp(clamp(x, -50, 50))`. Prevents `Inf` overflow
for large positive `x` while preserving finite behavior everywhere else.
The clamp bounds correspond to `exp(50) ≈ 5.18e21`, comfortably within
`Float64` range.
"""
pexp(x) = exp(clamp(x, -50.0, 50.0))

"""
    pinv(x)

Protected multiplicative inverse. Returns `zero(x)` when
`|x| < $(PROTECTED_EPS)`, otherwise `one(x) / x`.
"""
pinv(x) = abs(x) < PROTECTED_EPS ? zero(x) : one(x) / x

"""
    default_protected_function_set() -> FunctionSet

Return a symbolic-regression `FunctionSet` built around the protected
operators. Suitable for `ExprGenome`-based symbolic regression where
evolved programs must evaluate without raising domain errors.

Contents:
- Binary arithmetic: `+`, `-`, `*` for `Float32` and `Int32`; `pdiv` for `Float32`.
- Unary transcendentals: `plog`, `psqrt`, `pexp`, `sin`, `cos` for `Float32`.

The set follows the Nguyen/Keijzer convention used in the modern symbolic
regression literature (McDermott et al., 2012). `pinv` is not included by
default — use it as a drop-in replacement for `pdiv(1.0, x)` problems where
an explicit inverse primitive is desired.

`TreeGenome` users do not need this helper: pass the raw functions directly
to `DynamicExpressions.OperatorEnum`, e.g.
`OperatorEnum(; binary_operators=[+, -, *, pdiv], unary_operators=[plog, psqrt, pexp, sin, cos])`.
"""
function default_protected_function_set()
    fset = FunctionSet(Set{FunctionDetails}())
    for T in [Float32, Int32]
        for func in [:+, :-, :*]
            add!(fset, func, 2, T, T)
        end
    end
    add!(fset, :pdiv, 2, Float32, Float32)
    for func in [:plog, :psqrt, :pexp, :sin, :cos]
        add!(fset, func, 1, Float32, Float32)
    end
    return fset
end
