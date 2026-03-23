"""
    default_function_set() -> FunctionSet

Return a default function set containing basic arithmetic, transcendental,
and comparison operators suitable for numerical symbolic regression.

Includes:
- Binary arithmetic (`+`, `-`, `*`, `/`, `^`) for `Float32` and `Int32`
- Unary transcendentals (`cos`, `sin`, `tanh`, `exp`, `sign`) for `Float32`
- Binary comparisons (`>`, `<`, `==`, `!=`, `>=`, `<=`) for `Float32` and `Int32`,
  returning `Bool`
"""
function default_function_set()
    fset = FunctionSet(Set{FunctionDetails}())
    for T in [Float32, Int32]
        for func in [:+, :-, :*, :/, :^]
            add!(fset, func, 2, T, T)
        end
    end
    for func in [:cos, :sin, :tanh, :exp, :sign]
        add!(fset, func, 1, Float32, Float32)
    end
    for T in [Float32, Int32]
        for op in [:>, :<, :(==), :!=, :>=, :<=]
            add!(fset, op, 2, T, Bool)
        end
    end
    return fset
end

"""
    boolean_function_set() -> FunctionSet

Return a function set containing boolean operators suitable for boolean
GP problems (e.g., even parity).

Includes: AND (`&`), OR (`|`), NOT (`!`), NAND (`gp_nand`), NOR (`gp_nor`),
XOR (`xor`), all operating on `Bool`.
"""
function boolean_function_set()
    fset = FunctionSet(Set{FunctionDetails}())
    add!(fset, :&, 2, Bool, Bool)        # AND
    add!(fset, :|, 2, Bool, Bool)        # OR
    add!(fset, :!, 1, Bool, Bool)        # NOT
    add!(fset, :gp_nand, 2, Bool, Bool)  # NAND
    add!(fset, :gp_nor, 2, Bool, Bool)   # NOR
    add!(fset, :xor, 2, Bool, Bool)      # XOR
    return fset
end

# --- Boolean operator definitions for use in @eval'd evolved programs ---
# These are defined in the Arborist module so they resolve when evolved
# code is compiled via @eval.

"""
    gp_nand(a::Bool, b::Bool) -> Bool

Boolean NAND: returns `!(a & b)`.
"""
gp_nand(a::Bool, b::Bool) = !(a & b)

"""
    gp_nor(a::Bool, b::Bool) -> Bool

Boolean NOR: returns `!(a | b)`.
"""
gp_nor(a::Bool, b::Bool) = !(a | b)
