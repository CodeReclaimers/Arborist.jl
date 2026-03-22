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
