# Boolean benchmark generators.

"""
    multiplexer(address_bits::Int; T=Float64) -> NamedTuple

K-to-1 multiplexer truth table. For `address_bits = k`, the function
takes `k + 2^k` inputs: `k` address bits and `2^k` data bits; the
output is the data bit indexed by the address value. Full truth table
over `2^(k + 2^k)` rows is returned.

Common variants:
- `multiplexer(2)` → 6-bit (6 inputs, 2^6 = 64 rows).
- `multiplexer(3)` → 11-bit (11 inputs, 2^11 = 2048 rows).

Returns `(; X, y, n_inputs, n_rows, name, target_expr)` where
`X::Matrix{T}(n_inputs, n_rows)` and `y::Vector{T}(n_rows)` with `0/1`
values.
"""
function multiplexer(address_bits::Int; T::Type=Float64)
    address_bits >= 1 || throw(ArgumentError(
        "multiplexer: address_bits must be >= 1, got $address_bits"))
    n_data = 2^address_bits
    n_in = address_bits + n_data
    n_rows = 2^n_in

    X = Matrix{T}(undef, n_in, n_rows)
    y = Vector{T}(undef, n_rows)

    for row in 0:(n_rows - 1)
        for bit in 0:(n_in - 1)
            X[bit + 1, row + 1] = T((row >> bit) & 1)
        end
        # Address value = first `address_bits` bits; data bit index = address + address_bits + 1.
        addr = 0
        for a in 0:(address_bits - 1)
            addr |= ((row >> a) & 1) << a
        end
        data_bit_pos = address_bits + addr  # 0-indexed
        y[row + 1] = T((row >> data_bit_pos) & 1)
    end
    n_total_bits = address_bits + n_data
    return (; X=X, y=y,
              n_inputs=n_in, n_rows=n_rows,
              name="$(n_total_bits)-bit multiplexer",
              target_expr="output = data bit indexed by $(address_bits)-bit address")
end

"""
    parity(n_bits::Int; T=Float64) -> NamedTuple

`n`-bit parity truth table. Output is 1 when the number of 1-bits in
the input is odd, else 0. Full truth table over `2^n` rows.

Canonical neuroevolution difficulty ladder: 3-bit is solved by small
networks; 5-bit and higher is seed-sensitive.

Returns `(; X, y, n_inputs, n_rows, name, target_expr)`.
"""
function parity(n_bits::Int; T::Type=Float64)
    n_bits >= 1 || throw(ArgumentError(
        "parity: n_bits must be >= 1, got $n_bits"))
    n_rows = 2^n_bits
    X = Matrix{T}(undef, n_bits, n_rows)
    y = Vector{T}(undef, n_rows)
    for row in 0:(n_rows - 1)
        cnt = 0
        for b in 0:(n_bits - 1)
            bit = (row >> b) & 1
            X[b + 1, row + 1] = T(bit)
            cnt += bit
        end
        y[row + 1] = T(cnt & 1)
    end
    return (; X=X, y=y,
              n_inputs=n_bits, n_rows=n_rows,
              name="$(n_bits)-bit parity",
              target_expr="output = XOR of all input bits")
end

"""
    xor_env(; T=Float64) -> NamedTuple

2-bit XOR truth table in `GraphEvaluator`-ready shape. `input_data`
is `Matrix{T}(2, 4)`, `output_data` is `Matrix{T}(1, 4)`. Equivalent
to `parity(2)` but formatted for neuroevolution (row vectors per
feature, matrix outputs for multi-output networks).
"""
function xor_env(; T::Type=Float64)
    input_data  = T[0 0 1 1; 0 1 0 1]
    output_data = reshape(T[0, 1, 1, 0], 1, 4)
    return (; input_data=input_data, output_data=output_data,
              n_inputs=2, n_outputs=1,
              name="XOR",
              target_expr="binary XOR truth table")
end
