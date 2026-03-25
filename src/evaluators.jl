"""
    TableFitnessEvaluator <: AbstractEvaluator

Evaluates a function against a table of input/output examples.
Fitness is mean squared error over rows where execution succeeded.
Returns `Inf` if more than 50% of rows throw exceptions or exceed the time limit.

# Fields
- `input_cols::Dict{Symbol, DataType}`: input variable names and types
- `output_cols::Dict{Symbol, DataType}`: output variable names and types
- `input_rows::Vector{Dict{Symbol, Any}}`: input data rows
- `output_rows::Vector{Dict{Symbol, Any}}`: expected output data rows
- `time_limit_ns::Int`: per-call time limit in nanoseconds (default: 1,000,000)
"""
struct TableFitnessEvaluator <: AbstractEvaluator
    input_cols::Dict{Symbol, DataType}
    output_cols::Dict{Symbol, DataType}
    input_rows::Vector{Dict{Symbol, Any}}
    output_rows::Vector{Dict{Symbol, Any}}
    time_limit_ns::Int
end

"""
    TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows; time_limit_ns=1_000_000)

Construct a `TableFitnessEvaluator` with an optional time limit per function call.
"""
function TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                time_limit_ns::Int=1_000_000)
    TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows, time_limit_ns)
end

"""
    input_signature(fe::TableFitnessEvaluator) -> Dict{Symbol, DataType}

Return the input variable names and types expected by this evaluator.
"""
input_signature(fe::TableFitnessEvaluator) = fe.input_cols

"""
    output_signature(fe::TableFitnessEvaluator) -> Dict{Symbol, DataType}

Return the output variable names and types expected by this evaluator.
"""
output_signature(fe::TableFitnessEvaluator) = fe.output_cols

"""
    evaluate(fe::TableFitnessEvaluator, f::Function) -> Float64

Evaluate `f` against the table of examples. Returns mean squared error
for rows where execution succeeded. Returns `Inf` if the function fails
on more than 50% of rows or exceeds the per-call time limit.

Uses `Base.invokelatest` to handle world-age issues from `@eval`-defined functions.
"""
function evaluate(fe::TableFitnessEvaluator, f::Function)
    n_rows = length(fe.input_rows)
    n_errors = 0
    total_se = 0.0

    sorted_inputs = sort(collect(fe.input_cols), by=first)
    sorted_outputs = sort(collect(fe.output_cols), by=first)
    n_outputs = length(sorted_outputs)

    for (in_row, out_row) in zip(fe.input_rows, fe.output_rows)
        t0 = time_ns()
        result = try
            args = [in_row[name] for (name, _) in sorted_inputs]
            Base.invokelatest(f, args...)
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        elapsed_ns = time_ns() - t0

        if result === nothing || elapsed_ns > fe.time_limit_ns
            n_errors += 1
            continue
        end

        se = 0.0
        try
            if n_outputs == 1
                expected = out_row[sorted_outputs[1][1]]
                se = (Float64(result) - Float64(expected))^2
            else
                for (k, (name, _)) in enumerate(sorted_outputs)
                    expected = out_row[name]
                    se += (Float64(result[k]) - Float64(expected))^2
                end
            end
        catch e
            e isa InterruptException && rethrow()
            n_errors += 1
            continue
        end

        if !isfinite(se)
            n_errors += 1
            continue
        end

        total_se += se
    end

    if n_errors > n_rows / 2
        return Inf
    end

    n_success = n_rows - n_errors
    if n_success == 0
        return Inf
    end

    return total_se / n_success
end
