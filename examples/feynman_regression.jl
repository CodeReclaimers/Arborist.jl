#!/usr/bin/env julia
# examples/feynman_regression.jl — Feynman Symbolic Regression Benchmark
#
# Runs Arborist's TreeGenome (default) or ExprGenome against equations from
# the AI Feynman benchmark
# (Udrescu & Tegmark, 2019/2020).  Data is downloaded from PMLB on first run
# and cached locally.
#
# Six difficulty levels:
#   smoke          5 trivial equations (1-2 variables, simple products/ratios)
#   standard      12 moderate equations (2-5 variables, trig, sqrt, Lorentz)
#   bonus          8 hard equations from the AI Feynman 2.0 bonus set
#   full_standard all 99 standard Feynman equations
#   full_bonus    all 20 bonus equations
#   everything    all 119 equations
#
# Usage:
#   julia --project examples/feynman_regression.jl
#   julia --project examples/feynman_regression.jl --level=standard
#   julia --project examples/feynman_regression.jl --level=everything --seeds=5 --generations=500
#
# Options:
#   --level=LEVEL       Benchmark level (default: smoke)
#   --seed=N            Base RNG seed (default: 42)
#   --seeds=N           Number of independent seeds per equation (default: 3)
#   --generations=N     Generations per run (default: 300)
#   --pop_size=N        Population size (default: 200)
#   --n_train=N         Training points subsampled from PMLB data (default: 200)
#   --n_test=N          Test points held out for evaluation (default: 100)
#   --operators=SET     Operator set: "physics" or "arithmetic" (default: physics)
#   --genome=TYPE       Genome type: "tree" or "expr" (default: tree)

using Arborist
using DynamicExpressions
using Downloads
using Random

# =============================================================================
# Equation catalog
# =============================================================================

struct FeynmanEquation
    name::String       # PMLB dataset name
    formula::String    # human-readable formula
    n_vars::Int        # number of input variables
end

# --- Curated subsets with formulas for display ---

const SMOKE_EQUATIONS = [
    FeynmanEquation("feynman_I_12_1",  "F = μ·Nn",             2),
    FeynmanEquation("feynman_I_14_4",  "U = ½·k·x²",          2),
    FeynmanEquation("feynman_I_25_13", "V = q/C",              2),
    FeynmanEquation("feynman_I_29_4",  "k = ω/c",              2),
    FeynmanEquation("feynman_I_39_1",  "E = (3/2)·p·V",       2),
]

const STANDARD_EQUATIONS = [
    FeynmanEquation("feynman_I_10_7",   "m = m₀/√(1 - v²/c²)",                          3),
    FeynmanEquation("feynman_I_14_3",   "U = m·g·z",                                     3),
    FeynmanEquation("feynman_I_18_12",  "τ = r·F·sin(θ)",                                3),
    FeynmanEquation("feynman_I_26_2",   "θ₁ = arcsin(n·sin(θ₂))",                        2),
    FeynmanEquation("feynman_I_37_4",   "I = I₁+I₂+2√(I₁I₂)·cos(δ)",                    3),
    FeynmanEquation("feynman_I_39_22",  "p = n·kB·T/V",                                  4),
    FeynmanEquation("feynman_I_47_23",  "c = √(γ·p/ρ)",                                  3),
    FeynmanEquation("feynman_I_48_2",   "E = m·c²/√(1 - v²/c²)",                         3),
    FeynmanEquation("feynman_II_4_23",  "V = q/(4πε·r)",                                 3),
    FeynmanEquation("feynman_II_8_31",  "u = ε·E²/2",                                    2),
    FeynmanEquation("feynman_II_15_4",  "E = -μ·B·cos(θ)",                               3),
    FeynmanEquation("feynman_II_35_21", "M = nρ·μ·tanh(μ·B/(kB·T))",                     5),
]

const BONUS_EQUATIONS = [
    FeynmanEquation("feynman_test_3",  "r = d(1-α²)/(1+α·cos(θ₁-θ₂))",                  4),
    FeynmanEquation("feynman_test_5",  "T = 2π·d^(3/2)/√(G(m₁+m₂))",                    4),
    FeynmanEquation("feynman_test_8",  "K = E/(1+E/(mc²)·(1-cos θ))",                    4),
    FeynmanEquation("feynman_test_10", "θ₁ = arccos((cos θ₂-v/c)/(1-v/c·cos θ₂))",      3),
    FeynmanEquation("feynman_test_11", "I = I₀(sin(α/2)sin(nδ/2)/(α/2·sin(δ/2)))²",     4),
    FeynmanEquation("feynman_test_13", "V = q/(4πε)·1/√(r²+d²-2rd·cos α)",              5),
    FeynmanEquation("feynman_test_15", "ω₀ = √(1-v²/c²)·ω/(1+v/c·cos θ)",              4),
    FeynmanEquation("feynman_test_20", "Klein-Nishina cross-section",                     7),
]

# --- Full lists (PMLB dataset names only, formulas fetched from metadata) ---

const ALL_STANDARD_NAMES = [
    # Volume I (51 equations)
    "feynman_I_6_2",    "feynman_I_6_2a",   "feynman_I_6_2b",
    "feynman_I_8_14",   "feynman_I_9_18",   "feynman_I_10_7",
    "feynman_I_11_19",  "feynman_I_12_1",   "feynman_I_12_2",
    "feynman_I_12_4",   "feynman_I_12_5",   "feynman_I_12_11",
    "feynman_I_13_4",   "feynman_I_13_12",  "feynman_I_14_3",
    "feynman_I_14_4",   "feynman_I_15_3t",  "feynman_I_15_3x",
    "feynman_I_15_10",  "feynman_I_16_6",   "feynman_I_18_4",
    "feynman_I_18_12",  "feynman_I_18_14",  "feynman_I_24_6",
    "feynman_I_25_13",  "feynman_I_26_2",   "feynman_I_27_6",
    "feynman_I_29_4",   "feynman_I_29_16",  "feynman_I_30_3",
    "feynman_I_30_5",   "feynman_I_32_5",   "feynman_I_32_17",
    "feynman_I_34_1",   "feynman_I_34_8",   "feynman_I_34_14",
    "feynman_I_34_27",  "feynman_I_37_4",   "feynman_I_38_12",
    "feynman_I_39_1",   "feynman_I_39_11",  "feynman_I_39_22",
    "feynman_I_40_1",   "feynman_I_41_16",  "feynman_I_43_16",
    "feynman_I_43_31",  "feynman_I_43_43",  "feynman_I_44_4",
    "feynman_I_47_23",  "feynman_I_48_2",   "feynman_I_50_26",
    # Volume II (33 equations)
    "feynman_II_2_42",  "feynman_II_3_24",  "feynman_II_4_23",
    "feynman_II_6_11",  "feynman_II_6_15a", "feynman_II_6_15b",
    "feynman_II_8_7",   "feynman_II_8_31",  "feynman_II_10_9",
    "feynman_II_11_3",  "feynman_II_11_20", "feynman_II_11_27",
    "feynman_II_11_28", "feynman_II_13_17", "feynman_II_13_23",
    "feynman_II_13_34", "feynman_II_15_4",  "feynman_II_15_5",
    "feynman_II_21_32", "feynman_II_24_17", "feynman_II_27_16",
    "feynman_II_27_18", "feynman_II_34_2",  "feynman_II_34_2a",
    "feynman_II_34_11", "feynman_II_34_29a","feynman_II_34_29b",
    "feynman_II_35_18", "feynman_II_35_21", "feynman_II_36_38",
    "feynman_II_37_1",  "feynman_II_38_3",  "feynman_II_38_14",
    # Volume III (15 equations)
    "feynman_III_4_32",  "feynman_III_4_33",  "feynman_III_7_38",
    "feynman_III_8_54",  "feynman_III_9_52",  "feynman_III_10_19",
    "feynman_III_12_43", "feynman_III_13_18", "feynman_III_14_14",
    "feynman_III_15_12", "feynman_III_15_14", "feynman_III_15_27",
    "feynman_III_17_37", "feynman_III_19_51", "feynman_III_21_20",
]

const ALL_BONUS_NAMES = [
    "feynman_test_$i" for i in 1:20
]

# =============================================================================
# Data download and caching
# =============================================================================

const CACHE_DIR = joinpath(@__DIR__, ".feynman_cache")
const PMLB_DATA_URL = "https://github.com/EpistasisLab/pmlb/raw/master/datasets"
const PMLB_META_URL = "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets"

function ensure_gzip()
    try
        run(pipeline(`gzip --version`, stdout=devnull, stderr=devnull))
    catch
        error("gzip not found on PATH. Install gzip to decompress PMLB datasets.")
    end
end

function download_dataset(name::String)
    mkpath(CACHE_DIR)
    tsv_path = joinpath(CACHE_DIR, "$name.tsv")
    isfile(tsv_path) && return tsv_path

    gz_path = joinpath(CACHE_DIR, "$name.tsv.gz")
    url = "$PMLB_DATA_URL/$name/$name.tsv.gz"
    println("  Downloading $name ...")
    flush(stdout)
    try
        Downloads.download(url, gz_path)
    catch e
        error("Failed to download $name from PMLB: $e\n  URL: $url")
    end
    run(pipeline(`gzip -dc $gz_path`, stdout=tsv_path))
    rm(gz_path)
    return tsv_path
end

function fetch_formula_from_metadata(name::String)
    url = "$PMLB_META_URL/$name/metadata.yaml"
    try
        content = sprint() do io
            Downloads.download(url, io)
        end
        for line in split(content, "\n")
            stripped = strip(line)
            # The description field often contains the formula directly
            if occursin("description:", stripped)
                # Single-line description
                m = match(r"description:\s*(.+)", stripped)
                m !== nothing && return String(m[1])
            end
            # Multi-line description: formula is usually on the next non-empty line
            if startswith(stripped, "target:") || startswith(stripped, "dataset:")
                continue
            end
            # Look for lines that look like equations (contain = and variable names)
            if occursin("=", stripped) && !startswith(stripped, "#") &&
               !occursin(":", stripped)
                return String(stripped)
            end
        end
    catch
        # Metadata fetch is best-effort
    end
    return ""
end

function parse_tsv(path::String)
    lines = readlines(path)
    isempty(lines) && error("Empty TSV file: $path")

    header = split(lines[1], '\t')
    n_cols = length(header)
    n_rows = length(lines) - 1

    data = Matrix{Float64}(undef, n_cols, n_rows)
    for (i, line) in enumerate(lines[2:end])
        vals = split(line, '\t')
        for (j, v) in enumerate(vals)
            data[j, i] = parse(Float64, v)
        end
    end

    # Last column is always "target" in PMLB Feynman datasets
    n_features = n_cols - 1
    X = Float32.(data[1:n_features, :])   # n_features × n_samples
    y = Float32.(data[n_cols, :])          # n_samples
    col_names = String.(header[1:n_features])
    return X, y, col_names
end

function load_dataset(name::String; n_train::Int=200, n_test::Int=100, data_seed::Int=0)
    tsv_path = download_dataset(name)
    X_full, y_full, col_names = parse_tsv(tsv_path)
    n_total = size(X_full, 2)
    n_needed = n_train + n_test

    # Deterministic shuffle and subsample
    rng = Random.MersenneTwister(data_seed)
    perm = randperm(rng, n_total)

    if n_needed > n_total
        @warn "Dataset $name has only $n_total points, using all for train and test"
        n_train = div(n_total * 4, 5)
        n_test = n_total - n_train
    end

    train_idx = perm[1:n_train]
    test_idx  = perm[n_train+1:n_train+n_test]

    X_train = X_full[:, train_idx]
    y_train = y_full[train_idx]
    X_test  = X_full[:, test_idx]
    y_test  = y_full[test_idx]

    return (; X_train, y_train, X_test, y_test, col_names,
              n_features=size(X_full, 1), n_total)
end

# =============================================================================
# Fit statistics
# =============================================================================

function compute_r_squared(predictions, targets)
    n = length(targets)
    ss_res = sum((predictions .- targets) .^ 2)
    mean_y = sum(targets) / n
    ss_tot = sum((targets .- mean_y) .^ 2)
    return ss_tot > 0 ? 1.0 - ss_res / ss_tot : (ss_res ≈ 0.0 ? 1.0 : NaN)
end

function fit_stats(predictions::Vector{Float32}, targets::Vector{Float32})
    n = length(targets)
    residuals = predictions .- targets
    mse = sum(residuals .^ 2) / n
    rmse = sqrt(mse)
    r_squared = compute_r_squared(predictions, targets)
    max_err = maximum(abs.(residuals))
    return (; mse, rmse, r_squared, max_err)
end

# =============================================================================
# Operator sets
# =============================================================================

function make_operators(set_name::String)
    if set_name == "arithmetic"
        return OperatorEnum(; binary_operators=[+, -, *, /],
                              unary_operators=[abs])
    elseif set_name == "physics"
        return OperatorEnum(; binary_operators=[+, -, *, /],
                              unary_operators=[sin, cos, exp, abs])
    else
        error("Unknown operator set: $set_name (choose 'physics' or 'arithmetic')")
    end
end

# =============================================================================
# ExprGenome support
# =============================================================================

function make_function_set(set_name::String)
    fset = FunctionSet(Set{FunctionDetails}())
    for func in [:+, :-, :*, :/]
        add!(fset, func, 2, Float32, Float32)
    end
    if set_name == "physics"
        for func in [:sin, :cos, :exp, :abs]
            add!(fset, func, 1, Float32, Float32)
        end
    elseif set_name == "arithmetic"
        add!(fset, :abs, 1, Float32, Float32)
    end
    return fset
end

function make_table_evaluator(X_train::Matrix{Float32}, y_train::Vector{Float32},
                              col_names::Vector{String})
    n_features = size(X_train, 1)
    n_samples = size(X_train, 2)
    input_cols = Dict(Symbol(col_names[j]) => Float32 for j in 1:n_features)
    output_cols = Dict(:target => Float32)
    input_rows = [Dict{Symbol,Any}(Symbol(col_names[j]) => X_train[j, i]
                                    for j in 1:n_features)
                  for i in 1:n_samples]
    output_rows = [Dict{Symbol,Any}(:target => y_train[i]) for i in 1:n_samples]
    return TableFitnessEvaluator(input_cols, output_cols, input_rows, output_rows;
                                  time_limit_ns=1_000_000_000)
end

function expr_predict(genome::ExprGenome, X::Matrix{Float32}, col_names::Vector{String})
    n_features = size(X, 1)
    n_samples = size(X, 2)
    fname = gensym("predict")
    checked_body = Arborist.add_loop_checks(genome.body)
    harness = Arborist.create_harness(genome.state, checked_body, fname)
    f = @eval $harness

    # The harness sorts parameters alphabetically by name, so we need to
    # map from the sorted parameter order back to X row indices.
    sorted_names = sort(col_names)
    col_to_row = [findfirst(==(name), col_names) for name in sorted_names]

    preds = Vector{Float32}(undef, n_samples)
    for i in 1:n_samples
        args = [X[col_to_row[j], i] for j in 1:n_features]
        result = try
            Base.invokelatest(f, args...)
        catch
            Float32(NaN)
        end
        preds[i] = result === nothing ? Float32(NaN) : Float32(result)
    end
    return preds
end

# =============================================================================
# Build equation list for a given level
# =============================================================================

function equations_for_level(level::String)
    if level == "smoke"
        return SMOKE_EQUATIONS
    elseif level == "standard"
        return STANDARD_EQUATIONS
    elseif level == "bonus"
        return BONUS_EQUATIONS
    elseif level == "full_standard"
        return [FeynmanEquation(n, "", 0) for n in ALL_STANDARD_NAMES]
    elseif level == "full_bonus"
        return [FeynmanEquation(n, "", 0) for n in ALL_BONUS_NAMES]
    elseif level == "everything"
        return [FeynmanEquation(n, "", 0) for n in vcat(ALL_STANDARD_NAMES, ALL_BONUS_NAMES)]
    else
        error("Unknown level: $level\n  Choose: smoke, standard, bonus, full_standard, full_bonus, everything")
    end
end

# =============================================================================
# CLI argument parsing
# =============================================================================

level       = "smoke"
base_seed   = 42
n_seeds     = 3
generations = 300
pop_size    = 200
n_train     = 200
n_test      = 100
op_set      = "physics"
genome_type = "tree"

for arg in ARGS
    if startswith(arg, "--level=")
        global level = String(split(arg, "=")[2])
    elseif startswith(arg, "--seed=")
        global base_seed = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--seeds=")
        global n_seeds = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--generations=")
        global generations = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--pop_size=")
        global pop_size = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--n_train=")
        global n_train = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--n_test=")
        global n_test = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--operators=")
        global op_set = String(split(arg, "=")[2])
    elseif startswith(arg, "--genome=")
        global genome_type = String(split(arg, "=")[2])
    end
end

if genome_type ∉ ("tree", "expr")
    error("Unknown genome type: $genome_type (choose 'tree' or 'expr')")
end

# =============================================================================
# Main
# =============================================================================

ensure_gzip()
equations = equations_for_level(level)
operators = genome_type == "tree" ? make_operators(op_set) : nothing
fset = genome_type == "expr" ? make_function_set(op_set) : nothing

println("=" ^ 72)
println("Feynman Symbolic Regression Benchmark")
println("=" ^ 72)
println()
println("Level:       $level ($(length(equations)) equations)")
println("Genome:      $(genome_type == "tree" ? "TreeGenome (DynamicExpressions)" : "ExprGenome (@eval-based)")")
println("Seeds:       $n_seeds per equation (base seed $base_seed)")
println("Population:  $pop_size")
println("Generations: $generations")
println("Train/test:  $n_train / $n_test points")
println("Operators:   $op_set")
println()

if genome_type == "expr"
    n_total_runs = length(equations) * n_seeds
    est_minutes = round(n_total_runs * generations * pop_size / 60000 * 0.8, digits=0)
    println("WARNING: ExprGenome uses @eval compilation per evaluation, which is")
    println("  ~10x slower than TreeGenome. Estimated wall time: $(Int(est_minutes))-$(Int(est_minutes * 2)) minutes")
    println("  for $n_total_runs runs × $generations generations × $pop_size individuals.")
    println("  Method table growth may cause progressive slowdown on longer runs.")
    println()
end
flush(stdout)

algorithm = GeneticProgramming(
    pop_size=pop_size,
    generations=generations,
    mutation_rate=0.4,
    crossover_rate=0.2,
    elitism=2,
    bloat_penalty=0.005,
)

# Collect results for summary table
struct EquationResult
    name::String
    formula::String
    n_vars::Int
    best_train_r2::Float64
    best_test_r2::Float64
    best_train_mse::Float64
    best_test_mse::Float64
    seeds_converged::Int      # R² > 0.999 on test
    best_expr::String
    best_complexity::Int
    total_wall_time::Float64
end

results = EquationResult[]

for (eq_idx, eq) in enumerate(equations)
    println("-" ^ 72)
    formula_display = eq.formula
    if isempty(formula_display)
        formula_display = fetch_formula_from_metadata(eq.name)
    end
    println("[$eq_idx/$(length(equations))] $(eq.name)")
    if !isempty(formula_display)
        println("  Formula: $formula_display")
    end
    flush(stdout)

    # Load data
    data = try
        load_dataset(eq.name; n_train=n_train, n_test=n_test, data_seed=base_seed)
    catch e
        println("  ERROR loading data: $e")
        println()
        flush(stdout)
        push!(results, EquationResult(eq.name, formula_display, eq.n_vars,
            NaN, NaN, NaN, NaN, 0, "", 0, 0.0))
        continue
    end

    n_vars_actual = data.n_features
    if eq.n_vars > 0 && eq.n_vars != n_vars_actual
        println("  Note: catalog says $(eq.n_vars) vars, data has $n_vars_actual features")
    end
    println("  Variables ($n_vars_actual): $(join(data.col_names, ", "))")
    println("  Data: $(data.n_total) total, $n_train train, $n_test test")
    flush(stdout)

    # Build evaluator for training data
    if genome_type == "tree"
        evaluator = TreeFitnessEvaluator(data.X_train, data.y_train, operators)
    else
        evaluator = make_table_evaluator(data.X_train, data.y_train, data.col_names)
    end

    best_train_r2 = -Inf
    best_test_r2  = -Inf
    best_train_mse = Inf
    best_test_mse  = Inf
    best_expr = ""
    best_complexity = 0
    seeds_converged = 0
    total_wall = 0.0

    for s in 1:n_seeds
        seed = base_seed + s - 1
        if genome_type == "tree"
            problem = GPProblem(evaluator, TreeGenome{Float32}; seed=seed)
        else
            problem = GPProblem(evaluator, ExprGenome;
                                function_set=fset, num_temps=4, seed=seed)
        end

        t0 = time()
        result = solve(problem, algorithm; verbose=false)
        elapsed = time() - t0
        total_wall += elapsed

        # Evaluate on train and test
        if genome_type == "tree"
            train_preds = try
                Float32.(result.best_genome.tree(data.X_train, operators))
            catch
                fill(Float32(NaN), length(data.y_train))
            end
            test_preds = try
                Float32.(result.best_genome.tree(data.X_test, operators))
            catch
                fill(Float32(NaN), length(data.y_test))
            end
        else
            train_preds = try
                expr_predict(result.best_genome, data.X_train, data.col_names)
            catch
                fill(Float32(NaN), length(data.y_train))
            end
            test_preds = try
                expr_predict(result.best_genome, data.X_test, data.col_names)
            catch
                fill(Float32(NaN), length(data.y_test))
            end
        end

        train_stats = fit_stats(train_preds, data.y_train)
        test_stats  = fit_stats(test_preds, data.y_test)

        expr_str = try
            s_str = serialize(result.best_genome)
            length(s_str) > 120 ? s_str[1:117] * "..." : s_str
        catch
            "<serialization failed>"
        end
        n_nodes = Int(complexity(result.best_genome))

        converged = test_stats.r_squared > 0.999
        converged && (seeds_converged += 1)

        status = converged ? "CONVERGED" : "          "
        println("  Seed $seed: train R²=$(round(train_stats.r_squared, digits=6)), " *
                "test R²=$(round(test_stats.r_squared, digits=6)), " *
                "nodes=$n_nodes, $(round(elapsed, digits=1))s $status")
        flush(stdout)

        if test_stats.r_squared > best_test_r2
            best_train_r2  = train_stats.r_squared
            best_test_r2   = test_stats.r_squared
            best_train_mse = train_stats.mse
            best_test_mse  = test_stats.mse
            best_expr      = expr_str
            best_complexity = n_nodes
        end
    end

    println("  Best expression: $best_expr")
    println("  Converged: $seeds_converged/$n_seeds seeds (test R² > 0.999)")
    println()
    flush(stdout)

    push!(results, EquationResult(eq.name, formula_display, n_vars_actual,
        best_train_r2, best_test_r2, best_train_mse, best_test_mse,
        seeds_converged, best_expr, best_complexity, total_wall))
end

# =============================================================================
# Summary table
# =============================================================================

println()
println("=" ^ 72)
println("SUMMARY — Level: $level")
println("=" ^ 72)
println()

# Header
println(rpad("Equation", 24), " ",
        rpad("Vars", 4), " ",
        lpad("Train R²", 10), " ",
        lpad("Test R²", 10), " ",
        lpad("Conv", 6), " ",
        lpad("Nodes", 5), " ",
        lpad("Time", 7))
println("-" ^ 72)

n_solved = 0
n_attempted = 0
for r in results
    global n_attempted += 1
    isnan(r.best_test_r2) && continue
    r.seeds_converged > 0 && (global n_solved += 1)

    short_name = length(r.name) > 23 ? r.name[1:20] * "..." : r.name
    conv_str = "$(r.seeds_converged)/$(n_seeds)"
    println(rpad(short_name, 24), " ",
            rpad(string(r.n_vars), 4), " ",
            lpad(round(r.best_train_r2, digits=4), 10), " ",
            lpad(round(r.best_test_r2, digits=4), 10), " ",
            lpad(conv_str, 6), " ",
            lpad(r.best_complexity, 5), " ",
            lpad("$(round(r.total_wall_time, digits=1))s", 7))
end

println("-" ^ 72)
total_time = sum(r.total_wall_time for r in results)
println()
println("Solution rate: $n_solved / $n_attempted equations with test R² > 0.999")
println("Total wall time: $(round(total_time, digits=1))s")
println()
flush(stdout)
