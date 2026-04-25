using Documenter, Arborist

# Sync the top-level CHANGELOG.md into docs/src/changelog.md before each build
# so the rendered documentation site carries the same content as the GitHub
# changelog. The `docs/src/changelog.md` file itself is intentionally a
# generated artifact.
let src = joinpath(@__DIR__, "..", "CHANGELOG.md"),
    dst = joinpath(@__DIR__, "src", "changelog.md")
    if isfile(src)
        cp(src, dst; force=true)
    end
end

makedocs(
    sitename = "Arborist.jl",
    authors  = "CodeReclaimers LLC",
    modules  = [Arborist],
    checkdocs = :exports,
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home"          => "index.md",
        "Quick Start"   => "quickstart.md",
        "Tutorials"     => [
            "Symbolic Regression"       => "tutorials/symbolic_regression.md",
            "NEAT XOR"                  => "tutorials/neat_xor.md",
            "Control: Cart-Pole"        => "tutorials/control_cartpole.md",
            "Multi-Objective SR"        => "tutorials/nsga2_parsimony.md",
            "LLM-Enhanced GP"           => "tutorials/llm_binpacking.md",
        ],
        "User Guide"    => [
            "Genome Types"    => "genome_types.md",
            "Algorithms"      => "algorithms.md",
            "Evaluators"      => "evaluators.md",
            "Operators"       => "operators.md",
            "Speciation"      => "speciation.md",
            "Plotting"        => "plotting.md",
        ],
        "Advanced"      => [
            "LLM Operator"    => "llm_operator.md",
            "Tree Genome"     => "tree_genome.md",
            "Reproducibility" => "reproducibility.md",
            "Security"        => "security.md",
        ],
        "API Reference" => [
            "Overview"        => "api.md",
            "Genome Types"    => "api/genome.md",
            "Algorithms"      => "api/algorithms.md",
            "Operators"       => "api/operators.md",
            "Infrastructure"  => "api/infrastructure.md",
        ],
        "Changelog"     => "changelog.md",
    ]
)

deploydocs(
    repo = "github.com/CodeReclaimers/Arborist.jl.git",
    devbranch = "master"
)
