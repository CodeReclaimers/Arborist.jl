using Documenter, Arborist

makedocs(
    sitename = "Arborist.jl",
    authors  = "CodeReclaimers LLC",
    modules  = [Arborist],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home"          => "index.md",
        "Quick Start"   => "quickstart.md",
        "User Guide"    => [
            "Genome Types"    => "genome_types.md",
            "Algorithms"      => "algorithms.md",
            "Evaluators"      => "evaluators.md",
            "Operators"       => "operators.md",
            "Speciation"      => "speciation.md",
        ],
        "Advanced"      => [
            "LLM Operator"    => "llm_operator.md",
            "Tree Genome"     => "tree_genome.md",
            "Reproducibility" => "reproducibility.md",
            "Security"        => "security.md",
        ],
        "API Reference" => "api.md",
        "Changelog"     => "changelog.md",
    ]
)

deploydocs(
    repo = "github.com/CodeReclaimers/Arborist.jl.git",
    devbranch = "master"
)
