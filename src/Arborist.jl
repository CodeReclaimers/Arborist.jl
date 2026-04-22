"""
    Arborist

A generic, extensible genetic programming framework for Julia, built on the
Problem/Algorithm/Solve pattern. Provides first-class support for expression-tree
genomes, composable genetic operators, and a clean extension interface for
LLM-as-operator and alternative genome representations.

# Quick start
```julia
using Arborist

problem   = GPProblem(evaluator, ExprGenome; function_set=fset, num_temps=4)
algorithm = GeneticProgramming(pop_size=100, generations=200, mutation_rate=0.3)
result    = solve(problem, algorithm; verbose=true)
```
"""
module Arborist

using Random
using Distributed

# --- Abstract type hierarchy ---
include("abstractions.jl")

# --- Core code generation infrastructure (verbatim from existing project) ---
include("genome/codegen.jl")

# --- Evaluators (migrated from evolution.jl) ---
include("evaluators.jl")

# --- Evolutionary machinery (from existing project, minimally modified) ---
include("genome/evolution.jl")

# --- Default function set ---
include("defaults.jl")

# --- AST sanitizer ---
include("sanitizer.jl")

# --- Structured run-log (loaded before speciation so its optional
#     SpeciationSnapshot kwarg resolves at parse time) ---
include("run_log.jl")

# --- Speciation strategies ---
include("speciation.jl")

# --- Migration topology types ---
include("topology.jl")

# --- Result type ---
include("result.jl")

# --- ExprGenome wrapper and GPProblem (struct definitions needed by operators) ---
include("genome/expr_genome.jl")

# --- Operator types (need ExprGenome for method signatures) ---
include("operators/selection.jl")
include("operators/mutation.jl")
include("operators/crossover.jl")

# --- Algorithm configuration (needs operator types) ---
include("algorithm.jl")

# --- Solve entry point ---
include("solve.jl")

# --- Additional genome types ---
include("genome/ant_genome.jl")
include("genome/graph_genome.jl")

# --- NEAT mutation operators (need GraphGenome + NEATCrossover) ---
include("operators/neat_mutation.jl")

# --- Migration types (needs all genome types) ---
include("migration.jl")

# --- Distributed island support ---
include("distributed_island.jl")

# --- Prompt enrichment context for LLM operators ---
include("prompt_context.jl")

# --- LLM mutation operator (uses Downloads.jl stdlib) ---
include("llm_operator.jl")

# --- TreeGenome (DynamicExpressions.jl) ---
include("tree_genome.jl")

# --- NSGA-II multi-objective optimization ---
include("nsga2.jl")

# --- Public API exports ---
export
    # Abstract types
    AbstractGenome,
    AbstractEvaluator,
    AbstractMutationOperator,
    AbstractCrossoverOperator,
    AbstractSelectionStrategy,
    AbstractSpeciation,
    AbstractEvolutionaryAlgorithm,
    AbstractEvolutionResult,

    # Concrete types — genome and problem
    ExprGenome,
    AntGenome,
    GraphGenome,
    TreeGenome,
    GPProblem,
    GPResult,
    TableFitnessEvaluator,
    TreeFitnessEvaluator,
    SymbolicRegressionEvaluator,

    # Concrete types — algorithms
    GeneticProgramming,
    IslandModel,
    NSGAII,

    # Multi-objective types
    AbstractMultiObjectiveEvaluator,
    ParsimonyEvaluator,
    NSGAIIResult,

    # Topology types
    AbstractTopology,
    RingTopology,
    CompleteTopology,
    RandomTopology,
    migration_targets,

    # Migration
    MigrantGenome,
    to_migrant,
    from_migrant,

    # Concrete types — operators
    SubtreeMutation,
    PointMutation,
    HoistMutation,
    ExpansionMutation,
    SubtreeCrossover,
    TournamentSelection,
    LexicaseSelection,
    EpsilonLexicaseSelection,
    select_parent,
    needs_cases,
    WeightPerturbMutation,
    WeightReplaceMutation,
    AddConnectionMutation,
    AddNodeMutation,
    ToggleConnectionMutation,
    NEATDefaultMutation,
    NEATCrossover,
    neat_defaults,

    # Concrete types — speciation
    NoSpeciation,
    ThresholdSpeciation,
    BehavioralSpeciation,
    apply_sharing,

    # Security
    ASTSanitizer,
    DEFAULT_SAFE_CALLS,
    sanitize,

    # Code generation types (needed for custom function sets)
    FunctionDetails,
    FunctionSet,
    GenState,
    LoopLimitExceeded,

    # Ant trail types
    AntSimulator,
    AntEvaluator,
    gp_ant_move,
    gp_ant_left,
    gp_ant_right,
    gp_ant_food_ahead,

    # Graph genome types
    NodeGene,
    ConnectionGene,
    GraphEvaluator,
    EpisodicEvaluator,
    ACTIVATION_FNS,
    reset_innovation_counter!,
    init_innovation_range!,
    GraphGenomeContext,

    # Structured run log
    RunLog,
    GenerationLog,
    record!,
    entries,
    SpeciationSnapshot,

    # Interface functions
    solve,
    evaluate,
    evaluate_cases,
    evaluate_genome,
    evaluate_multi,
    objective_names,
    input_signature,
    output_signature,
    initialize,
    mutate,
    crossover,
    distance,
    complexity,
    serialize,
    deserialize,
    default_function_set,
    boolean_function_set,

    # Boolean operators (available in evolved code)
    gp_nand,
    gp_nor,

    # Initialization strategies
    behavioral_initialize,

    # Utility functions
    add!,
    create_random_assignment,
    create_random_statement,
    create_harness,
    add_loop_checks,
    unravel,
    setup_workers,

    # LLM mutation operator
    LLMMutationOperator,
    LLMCallStats,

    # Prompt enrichment
    MutationContext,
    AbstractPromptSection,
    FitnessSection,
    ElitesSection,
    GenerationSection,
    render_enrichment,

    # Legacy API (from evolution.jl)
    Individual,
    Population,
    evaluate_individual!,
    evolve!

end # module Arborist
