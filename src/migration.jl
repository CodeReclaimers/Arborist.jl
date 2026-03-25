# migration.jl — Genome migration types for island models.
#
# MigrantGenome is a lightweight, natively serializable carrier for genome
# data during cross-process migration. GenState and compiled functions are
# reconstructed on the receiving end.

"""
    MigrantGenome

Serializable carrier for genome data during island migration.
Contains only data that survives cross-process transfer — GenState,
compiled functions, and RNG state are reconstructed locally.

# Fields
- `data::Any`: genome-specific payload (Vector{Expr} for ExprGenome, Expr for AntGenome, etc.)
- `fitness::Float64`: fitness on the source island
- `genome_type::Symbol`: identifies the genome type for reconstruction
"""
struct MigrantGenome
    data::Any
    fitness::Float64
    genome_type::Symbol
end

# =============================================================================
# ExprGenome migration
# =============================================================================

"""
    to_migrant(g::ExprGenome, fitness::Float64) -> MigrantGenome

Extract serializable data from an ExprGenome for cross-process migration.
"""
function to_migrant(g::ExprGenome, fitness::Float64)
    MigrantGenome(deepcopy(g.body), fitness, :ExprGenome)
end

"""
    from_migrant(m::MigrantGenome, state::GenState) -> ExprGenome

Reconstruct an ExprGenome from a MigrantGenome using the local GenState.
"""
function from_migrant(m::MigrantGenome, state::GenState)
    @assert m.genome_type == :ExprGenome "Expected ExprGenome migrant, got $(m.genome_type)"
    ExprGenome(deepcopy(m.data), state)
end

# =============================================================================
# AntGenome migration
# =============================================================================

"""
    to_migrant(g::AntGenome, fitness::Float64) -> MigrantGenome

Extract serializable data from an AntGenome for cross-process migration.
"""
function to_migrant(g::AntGenome, fitness::Float64)
    MigrantGenome(deepcopy(g.program), fitness, :AntGenome)
end

"""
    from_migrant(m::MigrantGenome, primitives::Vector{Symbol},
                 conditions::Vector{Symbol}, max_depth::Int) -> AntGenome

Reconstruct an AntGenome from a MigrantGenome.
"""
function from_migrant(m::MigrantGenome, primitives::Vector{Symbol},
                      conditions::Vector{Symbol}, max_depth::Int)
    @assert m.genome_type == :AntGenome "Expected AntGenome migrant, got $(m.genome_type)"
    AntGenome(deepcopy(m.data), primitives, conditions, max_depth)
end

# =============================================================================
# GraphGenome migration
# =============================================================================

"""
    to_migrant(g::GraphGenome, fitness::Float64) -> MigrantGenome

Extract serializable data from a GraphGenome for cross-process migration.
"""
function to_migrant(g::GraphGenome, fitness::Float64)
    MigrantGenome((deepcopy(g.nodes), deepcopy(g.connections),
                   g.n_inputs, g.n_outputs), fitness, :GraphGenome)
end

"""
    from_migrant(m::MigrantGenome, ::Type{GraphGenome}) -> GraphGenome

Reconstruct a GraphGenome from a MigrantGenome.
"""
function from_migrant(m::MigrantGenome, ::Type{GraphGenome})
    @assert m.genome_type == :GraphGenome "Expected GraphGenome migrant, got $(m.genome_type)"
    nodes, connections, n_inputs, n_outputs = m.data
    GraphGenome(deepcopy(nodes), deepcopy(connections), n_inputs, n_outputs, m.fitness)
end
