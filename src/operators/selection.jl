"""
    TournamentSelection <: AbstractSelectionStrategy

Selection strategy that picks `tournament_size` individuals at random
and returns the one with the best (lowest) fitness.

# Fields
- `tournament_size::Int`: number of individuals competing in each tournament
"""
struct TournamentSelection <: AbstractSelectionStrategy
    tournament_size::Int
end
