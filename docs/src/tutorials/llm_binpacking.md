# Heuristic Discovery with LLMMutationOperator

This tutorial shows how to use an LLM as a semantic mutation operator
alongside classical GP operators. It follows the FunSearch / AlphaEvolve
pattern: serialize a genome to source, prompt the LLM to produce a
modified variant, parse and validate the response, fall back to a
classical operator on any failure.

## Prerequisites

Choose a backend:

- **Local Ollama** (recommended for getting-started): install
  [Ollama](https://ollama.com/) and pull a model. The published
  bin-packing study used `qwen3-coder:30b` (~24 GB VRAM); for a
  lightweight first try, `qwen2.5-coder:7b` runs on ~16 GB RAM.
- **Anthropic API**: set `ANTHROPIC_API_KEY` in your environment.
  Default endpoint and model names work out of the box.
- **OpenAI API**: set `OPENAI_API_KEY` and override the endpoint and
  model kwargs on `LLMMutationOperator`.

## Define the operator

```julia
using Arborist

llm_op = LLMMutationOperator(
    endpoint     = "http://localhost:11434/v1/chat/completions",
    model        = "qwen2.5-coder:7b",
    api_key_env  = "",                           # local Ollama: no key
    fallback_op  = SubtreeMutation(),            # on any failure
    temperature  = 0.8,
    max_tokens   = 512,
)
```

`fallback_op` dispatches any time the LLM path fails — API error,
timeout, parse failure, or AST-sanitizer rejection — so the
evolutionary loop is robust to 100% LLM failure. The operator
accumulates statistics in `llm_op.stats::LLMCallStats` (call counts,
tokens, wall time).

## Compose with classical operators

```julia
algorithm = GeneticProgramming(
    pop_size     = 100,
    generations  = 50,
    mutation_ops = [llm_op, SubtreeMutation(), PointMutation()],
)
```

Each mutation is drawn uniformly from `mutation_ops`, so the LLM
operates on roughly 1/3 of mutations above. Fallbacks to
`SubtreeMutation` from `llm_op` do not double-count — the classical
picks are separate draws.

## Genome support

`LLMMutationOperator` currently dispatches on **`ExprGenome`** only.
It serializes the AST to Julia source, prompts the LLM for a
semantically meaningful variation, and re-parses the response with
type-checking. See
[`examples/bin_packing.jl`](https://github.com/CodeReclaimers/Arborist.jl/blob/master/examples/bin_packing.jl)
for a full end-to-end bin-packing run that exercises this path with a
custom function set and evaluator.

`TreeGenome`, `GraphGenome` (via F.2), and `AntGenome` paths are
either implemented (Graph via `mutate(::LLMMutationOperator,
::GraphGenome, rng)` in Phase F.2) or planned; see the Deferred
Research Roadmap in `CLAUDE.md` for current status.

## Prompt enrichment (optional)

For problems where raw source is not enough context,
`AbstractPromptSection` subtypes enrich the user message with
population state:

```julia
llm_op = LLMMutationOperator(
    endpoint    = "http://localhost:11434/v1/chat/completions",
    model       = "qwen2.5-coder:7b",
    api_key_env = "",
    sections    = [
        FitnessSection(),          # parent fitness/rank, pop best/mean
        ElitesSection(3),          # top 3 programs with fitnesses
        GenerationSection(),       # generation number, budget
    ],
)
```

The sections are populated once per generation by the solve loop via
`MutationContext`; custom sections subtype `AbstractPromptSection` and
implement `render(section, context::MutationContext)`.

## Inspecting LLM I/O

Set `debug_log` to capture raw requests and responses:

```julia
io = open("llm_trace.jsonl", "w")
llm_op.debug_log = io
# ... run solve() ...
close(io)
```

Each line is a JSON object containing the full user message, raw
response, and outcome (success / parse-failure / API-error /
api-key-missing).

## What to expect

LLM-enhanced runs typically have:

- **~3× sample efficiency** in generation count (LLM produces
  meaningful mutations faster than syntactic operators on problems
  where the function set has semantic structure, such as loop
  patterns for heuristic discovery).
- **~12× wall-time overhead** per generation when running against
  local Ollama due to per-call latency.

See `examples/bin_packing_overnight_results.md` for the full
experimental write-up on a bin-packing heuristic discovery task.
