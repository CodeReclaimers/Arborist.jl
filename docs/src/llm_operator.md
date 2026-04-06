# LLM Mutation Operator

## The FunSearch and AlphaEvolve Connection

`LLMMutationOperator` implements the same "LLM as semantic mutation operator
in an evolutionary loop" pattern as FunSearch (Romera-Paredes et al., 2024)
and AlphaEvolve (DeepMind, 2025). In those systems, a large language model
is used to propose new candidate programs based on existing high-fitness
individuals, replacing or augmenting the blind stochastic mutations of
classical genetic programming.

Arborist.jl provides what FunSearch and AlphaEvolve bundle as bespoke
infrastructure: a typed, reproducible evolutionary loop with pluggable
genome types, explicit RNG seeding, fitness history tracking, speciation,
island models, and bloat control. The LLM operator is one operator among
many — it can be composed with `SubtreeMutation`, `PointMutation`,
`HoistMutation`, and `ExpansionMutation` in the same `mutation_ops` vector,
allowing the evolutionary loop to blend semantic LLM mutations with fast
classical mutations. This is the key architectural difference: the LLM is
a tool within the framework, not the framework itself.

The operator works as follows:

1. **Serialize**: Convert the genome to a human-readable Julia source string
   via `serialize(g::ExprGenome)`.
2. **Prompt**: Send the source to an LLM with a system prompt instructing it
   to produce a meaningfully modified variant.
3. **Parse**: Use `deserialize(ExprGenome, response, state)` to parse the
   LLM's response back into a genome, with partial recovery (invalid lines
   are skipped rather than rejecting the whole response).
4. **Validate**: Type-check parsed assignments against the GenState to ensure
   type consistency.
5. **Fallback**: If any step fails (API error, timeout, parse failure, type
   error), silently fall back to a classical mutation operator. The
   evolutionary loop is never interrupted by LLM failures.

## Contrast with Shadow Gradient Approaches

There is a structural parallel between `LLMMutationOperator` and
Differentiable Topology Search via Shadow Gradients (DTSE): both are
examples of "informed search" that replace blind stochastic operators with
operators that use richer signal.

- **DTSE** routes gradient information through an ODE solver adjoint to
  inform topology search. The gradients provide a local, continuous signal
  about which structural changes would improve the objective.
- **LLMMutationOperator** routes semantic knowledge from pre-training to
  inform program mutation. The LLM's understanding of code patterns provides
  a global, discrete signal about which program transformations are likely
  to be meaningful.

Both can be composed with classical operators in the same evolutionary loop.
In Arborist.jl, this composition is explicit: the `mutation_ops` vector in
`GeneticProgramming` can contain any mix of operator types, and the
evolutionary loop samples from them uniformly.

## Quick-Start Example

`LLMMutationOperator` is exported directly from `Arborist`. It uses
`Downloads.jl` (a Julia stdlib) for HTTP, so no extra HTTP package is required.
The operator currently dispatches on `ExprGenome` only.

### With Anthropic API

```julia
using Arborist

# Create the LLM operator (uses ANTHROPIC_API_KEY env var)
llm_op = LLMMutationOperator(
    model = "claude-sonnet-4-20250514",
    temperature = 0.8,
)

# Mix LLM mutations with classical operators
algorithm = GeneticProgramming(
    pop_size = 100,
    generations = 50,
    mutation_rate = 0.4,
    mutation_ops = AbstractMutationOperator[
        llm_op,              # ~33% LLM mutations
        SubtreeMutation(),   # ~33% subtree mutations
        PointMutation(),     # ~33% point mutations
    ],
)

problem = GPProblem(evaluator, ExprGenome; function_set=fset, seed=42)
result = solve(problem, algorithm; verbose=true)
```

### With Local Ollama

```julia
using Arborist

# Ollama: no API key needed, local endpoint
llm_op = LLMMutationOperator(
    endpoint = "http://localhost:11434/v1/chat/completions",
    model = "codellama:13b",
    api_key_env = "",          # empty = no Authorization header
    timeout_seconds = 60.0,    # local models may be slower
)

algorithm = GeneticProgramming(
    mutation_ops = AbstractMutationOperator[llm_op, SubtreeMutation()],
)
```

### With OpenAI API

```julia
llm_op = LLMMutationOperator(
    endpoint = "https://api.openai.com/v1/chat/completions",
    model = "gpt-4o",
    api_key_env = "OPENAI_API_KEY",
)
```

## Using Local Models via Ollama

Ollama provides an OpenAI-compatible API at
`http://localhost:11434/v1/chat/completions`. Set `api_key_env=""` to skip
the Authorization header.

Model selection for code generation tasks is model-dependent. Models that
have been observed to follow structured system prompts reliably for code
generation include `codellama`, `deepseek-coder`, and `qwen2.5-coder`.
However, adherence to the "return only assignment statements" instruction
varies significantly across models and sizes. We recommend testing any
new model with the mock infrastructure (see `test/mocks/mock_http.jl`)
before using it in production runs.

The `fallback_op` parameter (default: `SubtreeMutation()`) ensures that
if the local model produces unparseable output, the evolutionary loop
continues without interruption. In practice, LLM failure rates of 20-50%
are common with smaller local models, and the framework handles this
gracefully.

## Testing Without Network Access

`Arborist` provides a `_http_post` hook (`Ref{Function}`) that tests can
replace with a mock function. See `test/mocks/mock_http.jl` for the mock
infrastructure and `test/integration/test_llm_operator.jl` for complete
examples of testing all failure modes without network access.
