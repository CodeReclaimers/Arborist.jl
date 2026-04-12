# LLM mutation operator — implements the FunSearch/AlphaEvolve pattern:
# serialize genome -> prompt LLM -> parse response -> validate -> return.
# Uses Downloads.jl (stdlib) for HTTP POST requests.

using Downloads

# =============================================================================
# HTTP hook for testability
# =============================================================================

"""
Default HTTP POST implementation using Downloads.jl (stdlib).
Tests replace `_http_post[]` with a mock function.
"""
function _default_http_post(endpoint, headers, body, timeout)
    output = IOBuffer()
    resp = Downloads.request(endpoint;
        method="POST",
        headers=headers,
        input=IOBuffer(body),
        output=output,
        timeout=round(Int, timeout))
    status = resp.status
    if status >= 400
        error("HTTP request failed with status $status: $(String(take!(output)))")
    end
    return String(take!(output))
end

"""
    _http_post

Module-level hook for HTTP POST calls. Default implementation uses Downloads.jl.
Tests can replace this with a mock function via `_http_post[] = mock_fn`.
"""
const _http_post = Ref{Function}(_default_http_post)


# =============================================================================
# Default system prompt
# =============================================================================

const DEFAULT_GP_SYSTEM_PROMPT = """
You are a genetic programming mutation operator. You will be given the
body of a Julia function — a sequence of statements that may include
assignments, `while` loops, `if`/`elseif`/`else` branches, `for` loops,
`break`, `continue`, and standalone calls to the available primitives.
Your task is to produce a meaningfully modified variant that might
better approximate the target function or score higher on the objective.

Rules:
- Return ONLY the body of the function: statements at the top level,
  with control flow nested as needed. Do not wrap your answer in
  `function ... end`, `begin ... end`, or a code fence.
- Use only the variable names present in the original (do not invent
  new ones). `for`-loop iterator variables are the one exception and
  may be introduced locally.
- Preserve the types of all existing variables (e.g. Float32, Int32,
  Bool). Conditions of `while` and `if` must evaluate to `Bool`.
- Only call functions that already appear in the original body — do not
  invent new function names.
- You may change operators, constants, structure, and control flow.
- Do not add imports, function definitions, type declarations, macros,
  or comments. Do not explain your changes.

Respond with only the modified function body and nothing else.
"""


# =============================================================================
# LLM call statistics
# =============================================================================

"""
    LLMCallStats

Mutable accumulator for LLM mutation call outcomes and token usage.
Updated internally by `mutate(::LLMMutationOperator, ...)` — no
external instrumentation needed.

Token fields (`input_tokens`, `output_tokens`) are extracted from the
API response `usage` object when available (Anthropic and OpenAI both
provide this). Character-count fields (`input_chars`, `output_chars`)
are always populated and can be used as a ~4 chars/token estimate when
the API doesn't return exact counts (e.g. some Ollama versions).
"""
mutable struct LLMCallStats
    total_calls::Int        # mutate() invocations that reached LLM dispatch
    llm_successes::Int      # LLM output survived deserialize + sanitize
    llm_failures::Int       # LLM called but output rejected (parse/sanitize)
    fallback_skips::Int     # skipped LLM entirely (no API key, etc.)
    input_tokens::Int       # cumulative, from API response usage field
    output_tokens::Int      # cumulative, from API response usage field
    input_chars::Int        # cumulative char count of user message sent
    output_chars::Int       # cumulative char count of LLM response text
    total_latency::Float64  # cumulative wall-time of HTTP calls (seconds)
end

LLMCallStats() = LLMCallStats(0, 0, 0, 0, 0, 0, 0, 0, 0.0)


# =============================================================================
# LLMMutationOperator struct
# =============================================================================

"""
    LLMMutationOperator <: AbstractMutationOperator

Mutation operator that uses an LLM to generate semantically meaningful
program variations. Implements the FunSearch/AlphaEvolve pattern:
serialize the genome to source, prompt the LLM to improve it, parse
and validate the response.

Falls back to the provided `fallback_op` on any failure (API error,
timeout, parse failure, type-consistency failure). The evolutionary
loop is never interrupted by LLM failures.

# Fields
- `endpoint::String`: API endpoint URL
- `model::String`: model identifier
- `api_key_env::String`: name of environment variable holding the API key
  (empty string means no key needed, e.g. for local Ollama)
- `system_prompt::String`: system prompt for the LLM
- `temperature::Float64`: sampling temperature
- `max_tokens::Int`: maximum response tokens
- `timeout_seconds::Float64`: HTTP request timeout
- `fallback_op::AbstractMutationOperator`: operator to use on any failure
- `sections::Vector{AbstractPromptSection}`: prompt enrichment sections
  (default: empty — no enrichment, identical to pre-enrichment behavior)
- `context::Union{MutationContext, Nothing}`: populated by the solve loop
  each generation; `nothing` until the first generation runs
- `stats::LLMCallStats`: accumulated call outcomes and token usage
- `debug_log::Union{IO, Nothing}`: when set, writes the full user message,
  raw LLM response, and outcome for each call. Set to `open("log.jsonl", "w")`
  or `stdout` for debugging; `nothing` (default) disables logging.
"""
mutable struct LLMMutationOperator <: AbstractMutationOperator
    endpoint::String
    model::String
    api_key_env::String
    system_prompt::String
    temperature::Float64
    max_tokens::Int
    timeout_seconds::Float64
    fallback_op::AbstractMutationOperator
    sections::Vector{AbstractPromptSection}
    context::Union{MutationContext, Nothing}
    stats::LLMCallStats
    debug_log::Union{IO, Nothing}
end

"""
    LLMMutationOperator(; kwargs...) -> LLMMutationOperator

Construct an `LLMMutationOperator` with keyword arguments and sensible defaults.

Supports Anthropic API (default), OpenAI API, and local Ollama without code
changes — only constructor arguments differ.
"""
function LLMMutationOperator(;
    endpoint::String = "https://api.anthropic.com/v1/messages",
    model::String = "claude-sonnet-4-20250514",
    api_key_env::String = "ANTHROPIC_API_KEY",
    system_prompt::String = DEFAULT_GP_SYSTEM_PROMPT,
    temperature::Float64 = 0.8,
    max_tokens::Int = 512,
    timeout_seconds::Float64 = 30.0,
    fallback_op::AbstractMutationOperator = SubtreeMutation(),
    sections::Vector{<:AbstractPromptSection} = AbstractPromptSection[]
)
    LLMMutationOperator(endpoint, model, api_key_env, system_prompt,
                        temperature, max_tokens, timeout_seconds, fallback_op,
                        convert(Vector{AbstractPromptSection}, sections),
                        nothing, LLMCallStats(), nothing)
end


# =============================================================================
# mutate implementation
# =============================================================================

"""
    mutate(op::LLMMutationOperator, g::ExprGenome, rng::AbstractRNG) -> ExprGenome

Serialize the genome, send it to an LLM for semantic mutation, parse and
validate the response. Falls back to `op.fallback_op` on any failure.

All failures are silent at the framework level — uses `@warn` but never
rethrows. The evolutionary loop is robust to 100% LLM failure rate.
"""
function mutate(op::LLMMutationOperator, g::ExprGenome,
                rng::AbstractRNG)::ExprGenome
    stats = op.stats

    # 1. Serialize genome to source string.
    source = serialize(g)

    # 2. Resolve API key.
    api_key = ""
    if !isempty(op.api_key_env)
        if haskey(ENV, op.api_key_env)
            api_key = ENV[op.api_key_env]
        else
            @warn "LLMMutationOperator: API key env var '$(op.api_key_env)' not set, falling back"
            stats.total_calls += 1
            stats.fallback_skips += 1
            return mutate(op.fallback_op, g, rng)
        end
    end

    # 3. Detect provider format and build request.
    is_anthropic = occursin("anthropic.com", op.endpoint)

    headers = Pair{String,String}["Content-Type" => "application/json"]
    if is_anthropic
        push!(headers, "x-api-key" => api_key)
        push!(headers, "anthropic-version" => "2023-06-01")
    elseif !isempty(api_key)
        push!(headers, "Authorization" => "Bearer $api_key")
    end

    body, user_content_len = _build_request_body(op, source, is_anthropic)

    # 4. Make the HTTP call via the replaceable hook.
    t0 = time()
    response_text = try
        _http_post[](op.endpoint, headers, body, op.timeout_seconds)
    catch e
        dt = time() - t0
        @warn "LLMMutationOperator: HTTP request failed" exception=e
        stats.total_calls += 1
        stats.llm_failures += 1
        stats.total_latency += dt
        stats.input_chars += user_content_len
        return mutate(op.fallback_op, g, rng)
    end
    dt = time() - t0
    stats.total_latency += dt
    stats.input_chars += user_content_len

    # 5. Extract token usage from response (before discarding the JSON).
    in_tok, out_tok = _extract_usage(response_text, is_anthropic)
    stats.input_tokens += in_tok
    stats.output_tokens += out_tok

    # 6. Extract text from response.
    text = _extract_response_text(response_text, is_anthropic)

    if text === nothing
        @warn "LLMMutationOperator: failed to extract text from response"
        stats.total_calls += 1
        stats.llm_failures += 1
        return mutate(op.fallback_op, g, rng)
    end
    stats.output_chars += length(text)

    # 7. Deserialize and sanitize — wrapped in try/catch so that any
    #    unexpected exception (e.g. StackOverflowError from deeply nested
    #    LLM output) falls back gracefully rather than crashing the loop.
    outcome = "success"
    result = try
        r = deserialize(ExprGenome, text, g.state)
        if r === nothing
            @warn "LLMMutationOperator: deserialize returned nothing, falling back"
            outcome = "deserialize_nothing"
            nothing
        elseif !sanitize(_build_sanitizer(g.state), r.body)
            @warn "LLMMutationOperator: sanitizer rejected LLM output, falling back"
            outcome = "sanitizer_rejected"
            nothing
        else
            r
        end
    catch e
        e isa InterruptException && rethrow()
        @warn "LLMMutationOperator: deserialize/sanitize threw, falling back" exception=e
        outcome = "deserialize_threw"
        nothing
    end

    stats.total_calls += 1
    if result === nothing
        stats.llm_failures += 1
    else
        stats.llm_successes += 1
    end

    # Debug logging: write full context for each LLM call.
    if op.debug_log !== nothing
        _write_debug_entry(op.debug_log, source, text, outcome, stats.total_calls)
    end

    if result === nothing
        return mutate(op.fallback_op, g, rng)
    end
    return result
end


# =============================================================================
# Internal helpers
# =============================================================================

"""
Build an ASTSanitizer whose whitelist includes both DEFAULT_SAFE_CALLS and
the domain-specific function names from the genome's GenState. Without this,
LLM output containing calls to problem-specific primitives (e.g. bp_n_bins,
bp_place_in_bin) would be rejected by the sanitizer even though the
deserializer's _is_valid_call already validated them against the function set.
"""
function _build_sanitizer(state::GenState)::ASTSanitizer
    allowed = copy(DEFAULT_SAFE_CALLS)
    for fd in state.funcs.funcs
        push!(allowed, fd.name)
    end
    return ASTSanitizer(allowed_calls=allowed)
end

"""
Build JSON request body for the LLM API (no JSON library dependency).
Returns `(body::String, user_content_length::Int)` — the second element
is the character count of the user message before JSON escaping, used
for approximate token accounting.
"""
function _build_request_body(op::LLMMutationOperator, source::String, is_anthropic::Bool)
    # Build enrichment from prompt sections + context.
    enrichment = ""
    if !isempty(op.sections) && op.context !== nothing
        enrichment = render_enrichment(op.sections, op.context)
    end

    # Combine: enrichment first (context), then genome source (the thing to mutate).
    user_content = if isempty(enrichment)
        source
    else
        enrichment * "\n\n--- Program to mutate ---\n" * source
    end

    escaped_system = _json_escape(op.system_prompt)
    escaped_user = _json_escape(user_content)

    body = if is_anthropic
        """{"model":"$(op.model)","max_tokens":$(op.max_tokens),"temperature":$(op.temperature),"system":"$escaped_system","messages":[{"role":"user","content":"$escaped_user"}]}"""
    else
        """{"model":"$(op.model)","max_tokens":$(op.max_tokens),"temperature":$(op.temperature),"messages":[{"role":"system","content":"$escaped_system"},{"role":"user","content":"$escaped_user"}]}"""
    end

    return (body, length(user_content))
end

"""
    _extract_usage(response_text, is_anthropic) -> (input_tokens::Int, output_tokens::Int)

Extract token usage from the API response JSON. Returns `(0, 0)` if
the usage fields are not found (e.g. some Ollama versions omit them).

Anthropic format: `"usage": {"input_tokens": N, "output_tokens": N}`
OpenAI format: `"usage": {"prompt_tokens": N, "completion_tokens": N}`
"""
function _extract_usage(response_text::String, is_anthropic::Bool)
    in_key = is_anthropic ? "input_tokens" : "prompt_tokens"
    out_key = is_anthropic ? "output_tokens" : "completion_tokens"
    in_tok = 0
    out_tok = 0
    m_in = match(Regex("\"$in_key\"\\s*:\\s*(\\d+)"), response_text)
    m_out = match(Regex("\"$out_key\"\\s*:\\s*(\\d+)"), response_text)
    m_in !== nothing && (in_tok = parse(Int, m_in.captures[1]))
    m_out !== nothing && (out_tok = parse(Int, m_out.captures[1]))
    return (in_tok, out_tok)
end

"""
Write a debug log entry for one LLM mutation call. Each entry is a
self-contained block separated by a blank line, with the serialized
parent genome (INPUT), the raw LLM response text (OUTPUT), and the
pipeline outcome.
"""
function _write_debug_entry(io::IO, source::String, response::String,
                            outcome::String, call_num::Int)
    println(io, "===== CALL $call_num [$outcome] =====")
    println(io, "--- INPUT (serialized parent) ---")
    println(io, source)
    println(io, "--- OUTPUT (raw LLM response) ---")
    println(io, response)
    println(io, "--- OUTCOME: $outcome ---")
    println(io)
    flush(io)
end

"""Escape a string for embedding in a JSON string value (RFC 8259)."""
function _json_escape(s::String)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    s = replace(s, "\n" => "\\n")
    s = replace(s, "\r" => "\\r")
    s = replace(s, "\t" => "\\t")
    s = replace(s, "\b" => "\\b")
    s = replace(s, "\f" => "\\f")
    return s
end

"""
Extract the assistant's text from an API JSON response.
Handles both Anthropic (\"text\" field) and OpenAI-compatible (\"content\" field) formats.
Uses simple regex matching — no JSON library required.
"""
function _extract_response_text(body_str::String, is_anthropic::Bool)
    key = is_anthropic ? "text" : "content"
    return _find_last_json_string(body_str, key)
end

"""Find the last occurrence of a \"key\": \"value\" pattern in a JSON string."""
function _find_last_json_string(json::String, key::String)
    # Match "key" : "value" with JSON string escapes.
    pattern = Regex("\"" * key * "\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
    last_match = nothing
    for m in eachmatch(pattern, json)
        last_match = m
    end
    last_match === nothing && return nothing
    raw = last_match.captures[1]
    # Unescape JSON string escapes. The \\\\ -> \\ replacement MUST come
    # first to avoid double-unescaping (e.g., "\\\\n" should become "\\n",
    # not a newline).
    raw = replace(raw, "\\\\" => "\x00BACKSLASH\x00")  # placeholder to avoid interference
    raw = replace(raw, "\\n" => "\n")
    raw = replace(raw, "\\t" => "\t")
    raw = replace(raw, "\\r" => "\r")
    raw = replace(raw, "\\b" => "\b")
    raw = replace(raw, "\\f" => "\f")
    raw = replace(raw, "\\\"" => "\"")
    raw = replace(raw, "\\/" => "/")
    raw = replace(raw, "\x00BACKSLASH\x00" => "\\")
    return raw
end
