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
You are a genetic programming mutation operator. You will be given a
Julia program body as a list of assignment statements. Your task is to
produce a meaningfully modified variant that might better approximate
the target function.

Rules:
- Return ONLY Julia assignment statements, one per line
- Use only the variable names present in the original (do not invent new ones)
- Preserve the types of all variables (Float32, Int32, Bool)
- You may change operators, constants, or structure
- Do not add imports, function definitions, or comments
- Do not explain your changes

Respond with only the modified statements and nothing else.
"""


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
"""
struct LLMMutationOperator <: AbstractMutationOperator
    endpoint::String
    model::String
    api_key_env::String
    system_prompt::String
    temperature::Float64
    max_tokens::Int
    timeout_seconds::Float64
    fallback_op::AbstractMutationOperator
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
    fallback_op::AbstractMutationOperator = SubtreeMutation()
)
    LLMMutationOperator(endpoint, model, api_key_env, system_prompt,
                        temperature, max_tokens, timeout_seconds, fallback_op)
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
    # 1. Serialize genome to source string.
    source = serialize(g)

    # 2. Resolve API key.
    api_key = ""
    if !isempty(op.api_key_env)
        if haskey(ENV, op.api_key_env)
            api_key = ENV[op.api_key_env]
        else
            @warn "LLMMutationOperator: API key env var '$(op.api_key_env)' not set, falling back"
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

    body = _build_request_body(op, source, is_anthropic)

    # 4. Make the HTTP call via the replaceable hook.
    response_text = try
        _http_post[](op.endpoint, headers, body, op.timeout_seconds)
    catch e
        @warn "LLMMutationOperator: HTTP request failed" exception=e
        return mutate(op.fallback_op, g, rng)
    end

    # 5. Extract text from response.
    text = _extract_response_text(response_text, is_anthropic)

    if text === nothing
        @warn "LLMMutationOperator: failed to extract text from response"
        return mutate(op.fallback_op, g, rng)
    end

    # 6. Deserialize the response into an ExprGenome.
    result = deserialize(ExprGenome, text, g.state)

    if result === nothing
        @warn "LLMMutationOperator: deserialize returned nothing, falling back"
        return mutate(op.fallback_op, g, rng)
    end

    return result
end


# =============================================================================
# Internal helpers
# =============================================================================

"""Build JSON request body for the LLM API (no JSON library dependency)."""
function _build_request_body(op::LLMMutationOperator, source::String, is_anthropic::Bool)
    escaped_system = _json_escape(op.system_prompt)
    escaped_source = _json_escape(source)

    if is_anthropic
        return """{"model":"$(op.model)","max_tokens":$(op.max_tokens),"temperature":$(op.temperature),"system":"$escaped_system","messages":[{"role":"user","content":"$escaped_source"}]}"""
    else
        return """{"model":"$(op.model)","max_tokens":$(op.max_tokens),"temperature":$(op.temperature),"messages":[{"role":"system","content":"$escaped_system"},{"role":"user","content":"$escaped_source"}]}"""
    end
end

"""Escape a string for embedding in a JSON string value."""
function _json_escape(s::String)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    s = replace(s, "\n" => "\\n")
    s = replace(s, "\r" => "\\r")
    s = replace(s, "\t" => "\\t")
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
    # Unescape basic JSON string escapes.
    raw = replace(raw, "\\n" => "\n")
    raw = replace(raw, "\\t" => "\t")
    raw = replace(raw, "\\\"" => "\"")
    raw = replace(raw, "\\\\" => "\\")
    return raw
end
