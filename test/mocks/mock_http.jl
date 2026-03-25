# Lightweight mock HTTP layer for testing LLMMutationOperator.
# Intercepts _http_post[] calls and returns canned response body strings.

# Registry of mock responses keyed by a string pattern matched
# against the endpoint URL.
const MOCK_RESPONSES = Dict{String, Any}()

function register_mock_response!(pattern::String, response)
    MOCK_RESPONSES[pattern] = response
end

function clear_mock_responses!()
    empty!(MOCK_RESPONSES)
end

# Install the mock into the module-level hook.
function install_mock_http!()
    Arborist._http_post[] = function(endpoint, headers, body, timeout)
        for (pattern, response) in MOCK_RESPONSES
            if occursin(pattern, endpoint)
                response isa Exception && throw(response)
                return response
            end
        end
        error("No mock registered for endpoint: $endpoint")
    end
end

function restore_http!()
    Arborist._http_post[] = Arborist._default_http_post
end

# --- Canned response builders ---

"""Build a mock Anthropic API JSON response body containing the given text."""
function mock_anthropic_response(text::String)
    escaped = replace(text, "\\" => "\\\\")
    escaped = replace(escaped, "\"" => "\\\"")
    escaped = replace(escaped, "\n" => "\\n")
    return """{"id":"msg_mock","type":"message","role":"assistant","content":[{"type":"text","text":"$escaped"}],"model":"mock","stop_reason":"end_turn"}"""
end

"""Build a mock OpenAI-compatible API JSON response body containing the given text."""
function mock_openai_response(text::String)
    escaped = replace(text, "\\" => "\\\\")
    escaped = replace(escaped, "\"" => "\\\"")
    escaped = replace(escaped, "\n" => "\\n")
    return """{"id":"chatcmpl-mock","choices":[{"index":0,"message":{"role":"assistant","content":"$escaped"},"finish_reason":"stop"}]}"""
end
