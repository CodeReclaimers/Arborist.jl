# Lightweight mock HTTP layer for testing LLMMutationOperator.
# Intercepts _http_post[] calls and returns canned responses.

struct MockHTTPResponse
    status::Int
    body::String
end

# Registry of mock responses keyed by a string pattern matched
# against the endpoint URL.
const MOCK_RESPONSES = Dict{String, Any}()

function register_mock_response!(pattern::String, response)
    MOCK_RESPONSES[pattern] = response
end

function clear_mock_responses!()
    empty!(MOCK_RESPONSES)
end

# Install the mock into the extension's hook.
function install_mock_http!(ext)
    ext._http_post[] = function(endpoint, headers, body, timeout)
        for (pattern, response) in MOCK_RESPONSES
            if occursin(pattern, endpoint)
                response isa Exception && throw(response)
                return response
            end
        end
        error("No mock registered for endpoint: $endpoint")
    end
end

function restore_http!(ext)
    ext._http_post[] = ext._default_http_post
end

# --- Canned response builders ---

"""Build a mock Anthropic API JSON response containing the given text."""
function mock_anthropic_response(text::String; status::Int=200)
    escaped = replace(text, "\\" => "\\\\")
    escaped = replace(escaped, "\"" => "\\\"")
    escaped = replace(escaped, "\n" => "\\n")
    body = """{"id":"msg_mock","type":"message","role":"assistant","content":[{"type":"text","text":"$escaped"}],"model":"mock","stop_reason":"end_turn"}"""
    return MockHTTPResponse(status, body)
end

"""Build a mock OpenAI-compatible API JSON response containing the given text."""
function mock_openai_response(text::String; status::Int=200)
    escaped = replace(text, "\\" => "\\\\")
    escaped = replace(escaped, "\"" => "\\\"")
    escaped = replace(escaped, "\n" => "\\n")
    body = """{"id":"chatcmpl-mock","choices":[{"index":0,"message":{"role":"assistant","content":"$escaped"},"finish_reason":"stop"}]}"""
    return MockHTTPResponse(status, body)
end
