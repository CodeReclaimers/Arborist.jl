# Security Considerations

## The @eval Risk

GenProg.jl's `ExprGenome` and `AntGenome` use Julia's `@eval` to compile
evolved programs into callable functions. This means **any valid Julia
expression can be executed** if it appears in a genome.

When using `LLMMutationOperator`, the LLM generates program text that is
parsed and compiled. If the LLM (or a compromised endpoint) generates
malicious code — filesystem access, network calls, process spawning — it
will be executed with the same privileges as the Julia process.

**This is automated remote code execution** if you use an untrusted LLM
endpoint with ExprGenome or AntGenome.

## The AST Sanitizer

`ASTSanitizer` provides defense-in-depth by checking every function call
in an expression tree against a whitelist of safe operations:

```julia
san = ASTSanitizer()   # default: math + logic only
sanitize(san, :(y = sin(x) + 1.0))   # true — safe
sanitize(san, :(run(`rm -rf /`)))     # false — blocked
```

**What the sanitizer protects against:**
- Calls to dangerous Base functions (run, open, read, write, eval, etc.)
- Qualified calls (Base.run, Sys.exit, etc.)
- Macro calls (@eval, @ccall, etc.)
- Quote/interpolation nodes

**What the sanitizer does NOT protect against:**
- Type confusion attacks using valid operators
- Denial of service via legitimate but expensive computations
- Side-channel attacks
- Exploitation of bugs in Julia itself

## Recommendations

1. **For classical GP (no LLM operator):** The function set already
   constrains what can appear in evolved programs. The sanitizer is
   optional but recommended as defense-in-depth.

2. **For LLM-assisted GP with trusted endpoints:** Enable the sanitizer
   (it is enabled by default when LLMMutationOperator is active).

3. **For LLM-assisted GP with untrusted endpoints:** Run GenProg.jl in
   a container, VM, or sandboxed environment. The AST sanitizer is
   necessary but not sufficient.

4. **TreeGenome is inherently safe:** TreeGenome evaluates expression
   trees via DynamicExpressions.jl without `@eval`. It cannot execute
   arbitrary code. For problems where TreeGenome is applicable
   (symbolic regression, function approximation), it is the safer
   choice.

## Future Work

- `Sandbox.jl` integration for process-isolated evaluation
- Resource limits (CPU time, memory) per evaluation
- Network namespace isolation for LLM calls
