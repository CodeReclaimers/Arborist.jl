# Iteration 003 — Focus

**Area:** LLM mutation operator, AST sanitizer, paper observations accuracy, and Dict iteration determinism in GraphGenome.

**Files under review:**
- `src/llm_operator.jl` — LLMMutationOperator, HTTP integration, JSON parsing, response extraction
- `src/sanitizer.jl` — ASTSanitizer, function call whitelist
- `arborist_paper_observations.md` — claims about algorithm behavior vs actual code
- `src/genome/graph_genome.jl` — Dict iteration patterns (follow-up from iteration 002)

**Why this focus:**
1. The LLM operator is the most complex component not yet reviewed. It handles HTTP, JSON, string escaping, and response parsing without external libraries — each is a potential source of silent failures.
2. The AST sanitizer is a security-relevant component that needs to be correct.
3. Paper observations accuracy is a primary research question (RESEARCH.md questions 3-5) not yet addressed.
4. GraphGenome Dict iteration order is a follow-up from iteration 002 — potential violation of the project's deterministic-ordering convention.
