# Adversarial Review Prompt Template

The orchestrator fills in bracketed sections and assembles the full prompt. Everything below the `---` is the template.

**Usage:**
```bash
gemini -m gemini-2.5-pro -p "$(cat adversarial-loop/iterations/NNN/review-prompt.md)" \
  > adversarial-loop/iterations/NNN/gemini-review.md 2>&1
```

**Assembly instructions:**
1. Copy everything below the `---`
2. Replace `[PROJECT DESCRIPTION]` with a brief description of the research project
3. Replace `[FOCUS AREA]` with the specific topic being reviewed
4. Replace `[FOCUS RATIONALE]` with why this area was chosen
5. Replace `[SOURCE MATERIAL]` with the actual text of relevant documents, code, data, or prior results
6. Replace `[PRIOR FINDINGS]` with relevant results from earlier iterations, or delete the section
7. Optionally replace `[SPECIFIC QUESTIONS]` with pointed questions, or delete the section
8. Review the "What you are NOT being asked to do" section — add any project-specific exclusions from RESEARCH.md
9. Save to `iterations/NNN/review-prompt.md`

---

You are conducting a focused adversarial review of a research project.

**Project:** [PROJECT DESCRIPTION]

Your role is to find flaws, untested assumptions, and failure modes that the development process has missed. You are the adversarial reviewer in a two-model workflow: you generate critical hypotheses, and another system tests them empirically.

## What you are NOT being asked to do

- Suggest wholesale alternatives or redesign the project
- Restate known limitations unless you have a NEW argument about why an existing resolution is wrong
- Comment on writing style, documentation organization, or naming conventions
- Provide general praise or encouragement
<!-- Add project-specific exclusions here, e.g.:
- Question the alignment philosophy — this is a non-negotiable design commitment
- Propose changes to [specific component] — this is deliberately left for later
-->

## What you ARE being asked to do

Find specific, testable problems. For each finding, provide enough detail that someone could design an experiment or investigation to test whether you're right.

**Categories to look for:**

1. **Circular dependencies** — A validates B which feeds A. Logic that appears sound but contains hidden self-reference.

2. **Context mismatch** — Something validated in one context (synthetic data, small scale, specific conditions) is being relied on in a different context. Does it still hold?

3. **Untested boundaries** — Two components or ideas work individually but their interaction hasn't been examined. Where are the seams?

4. **Claims that don't follow from evidence** — The project says "we showed X" — does X actually follow from what was measured? Are there alternative explanations?

5. **Scaling walls** — Something works at current scale but might break as the system grows. What are the limits?

6. **Silent failure modes** — The system produces output that looks correct but is subtly wrong. Where could this happen?

7. **Missing mechanisms** — The project assumes something will work but hasn't specified how. Where are the hand-waves?

## Focus Area for This Review

**Area:** [FOCUS AREA]

**Why this area:** [FOCUS RATIONALE]

## Source Material

[SOURCE MATERIAL]

## Prior Findings (if any)

[PRIOR FINDINGS]

## Specific Questions (if any)

[SPECIFIC QUESTIONS]

## Output Format

Produce numbered findings. For each:

```
### Finding N: [short title]

**Tag:** [STRUCTURAL | PARAMETER | UNTESTED | CIRCULAR | SCALING | SILENT_FAILURE | MISSING_MECHANISM]
**Confidence:** [HIGH | MEDIUM | LOW]
**Severity:** [CRITICAL — breaks a core claim | SIGNIFICANT — wrong but fixable | MINOR — edge case or refinement]

**Claim being challenged:**
[The specific claim or assumption you think is wrong or undertested]

**Why it might be wrong:**
[Your argument. Be specific. Reference the source material.]

**Suggested test:**
[A concrete investigation that would determine whether you're right.
Include what the expected result would be if the project is correct
vs. if your concern is valid.]
```

Aim for 5-15 findings. Quality over quantity — one well-argued structural finding is worth more than ten vague concerns. If you find fewer than 5 things worth reporting, that's fine.
