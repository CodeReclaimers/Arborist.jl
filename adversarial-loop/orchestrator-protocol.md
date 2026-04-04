# Adversarial Research Loop — Orchestrator Protocol

**Purpose:** You (Claude Code) are the orchestrator of an adversarial research loop. Gemini serves as a critical reviewer that generates testable hypotheses. You test those hypotheses, filter for substance, and maintain the research record. Your job is the one a human researcher would do: routing between critique and investigation, distinguishing signal from noise, and knowing when to stop.

**Directory structure:**
```
adversarial-loop/
├── orchestrator-protocol.md      # this file
├── gemini-prompt-template.md     # template for Gemini review prompts
├── iteration-log.md              # running summary across all iterations
└── iterations/
    └── NNN/                      # one directory per iteration
        ├── focus.md              # what this iteration is reviewing and why
        ├── review-prompt.md      # the assembled prompt sent to Gemini
        ├── gemini-review.md      # Gemini's raw output
        ├── investigation-plan.md # planned investigations from the review
        ├── investigations/       # one subdir per investigation
        │   └── inv_NN/
        │       ├── [scripts, notes, artifacts]
        │       └── RESULTS.md
        ├── synthesis.md          # gathered results and interpretation
        └── doc-updates.md        # what changed in project documents
```

---

## The Loop

### Step 0: Orient

Read `RESEARCH.md` to understand the project scope, constraints, and what's out of bounds. Read `adversarial-loop/iteration-log.md` to understand what previous iterations have covered. Read any source material referenced in RESEARCH.md that you haven't already loaded.

Determine the iteration number (one higher than the last entry in `iteration-log.md`, or 001 if empty). Create the iteration directory.

Decide what to focus this iteration on. Priorities, in order:
1. **Areas that changed since the last review.** Check prior iteration syntheses and doc-updates.
2. **Boundaries and interactions.** The most valuable findings come from testing how components or ideas interact, not how they work in isolation.
3. **Claims applied beyond their original context.** Something validated in one setting but being relied on in a different setting.
4. **The highest-priority research questions** from RESEARCH.md, excluding anything marked out of scope.
5. **Follow-up candidates** from prior iterations that haven't been addressed yet.

Write the focus rationale to `iterations/NNN/focus.md`.

### Step 1: Assemble the Gemini Review Prompt

Use the template in `gemini-prompt-template.md`. Fill in:
- The **focus area** from Step 0
- The **relevant source material** (include actual content, not just references — Gemini needs the text)
- Any **prior findings** that bear on the focus area
- **Specific questions** if you have them

Write the assembled prompt to `iterations/NNN/review-prompt.md`.

**Size guidance:** Include enough context for a thorough review, but don't dump everything if only one area is relevant. A focused review of 2000-5000 lines produces better findings than a shallow review of everything.

### Step 2: Launch Gemini

Run:
```bash
gemini -m gemini-2.5-pro -p "$(cat adversarial-loop/iterations/NNN/review-prompt.md)" > adversarial-loop/iterations/NNN/gemini-review.md 2>&1
```

**Model selection:** Use the best available Gemini model. Try in order:
1. `gemini-3.1-pro-preview` (if available)
2. `gemini-2.5-pro`
3. `gemini-2.5-flash` (fallback — less thorough but functional)

If the prompt exceeds shell argument limits, write it to a temp file and pipe it:
```bash
cat adversarial-loop/iterations/NNN/review-prompt.md | gemini -m gemini-2.5-pro > adversarial-loop/iterations/NNN/gemini-review.md 2>&1
```

If the command fails twice, log the failure in the iteration log and skip to the next focus area.

**Verify output.** Read `gemini-review.md`. If it contains structured findings, proceed. If it's general commentary, extract the actionable claims yourself.

### Step 3: Filter and Plan Investigations

Read Gemini's review. Classify each finding:

- **ACTIONABLE**: A specific, testable claim that could be wrong. Design an investigation.
- **KNOWN**: Restates something already established in the project. Skip, but note the correspondence.
- **PHILOSOPHICAL**: Touches areas RESEARCH.md marks as needing human judgment. Note for the human but do not investigate autonomously.
- **STYLE**: Comments on presentation, naming, or formatting. Ignore.
- **VAGUE**: Sounds plausible but isn't specific enough to test. Can it be sharpened into a testable hypothesis? If yes, sharpen it. If no, note as "requires human judgment."

For each ACTIONABLE finding, design an investigation with:
- A **clear hypothesis** (what Gemini claims might be wrong)
- **Success/failure criteria** (how we'll know if Gemini is right — this is mandatory)
- **Investigation mode** (from the list in RESEARCH.md)
- **Estimated effort/runtime**
- **Blind test required?** (see Step 4, "Three-Agent Blind Test" — use when evaluating detection, monitoring, or defense systems to eliminate hypothesis leakage)

Write the plan to `iterations/NNN/investigation-plan.md`.

**Filtering judgment:** Not every actionable finding is worth investigating. Prefer findings about:
- Untested interactions between components or ideas
- Assumptions being applied beyond their original validation scope
- Failure modes at boundaries
- Claims where being wrong would have significant consequences

Skip findings that are marginal refinements of already-validated results unless the potential impact justifies the effort.

### Step 4: Run Investigations

For each investigation in the plan, choose the appropriate approach based on the investigation mode:

**Code experiments:** Write a script with measurable outputs and explicit pass/fail criteria. Run it. For longer experiments, launch a subagent.

**Proof of concept:** Build the minimal thing needed to test feasibility. Document what worked and what didn't.

**Literature/source search:** Use available search tools. Look for confirming AND disconfirming evidence.

**Formal analysis:** Work through the logic. Show your work. Be explicit about assumptions.

**Counterexample search:** Try to construct the case Gemini describes. If you can construct it, the concern is valid. If you can't after a reasonable effort, note what you tried.

**Benchmarks:** Run the measurement. Report raw numbers, not just pass/fail.

**Three-Agent Blind Test:** Use this mode when investigating detection, monitoring, or defense mechanisms — any scenario where the evaluator knowing the attack could inflate confirmation rates through hypothesis leakage. The problem: if the same agent designs an adversarial input and then evaluates whether the defense caught it, the agent's knowledge of what to look for biases the evaluation. Three subagents with separated knowledge eliminate this:

1. **Agent A (Attacker):** Receives the defense specification (what the monitor checks, what thresholds it uses) and crafts adversarial inputs designed to evade it. Agent A writes the attack data and a sealed description of what each attack does to `inv_NN/attack_manifest.json`. Agent A does NOT see the evaluation criteria.

2. **Agent B (Defender):** Receives the adversarial inputs and runs the monitor, defense, or detection system against them. Agent B records raw outputs (scores, flags, classifications) to `inv_NN/defense_outputs.json`. Agent B does NOT know which inputs are attacks vs. controls, or what the attacks are trying to do.

3. **Agent C (Blind Judge):** Receives Agent B's raw outputs and must determine — without seeing Agent A's attack manifest — which inputs were adversarial, whether the defense caught them, and what the detection rate is. Agent C writes its judgment to `inv_NN/blind_judgment.md`. Only AFTER Agent C's judgment is written does the orchestrator unseal Agent A's manifest and compute the actual confusion matrix.

**When to use blind tests:**
- Testing whether a monitoring layer catches a specific failure mode
- Evaluating adversarial robustness of any classifier or detector
- Any experiment where the hypothesis is "this defense doesn't work" — the confirmation bias risk is highest when you're hoping to find a gap

**When blind tests are overkill:**
- Parameter sweeps and sensitivity analyses (no adversarial component)
- Integration tests checking whether components connect correctly
- Benchmarks measuring performance rather than detection

Each investigation writes a RESULTS.md with:
- What was tested
- Key results (with numbers where applicable)
- Success/failure criteria: pass or fail
- Key findings and decisions

### Step 5: Synthesize

Read all investigation results. Write `iterations/NNN/synthesis.md`:
1. **Confirmed findings** — Gemini was right, something needs to change
2. **Refuted findings** — the current state is correct
3. **Inconclusive** — the investigation didn't clearly resolve the question
4. **Document changes needed** based on confirmed findings
5. **Follow-up candidates** for future iterations

### Step 6: Update Project Documents

Apply confirmed findings to the project's living documents. Whatever the project uses as its source of truth — specs, design docs, READMEs, code comments — update them to reflect what was learned.

Write a summary of changes to `iterations/NNN/doc-updates.md`.

### Step 7: Update Iteration Log

Append to `adversarial-loop/iteration-log.md`:
```markdown
## Iteration NNN — [DATE]

**Focus:** [one-line description]
**Gemini findings:** X actionable, Y known, Z philosophical, W style/vague
**Investigations run:**
- [title]: [one-line result]

**Confirmed findings:** [which Gemini findings turned out to be right]
**Refuted findings:** [which ones the project already handled correctly]
**Document changes:** [brief list]
**Follow-up candidates:** [what the next iteration might focus on]
**So-what assessment:** [Did this iteration produce anything that matters? Be honest.
  A single sentence is fine if the answer is "not much."]
```

### Step 8: Evaluate Stopping Criteria

Check the stopping criteria defined in RESEARCH.md. Default criteria:
- 3rd consecutive iteration → stop for human review
- Fewer than 2 actionable findings → stop
- 3 consecutive iterations with only minor refinements → stop
- Any finding challenges a core assumption marked out-of-scope → stop immediately
- Wall-clock time exceeded → stop
- Uncertain whether to act on a finding → stop and ask

If stopping: write a final summary with cumulative results table and ranked findings, as the last entry in the iteration log.

If continuing: return to Step 0.

---

## Principles

**You are not neutral.** You advocate for the project's integrity. If Gemini finds something real, act on it. If Gemini is wrong, say so clearly.

**Most findings won't pan out.** That's expected. A good adversarial reviewer generates many hypotheses. The value is in the ones that survive testing.

**The human trusts you to run this but wants to review results.** Write the iteration log for someone reading it cold. They should understand what happened, what changed, and whether it matters without reading the raw investigation artifacts.

**Know when you've hit bedrock.** When findings shift from "the design is wrong" to "the design is right but the code is missing" to "the code matches the spec but the spec doesn't match reality" to "everything testable has been tested" — that's a natural progression toward exhaustion, not a failure. Recognize it and stop cleanly.
