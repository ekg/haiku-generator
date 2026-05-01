# WorkGraph zero-score task transitioned to Done

Task: `wg-bug-report-zero-score-done-transition`
Date: 2026-05-01

## Summary

AutoHaiku exposed a WorkGraph evaluation gate/state-transition bug: the normal
research task `local-haiku-tokenization-research` produced no valid deliverable,
received both FLIP and evaluator scores of `0.00`, and still transitioned from
`PendingEval` to `Done`.

The assignment bug that put this task on the `Default Evaluator` is related, but
this report is about the separate post-evaluation failure. Once WorkGraph had two
zero scores and evidence that the required artifact was missing, the target task
should not have been accepted as complete or allowed to unblock downstream
synthesis.

## Evidence

### Source task state

`wg show local-haiku-tokenization-research` reports:

- Task: `local-haiku-tokenization-research`
- Title: `Research: compact haiku tokenization and structure`
- Status: `done`
- Assigned worker: `agent-65`
- Agent hash: `3184716484e6f0ea08bb13539daf07686ee79d440505f1fdf2de0357707034c3`
- Downstream dependents include `.flip-local-haiku-tokenization-research` and
  `local-haiku-research-synthesis`.

The WorkGraph config identifies that same hash as the configured evaluator:

- `.workgraph/config.toml`: `evaluator_agent =
  "3184716484e6f0ea08bb13539daf07686ee79d440505f1fdf2de0357707034c3"`

The assignment output for `.assign-local-haiku-tokenization-research` says:

```text
Assigned agent to task 'local-haiku-tokenization-research':
  Agent:      Default Evaluator (31847164)
  Role:       Evaluator (75d2fab8)
  Tradeoff:   Evaluator Balanced (ac9bc242)
[assign] Auto-selecting agent: 31847164 for task 'local-haiku-tokenization-research'
```

### Prompt mismatch

The prompt captured at `.workgraph/agents/agent-65/prompt.txt` starts with an
Evaluator identity:

```text
### Role: Evaluator
Grades actor-agents that have completed tasks. Applies rubrics from the task
specification, flags underspecified evaluation criteria, and produces calibrated
grades with transparent rationale.
```

That identity was paired with a normal actor task whose expected output was a
tokenization design note:

```text
Produce a compact design note with 2-4 candidate representations, rough
vocabulary/state sizes, advantages, failure modes, and one recommended baseline
plus one experimental path.
```

### No valid deliverable

The evaluator-run output for the actor task recorded that no actual design note
was present:

```text
Evaluator finding: task is still InProgress and has no
artifacts/output/commits/logged work beyond spawn; wg evaluate run refused
because task is not done/pending-eval. Manual score: 0.00 due no evidence
satisfying validation criteria.
```

The task output directory confirms the absence of a usable artifact:

- `.workgraph/output/local-haiku-tokenization-research/artifacts.json` contains
  `[]`.
- `.workgraph/output/local-haiku-tokenization-research/changes.patch` only
  contains unrelated `.gitignore` and `haiku-log.txt` changes.
- There is no compact tokenization design note satisfying the task's validation
  criteria.

The captured `changes.patch` changes only:

```text
diff --git a/.gitignore b/.gitignore
...
+.wg
diff --git a/haiku-log.txt b/haiku-log.txt
...
+[news]
+Testing duplicate detection\nThis haiku should be unique\nFor now at least, yes
```

### Zero scores still accepted

`wg show local-haiku-tokenization-research` reports both evaluation signals as
hard failure:

```text
Evaluations:
  [iter 0] Score: 0.00  Source: llm
    intent_fidelity:    0.00
  [iter 0] Score: 0.00  Source: flip
```

The `.evaluate-local-haiku-tokenization-research` output is also explicit:

```text
Score:      0.00
  intent_fidelity:        0.00
  correctness:            0.00
  completeness:           0.00
  efficiency:             0.00
  style_adherence:        0.00
  downstream_usability:   0.00
  coordination_overhead:  0.00
  blocking_impact:        0.00
Notes:      No artifacts, outputs, or committed work were recorded, so there is
no observable deliverable to evaluate against the research brief.
```

Despite that, the source task log ends with:

```text
PendingEval → Done (evaluator passed; downstream unblocks)
```

The downstream synthesis task, `local-haiku-research-synthesis`, then completed
with `local-haiku-tokenization-research` listed as a done upstream dependency.

## Analysis

This is related to evaluator misassignment, but distinct from it.

The routing issue explains why the actor task was assigned to `Default Evaluator`
and why its prompt told the worker to grade instead of produce tokenization
research. The state-transition issue is that WorkGraph later had enough evidence
to reject the task and still marked it `Done`.

The observed behavior suggests that the post-evaluation state machine may be
treating successful execution of the evaluator/FLIP tasks as acceptance of the
target task, regardless of the resulting score or gate threshold. In this case,
the evaluator task successfully ran and produced a structured `0.00` score; that
should mean "target task failed acceptance," not "evaluator passed; downstream
unblocks."

This creates false confidence in the graph:

- Downstream implementation and synthesis can proceed with missing research input.
- Consumers see the source task as `Done` even though its required artifact is
  absent.
- A zero-score evaluation becomes informational instead of gate-enforcing.
- The graph records the downstream path as valid despite having no tokenization
  design note, baseline recommendation, multilingual analysis, or control-token
  plan.

Current downstream work should treat tokenization research as suspect. Any plan
or implementation that depended on `local-haiku-tokenization-research` should be
reviewed as if that dependency were missing, unless another later artifact
independently supplied the tokenization design.

## Proposed Upstream Fixes

### Honor the evaluation gate

When moving a target task from `PendingEval` to `Done`, WorkGraph should compare
available evaluator/FLIP scores against `eval_gate_threshold`.

Scores below threshold should not transition to `Done`. The exact terminal state
should follow intended WorkGraph semantics, but it should be one of:

- `incomplete`
- `pending-validation`
- `failed`
- another explicit non-accepted state

The important invariant is that `0.00` cannot mean accepted.

### Treat missing artifacts as non-pass

For tasks whose description requires artifacts, output files, commits, or other
observable deliverables, missing output should be a non-pass unless the task
explicitly allows a no-artifact completion path.

For this task, `artifacts.json == []` and an unrelated `changes.patch` should have
been enough to prevent acceptance even before reading the natural-language score.

### Separate evaluator execution success from target acceptance

WorkGraph should distinguish:

- Evaluator task success: the `.evaluate-*` or `.flip-*` task executed and
  produced a score.
- Target task acceptance: the actor task met its rubric and crossed the configured
  gate threshold.

A successful evaluator run with score `0.00` is a successful evaluation execution
and a failed target-task acceptance.

### Add regression coverage

Add regression tests for a target task that reaches `PendingEval`, receives
scores of `0.00`, and must not transition to `Done`.

Suggested acceptance criteria:

- A task with `llm` score `0.00` and `flip` score `0.00` remains non-accepted.
- A task with any score below `eval_gate_threshold` does not unblock downstream
  tasks.
- A task with empty `artifacts.json` and no relevant committed output fails the
  gate when artifacts are required.
- `.evaluate-*` task completion is recorded separately from target-task
  acceptance.
- The transition log never says `evaluator passed; downstream unblocks` for a
  target task whose evaluator or FLIP score is below threshold.

## Suggested Test Shape

Create a small graph fixture with:

1. A normal actor task that requires an artifact.
2. One downstream task depending on that actor task.
3. Captured evaluation records for the actor task:
   - `Source: llm`, score `0.00`
   - `Source: flip`, score `0.00`
4. An empty output artifact manifest.

Then run the same coordinator/evaluation-finalization path that currently moves
`PendingEval` tasks to `Done`.

Expected result:

- The source task is not `Done`.
- The downstream task is not ready/unblocked.
- The logs explain that evaluation completed but target acceptance failed because
  scores were below `eval_gate_threshold`.

## Relationship To The Routing Bug

`reports/workgraph-evaluator-assignment-routing-bug.md` covers the earlier
assignment-routing failure: normal domain tasks mentioning evaluation-related
language were routed to the configured evaluator identity.

This report covers the later acceptance failure: after WorkGraph produced
explicit `0.00` evaluation results and observed missing output, the target task
still transitioned to `Done`.

Both should be fixed. The routing fix prevents evaluator misassignment; the gate
fix prevents bad or empty work from being accepted when evaluation correctly
detects the failure.
