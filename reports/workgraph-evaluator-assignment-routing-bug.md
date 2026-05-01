# WorkGraph evaluator assignment routing bug

Task: `wg-bug-report-evaluator-assignment-routing`
Date: 2026-05-01

## Summary

AutoHaiku exposed a WorkGraph assignment-routing bug: a normal domain research task,
`local-haiku-validation-research`, was assigned to the configured `Default Evaluator`
agent and then failed by trying to evaluate its own still-in-progress task.

The root issue is that WorkGraph assignment currently treats domain words such as
"validation", "evaluation", "quality", and "metrics" as evaluator-role signals
without first distinguishing normal actor work from internal WorkGraph scoring work.
That distinction matters because WG already has an orthogonal scoring path:

1. An actor completes a task and produces artifacts or commits.
2. `.flip-*` reconstructs the intended output for comparison.
3. `.evaluate-*` / evaluator scoring reviews completed actor output.

Domain tasks that design validation plans, quality metrics, benchmark fixtures, or
acceptance gates are still actor tasks. They should produce project artifacts. They
should not become internal evaluator tasks merely because they mention evaluation.

## What happened in AutoHaiku

`local-haiku-validation-research` was intended to define how the local haiku model
would be validated. Its expected output was an evaluation plan with metrics, test
fixtures, smoke prompts, and staged acceptance gates.

Instead, the assignment path selected agent `31847164...`, the configured
`Default Evaluator`. The spawned worker behaved like a WorkGraph scorer, not a
domain researcher: it attempted to run `wg evaluate run local-haiku-validation-research`
against its own task while that task was still `InProgress`. Because there were no
completed artifacts, commits, or output directory to grade, it failed the task.

The important failure mode is not just one bad assignment. The evaluator prompt and
agent role were mismatched to the task deliverable. The task asked for a haiku-model
validation plan; the agent identity told the worker to grade completed actor output.

## Evidence from this graph

- `local-haiku-validation-research`
  - Status: `abandoned`.
  - Assigned agent: `3184716484e6f0ea08bb13539daf07686ee79d440505f1fdf2de0357707034c3`.
  - The upstream hygiene artifact identifies that hash as `Default Evaluator` and
    notes it is configured as `config.agency.evaluator_agent`.
  - Log at `2026-05-01T14:50:31Z`: "Evaluation attempted: wg evaluate run refused
    because task is still InProgress; no output artifacts or commits found to grade."
  - Log at `2026-05-01T14:50:41Z`: "Evaluator finding: no evaluation plan artifact
    exists; validation checklist cannot be satisfied; graph growth not performed by
    actor."
  - Log at `2026-05-01T14:50:46Z`: task marked failed because the target task was
    still `InProgress` and had no output to grade.

- `.assign-local-haiku-validation-research`
  - Assignment task for `local-haiku-validation-research`.
  - Completed at `2026-05-01T14:47:42Z`.
  - The downstream source task then spawned with the evaluator agent.

- `wg-hygiene-evaluator-assignment`
  - Produced `wg-evaluator-assignment-hygiene.md`.
  - Documented the root cause as assignment policy failing to distinguish WG task
    scoring from project/domain quality work.
  - Recorded that `local-haiku-quality-research` repeated the same pattern even
    after task text explicitly said not to run `wg evaluate run`.
  - Recorded that normal open tasks were locally reassigned away from
    `Default Evaluator`, while real `.evaluate-*` tasks were left on the evaluator.
  - Completed and passed evaluator review with score `0.92`.

- Prompt role mismatch
  - The `Default Evaluator` role is for grading completed actor-agent work.
  - `local-haiku-validation-research` was a normal research task whose deliverable
    was a plan, follow-up tasks, and acceptance criteria.
  - The worker followed evaluator behavior, so it tried to score the source task
    instead of producing the requested artifact.

- Stale prompt/path wording
  - `wg-evaluator-assignment-hygiene.md` identified generated evaluator task
    descriptions in upstream `src/commands/eval_scaffold.rs` that still hard-code
    `.workgraph/output/<task>/`.
  - AutoHaiku's active graph uses `.wg`.
  - Hard-coding `.workgraph/output` in generated evaluator descriptions is stale
    prompt hygiene and can mislead agents even when graph discovery supports `.wg`.

## Why this matters in WorkGraph terms

WorkGraph has two separate concepts that must stay separate:

- **WG task scoring:** system/meta work such as `.flip-*`, `.evaluate-*`, explicit
  `wg evaluate run ...`, assignment review, and completed-output evaluation.
- **Project/domain quality work:** normal actor work such as creating validation
  datasets, writing metric plans, designing smoke prompts, building benchmark
  harnesses, or defining acceptance gates.

Routing domain quality work to `config.agency.evaluator_agent` collapses these two
layers. The evaluator then grades non-existent output instead of creating the output.
That causes false task failures, blocks downstream work, and can lead the graph to
diagnose the actor as incomplete when the real problem was assignment routing.

## Abandoned upstream attempts

Two direct implementation attempts were created from the hygiene task and then
abandoned:

- `wg-upstream-guard`: intended to add an assignment prompt policy and a
  defense-in-depth post-verdict guard in upstream WorkGraph
  `src/commands/service/assignment.rs` and
  `src/commands/service/coordinator.rs`.
- `wg-upstream-remove`: intended to remove stale `.workgraph/output/<task>/`
  wording from upstream generated evaluator prompts in
  `src/commands/eval_scaffold.rs`.

Both were stopped because editing `/home/erik/workgraph` directly from the
AutoHaiku graph was too rough an execution boundary. Future upstream work should run
inside the WorkGraph repo's own graph, or inside an explicitly created isolated
upstream worktree with a named branch and file scope.

Two safer follow-up shapes were also sketched and then abandoned before execution:

- `wg-upstream-worktree-guard`: same assignment guard, but confined to
  `/home/erik/workgraph-upstream-worktrees/evaluator-assignment-guard`.
- `wg-upstream-worktree-prompt-path`: same prompt/path hygiene fix, but confined to
  `/home/erik/workgraph-upstream-worktrees/evaluator-prompt-path`.

Those task descriptions are useful as implementation notes, but the preferred next
step is to re-create the work in the WorkGraph repo graph or another explicit
isolated upstream worktree.

## Proposed upstream fixes

### 1. Assignment guard

Reserve `config.agency.evaluator_agent` for WorkGraph scoring work only:

- `.evaluate-*` tasks.
- `.flip-*` tasks when they explicitly require evaluator/scoring behavior.
- Explicit `wg evaluate run ...` or equivalent scoring tasks.
- Completed-output review tasks, especially tasks tagged or typed as
  `evaluation,agency`.

Do not select the evaluator for ordinary research, implementation, testing,
validation, quality, metrics, benchmark, or review tasks solely because their text
mentions evaluation-related language.

Recommended implementation shape:

- Add explicit policy wording to the assignment prompt in upstream assignment code.
- Add a post-verdict guard before applying the LLM assignment:
  if the selected agent equals `config.agency.evaluator_agent` and the source task is
  not a WorkGraph scoring task, reject the assignment, choose the best non-evaluator
  actor, or fail the assignment task with a clear routing-invalid log.

### 2. Evaluator prompt self-check

Evaluator prompts should include a defense-in-depth self-check:

- If the assigned task is not an evaluation/scoring task, do not run
  `wg evaluate run <same-task-id>` against itself.
- If the task has a normal artifact deliverable, follow that deliverable as an actor
  would, or flag the assignment as invalid.
- If the task is still `InProgress` and the evaluator is being asked to grade it,
  fail the evaluator assignment cleanly as a routing error rather than marking the
  source actor task failed for missing output.

This would have prevented `local-haiku-validation-research` from being failed as
though an actor had omitted artifacts. The system should have identified that the
evaluator was assigned to the wrong kind of task.

### 3. Prompt/path hygiene

Generated evaluator descriptions should not hard-code `.workgraph/output/<task>/`
when the active graph is `.wg`.

Recommended wording:

- Use the resolved active graph directory name when generating prompt text, or
- Use neutral wording such as "the task output directory under the active WG graph."

This should preserve legacy `.workgraph` graph discovery while avoiding stale
instructions in modern `.wg` graphs.

## Suggested upstream acceptance criteria

- A regression test creates or simulates a normal research task titled like
  `Research: local haiku model validation and quality metrics`; assignment must not
  select `config.agency.evaluator_agent`.
- A regression test confirms a real `.evaluate-*` task still selects or permits the
  configured evaluator agent.
- A regression test confirms explicit scoring work, such as a task whose command is
  `wg evaluate run <completed-task>`, still routes to evaluator behavior.
- A guard test confirms that if the LLM verdict chooses the evaluator for a
  non-scoring actor task, the coordinator rejects or rewrites that verdict and logs
  a clear reason.
- An evaluator-prompt test confirms generated evaluator text no longer hard-codes
  `.workgraph/output` when the active graph directory is `.wg`.
- A compatibility test confirms legacy `.workgraph` graph discovery still works.
- End-to-end smoke: an actor task mentioning validation, evaluation, quality, and
  metrics should be assigned to a normal actor role, complete with an artifact, then
  be scored later by the existing FLIP/evaluator path.

## Execution boundary recommendation

Do not patch `/home/erik/workgraph` directly from AutoHaiku tasks.

For upstream implementation, use one of these boundaries:

1. Create the fix as tasks in the WorkGraph repo's own graph, where commits, tests,
   and smoke gates belong to that repository.
2. Use an explicitly named isolated upstream worktree with a branch, fixed file
   scope, and validation commands recorded in the task.

This AutoHaiku artifact is a handoff note and bug report. It intentionally does not
modify upstream WorkGraph code or create upstream worktrees.
