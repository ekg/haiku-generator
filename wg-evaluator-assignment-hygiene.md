# WG evaluator assignment hygiene

Task: `wg-hygiene-evaluator-assignment`
Date: 2026-05-01

## Root cause

Normal AutoHaiku work was routed to the Default Evaluator because the WG assignment prompt and learning-mode scoring do not distinguish WG task scoring from project/domain quality work.

Concrete evidence:

- `local-haiku-validation-research` was a normal research task, but its assignment record at `/home/erik/autohaiku/.wg/agency/assignments/local-haiku-validation-research.yaml:2` selected agent `31847164...` (`Default Evaluator`). The task log shows the spawned agent attempted `wg evaluate run local-haiku-validation-research` against its own still-running task instead of producing the requested validation plan.
- `local-haiku-quality-research` repeated the same pattern even after the task text explicitly said not to run `wg evaluate run`; `/home/erik/autohaiku/.wg/agency/assignments/local-haiku-quality-research.yaml:2` still selected `31847164...`.
- The active WG config identifies `31847164...` as `evaluator_agent` at `/home/erik/autohaiku/.wg/config.toml:61`.
- The evaluator role definition in `/home/erik/autohaiku/.wg/agency/cache/roles/75d2fab8c1f8e0378e335701132a3b68ef9aa32b78841ce73f558116b725b68b.yaml:3` is specifically about grading completed actor-agents. It is not a domain quality measurement planner.
- The assignment prompt builder in `/home/erik/workgraph/src/commands/service/assignment.rs:231` says to match role fit and performance, but does not say that the Evaluator is reserved for `.evaluate-*`, `evaluation`/`agency` task-scoring jobs, or explicit WG grading commands.
- The coordinator assignment path in `/home/erik/workgraph/src/commands/service/coordinator.rs:1620` applies the LLM verdict directly to the source task without a defense-in-depth check against assigning the configured evaluator agent to ordinary tasks.

This is an assignment policy bug amplified by overloaded words: "validation", "evaluation", "quality", and "metrics" in project-domain tasks are being treated as signals for the WG Evaluator identity.

## Safe local correction applied

Open normal tasks that had been assigned to `Default Evaluator` were reassigned with `wg assign` to `Careful Programmer` (`f5143935...`), leaving real `.evaluate-*` tasks on the evaluator:

- `agent-news-haiku-review`
- `local-haiku-impl-first-model`
- `local-haiku-impl-hardware-probe`
- `local-haiku-impl-tokenizer`
- `local-haiku-impl-eval-harness`
- `local-haiku-first-baseline-validation`
- `local-haiku-local-deployment-path`

Actual evaluator tasks such as `.evaluate-local-haiku-impl-eval-harness` should continue to use `Default Evaluator`.

## Required upstream WG fix

The durable fix belongs in the Workgraph assignment code, not by hand-editing graph internals.

Patch points:

1. In `/home/erik/workgraph/src/commands/service/assignment.rs`, add explicit policy to the assignment prompt:

```text
Evaluator agents are only valid for WG task-scoring work: dot-prefixed .evaluate-* tasks, tasks tagged evaluation+agency, or tasks whose explicit command is wg evaluate run.
Do not select an Evaluator for ordinary research, implementation, validation, review, testing, quality, or metrics tasks solely because they mention evaluation or quality checks.
For project/domain quality-measurement work, choose a Programmer, Documenter, Architect, or other actor role.
```

2. In `/home/erik/workgraph/src/commands/service/coordinator.rs`, add a post-verdict guard before applying the assignment: if the selected agent is `config.agency.evaluator_agent` and the source task is not a WG evaluation task, replace it with the best non-evaluator actor or fail the assignment task with a clear log entry.

3. Add a regression test with a task like `Research: local haiku model quality measurement plan` containing `validation`, `evaluation`, and `quality` terms. The expected result is not `config.agency.evaluator_agent`.

4. Generated evaluator task descriptions in `/home/erik/workgraph/src/commands/eval_scaffold.rs:289` still say outputs are read from `.workgraph/output/<task>/`. The active graph is `.wg`, and `wg` now resolves `.wg` as the modern directory. Replace hard-coded `.workgraph/output/...` prompt text with the resolved graph directory basename, or use neutral wording such as "the task output directory under the active WG graph".

## Boundary wording

WG task scoring is the Workgraph meta-evaluation pipeline: `.flip-*`, `.evaluate-*`, `wg evaluate run`, and `evaluation,agency` system tasks.

Project/domain quality measurement is normal actor work: designing haiku metrics, validation datasets, smoke prompts, local test harnesses, or acceptance gates. These tasks may mention "evaluation" and "validation", but they should produce project artifacts and follow-up tasks, not run `wg evaluate run` unless they are explicitly WG evaluator tasks.
