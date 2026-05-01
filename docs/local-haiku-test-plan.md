# Local Haiku Model Test Plan

Date: 2026-05-01

## Purpose

This plan defines how to measure whether the repo-local learned haiku
generator is improving. It is for the local haiku model and validation
harness, not for WorkGraph task scoring.

The first useful target is a local CPU generator that can train from the
existing `haikus/*.haiku` corpus, generate from a prompt or observer tag, and
report repeatable local measurements. Automatic checks should gate obvious
regressions, but they are only proxies for poetic quality. Human review remains
the authority for whether outputs are worth keeping.

## Evaluation Loop

Each model run should produce a small evaluation report from the same command
shape:

1. Train or load the local model.
2. Generate a fixed number of samples for the development prompt set.
3. Run automatic local checks on every sample.
4. Summarize pass rates, rejection reasons, novelty, memorization risk, and
   held-out loss when the model can report it.
5. Save representative accepted and rejected examples for periodic review.

Use deterministic seeds for smoke and regression checks. Exploratory training
can use varied seeds, but it must still report the seed used for generation.

## Prompt Splits

Keep prompts separate from poem train/dev/test examples. A model can train on
poems and metadata, but evaluation prompts must be held stable so runs can be
compared.

Recommended split files:

- `data/prompts/train.txt`: prompts allowed for training-time conditioning,
  prompt augmentation, tuning, and manual iteration.
- `data/prompts/dev.txt`: stable prompts used on every local development run.
  These are visible during development and can influence model changes.
- `data/prompts/heldout.txt`: prompts used only for release checks and stage
  gates. Do not tune prompt parsing, reranking thresholds, or templates against
  this set.

Initial split strategy:

- Use a deterministic hash of the normalized prompt text to assign prompts:
  70% train, 15% development, 15% held-out.
- Stratify by observer/topic where possible: `disk`, `process`, `network`,
  generic nature, local machine state, and news-like prompts.
- Keep tiny fixture prompts outside the statistical split. They exist to test
  checker behavior, not model quality.
- Version the prompt files. If a prompt moves between splits, record why in the
  commit message or adjacent changelog.

The haiku corpus should also be split deterministically before training. Strip
frontmatter, normalize line endings, preserve observer/source metadata, dedupe
by normalized poem text, then split by stable file hash or timestamp. The
held-out poem split must never be used for n-gram counts, neural training, or
retrieval examples.

## Local Automatic Checks

Run these checks on generated poems before accepting them into evaluation
summaries or publishable output.

| Check | Local method | Gate | Known limits |
| --- | --- | --- | --- |
| Three-line form | Strip frontmatter and blank outer whitespace; require exactly three non-empty poem lines. | Hard fail. | Does not prove the poem is a haiku. |
| Approximate syllables or length proxy | Prefer a small local syllable estimator with a fallback length band: line word counts near 5/7/5 or character ranges such as 8-32, 12-44, 8-32. | Hard fail for extreme length; warning for near misses. | English syllable estimation is unreliable, especially for numbers, hostnames, acronyms, and technical terms. |
| Language | Require mostly English alphabetic tokens plus expected technical tokens, numbers, paths, interface names, and punctuation. Flag mojibake and non-English-heavy output. | Warning at first; hard fail for unreadable output. | Short poems have too little text for robust language ID. |
| Lexical coherence | Reject obvious fused fragments such as concatenated prompt/topic words, subword markers like `@@`/`##`, and high ratios of tiny leftover fragments. | Hard fail. | Conservative by design; it catches tokenization artifacts, not subtle awkward diction. |
| Repetition | Reject repeated full lines, repeated 3+ word phrases, excessive repeated characters, or a high duplicate-token ratio. | Hard fail for exact repeated lines; warning for softer repetition. | Some intentional repetition can be poetic. |
| Prompt-topic overlap | Extract prompt keywords after stopword removal; require at least one direct keyword, synonym bucket, observer tag, or concrete datum overlap when the prompt has usable content. | Warning for prototype; gate for deployable command. | N-grams may show weak semantics even when they learn local style. Keyword overlap can reward clumsy copying. |
| Novelty | Compare normalized output against prior generated samples from the same run and historical accepted outputs. Reject exact duplicates; report near-duplicate similarity. | Hard fail for exact duplicates. | High novelty does not mean high quality. |
| Train-set overlap | Compare normalized poem hashes and line hashes against train poems. Reject exact full-poem matches; flag two-line overlap and high character/word similarity. | Hard fail for exact train poem; warning or fail for high near-match depending stage. | Short poems naturally share common phrases; thresholds must be conservative. |

The report should include both pass rates and examples. A single aggregate score
is less useful than a breakdown such as `line_count_pass`, `length_pass`,
`topic_overlap_pass`, `exact_train_overlap_count`, and `repetition_fail_count`.

## Prompt-Topic Overlap

Topic overlap exists to catch generators that ignore their prompt. It should be
measured with simple local rules before adding heavier dependencies:

- Normalize prompt and poem to lowercase words.
- Remove common stopwords and haiku task words such as `haiku`, `poem`, and
  `write`.
- Preserve technical tokens such as `cpu`, `ram`, `disk`, `root`, `latency`,
  `localhost`, `network`, `process`, interface names, percentages, and IP-like
  numbers.
- Count direct overlap between prompt keywords and generated poem tokens.
- Add small hand-maintained synonym/topic buckets for the project domains:
  storage/disk, memory/process/cpu, network/latency/packet, local machine, and
  weather/nature.

For a prompt like `write about localhost latency`, a poem mentioning
`localhost`, `packet`, `ping`, `loopback`, or a concrete millisecond value
should pass. A generic poem about moonlight should fail topic overlap even if it
has good structure.

The checker must report its matched keywords or buckets so failures are
debuggable.

## Memorization Checks

The model must not be allowed to look good by replaying the training corpus.
Use layered local checks:

- Exact normalized poem hash against train, development, held-out, and prior
  generated outputs.
- Exact normalized line hash against train lines.
- Two-line overlap check: flag when any two generated lines appear together in
  one training poem.
- Near-match check using a local similarity metric such as character 5-gram
  Jaccard or normalized edit distance against the nearest training poem.
- Retrieval/template baseline exception: it may intentionally return corpus
  poems, but its reports must label those outputs as retrieval and must not be
  counted as learned non-memorized generations.

Release gates should fail exact full-poem train overlap. Near-match thresholds
should start as warnings until fixtures prove the checker is not rejecting too
many legitimate short poems.

## Smoke Tests And Fixtures

Smoke tests must be local-only and deterministic. They should not call external
LLM APIs, WorkGraph evaluation commands, or network services.

Recommended fixtures:

- A valid three-line poem with approximate 5/7/5 shape.
- A one-line output that must fail line count.
- A four-line output that must fail line count.
- A poem with repeated lines that must fail repetition.
- A poem with an exact train-set duplicate that must fail memorization.
- A poem with two train lines and one new line that must be flagged.
- A generic poem for a technical prompt that must fail prompt-topic overlap.
- A prompt-grounded poem mentioning `disk`, `cpu`, or `localhost` that must pass
  topic overlap.
- A poem containing numbers, interface names, or paths to ensure language and
  length checks do not reject normal observer content.
- A malformed fragment fixture with fused topic words or subword markers that
  must fail lexical coherence and appear in JSON and Markdown reports.

Suggested fixture layout:

```text
tests/fixtures/haiku_eval/
  prompts/
    smoke.txt
  train/
    tiny_train.haiku
  generated/
    valid.haiku
    one_line.txt
    repeated_lines.txt
    malformed_fragments.txt
    train_duplicate.haiku
    topic_miss.haiku
    topic_hit.haiku
```

Suggested smoke commands:

- `python -m pytest tests/test_haiku_eval.py`
- `python scripts/evaluate_local_haiku.py --fixtures tests/fixtures/haiku_eval`
- `python scripts/generate_local_haiku.py --model artifacts/local-haiku/latest --prompt "localhost latency" --seed 1`

The final command should be added only after a local generator exists. Until
then, the checker fixture tests are the required smoke surface.

## Human Review Cadence

Human review should improve model direction without blocking every training
run.

- Every local training run: automatic checks only. Save the metrics report and a
  small sample set.
- Daily or after a meaningful model change: review 20 generated poems sampled
  from accepted outputs and 10 sampled from rejected outputs.
- Before promoting a stage gate: review at least 50 poems across the development
  prompts, including all observer/topic categories.
- Before calling a command deployable: review 30 held-out prompt outputs that
  passed automatic checks, plus the top memorization/topic-overlap warnings.

Use a small rubric with `accept`, `borderline`, or `reject` plus optional tags:
`good_image`, `awkward_language`, `missed_prompt`, `memorized`, `too_long`,
`repetitive`, `technical_noise`, and `not_haiku_like`.

Human review trends should be compared over time, but they should not be
optimized as a strict scalar score. The goal is to catch failure modes that the
automatic checks cannot see.

## Stage Gates

### Prototype Gate

The prototype may be a retrieval/template sanity model. It passes when:

- It can parse local `.haiku` files with frontmatter and extract exactly the poem
  lines.
- It can run local fixture checks without network access.
- It emits exactly three poem lines for fixed smoke prompts.
- It rejects exact duplicate publication unless explicitly run in retrieval
  baseline mode.
- It produces an evaluation report with check pass/fail counts and example
  failures.

This gate does not require learned generation.

### First Learned Model Gate

The first learned model should be the CPU-friendly n-gram or equivalent learned
baseline. It passes when:

- Training uses only the training poem split.
- Development evaluation uses the fixed development prompt split.
- The model reports a held-out or development negative log-likelihood,
  perplexity, or equivalent likelihood metric when applicable.
- At least 90% of development samples pass the hard three-line check.
- At least 80% of development samples pass the hard length/syllable proxy.
- Exact train-set poem overlap is zero after rejection sampling.
- Repetition hard failures are below 10% after rejection sampling.
- Prompt-topic overlap is reported for every prompt and shows measurable
  improvement over an unconditioned or shuffled-prompt baseline.
- A short human review finds some outputs that are locally styled and not just
  copied corpus poems.

This gate can still produce awkward poems. It is a learning and measurement
gate, not a publication-quality gate.

### Deployable Local Command Gate

The deployable local command is ready when:

- A user can run one local command with a prompt and seed and receive one
  accepted three-line poem on stdout.
- The command requires no network access and no privileged package install.
- It loads a versioned local model artifact or trains a small model explicitly
  when requested.
- It runs the automatic checks before returning success.
- It exits non-zero or retries with clear local diagnostics when no candidate
  passes.
- Held-out prompts meet or exceed the first learned model thresholds.
- Exact train-set overlap remains zero.
- Prompt-topic overlap is a gate, not only a reported warning.
- A pre-promotion human review finds the accepted held-out samples suitable for
  local use, with known limitations documented.

## Reporting Format

Each evaluation report should include:

- model identifier, git commit, command, seed, timestamp, and corpus split ids;
- number of prompts, candidates, accepted outputs, and rejected outputs;
- automatic check pass/fail counts;
- memorization nearest-neighbor summary;
- prompt-topic overlap summary;
- representative accepted examples;
- representative rejected examples with rejection reasons;
- comparison to the previous promoted run.

Reports can start as Markdown or JSON. If both are emitted, JSON should be the
source of truth and Markdown should be a readable rendering.

## Follow-Up Implementation Work

The documentation plan assumes these implementation tasks exist:

- Build local haiku evaluation checkers and fixture tests.
- Add deterministic prompt and corpus split files.
- Add a local evaluation report command that uses the checkers.
- Wire the deployable local generator command to run checks before success.
