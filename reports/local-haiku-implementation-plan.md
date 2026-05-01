# Local Haiku Implementation Plan

Date: 2026-05-01

## Decision

Build the first end-to-end local learned haiku generator as a CPU-only smoothed
n-gram model over a compact character/control-token stream. The repo already
has about 2,522 `haikus/*.haiku` files with frontmatter and three poem lines,
which is enough for an initial learned baseline. The neural path should wait
until the dataset, tokenizer, evaluator, and command shape are stable.

The coherent path is:

1. Normalize repo-local `.haiku` files into split JSONL records.
2. Convert records into explicit control-token training examples.
3. Train a CPU n-gram model from the training split.
4. Generate constrained three-line candidates from a prompt or observer tag.
5. Reject exact memorization and obvious form failures.
6. Run the local evaluation harness and save metrics and samples.
7. Package one local prompt command after baseline validation passes.

## Repository Layout

Use this layout for the implementation fan-out:

```text
data/
  local-haiku/
    dataset.jsonl
    splits.json
    manifest.json
  prompts/
    train.txt
    dev.txt
    heldout.txt
artifacts/
  local-haiku/
    ngram/<run-id>/
      model.json.gz
      metrics.json
      samples.jsonl
      report.md
scripts/
  build_local_haiku_dataset.py
  train_local_haiku_ngram.py
  generate_local_haiku.py
  evaluate_local_haiku.py
  probe_local_ml_env.py
tests/
  fixtures/haiku_eval/
  test_local_haiku_*.py
reports/
  local-haiku-implementation-plan.md
```

The exact module names may change to match local implementation style, but the
artifact contract should not: dataset JSONL, prompt split files, model artifact,
sample JSONL, and evaluation report.

## Data Artifacts

The first dataset builder should read only repo-local `haikus/*.haiku` by
default. External downloads remain optional and explicit because licensing and
quality vary. Each normalized JSONL record should include:

- `id`: stable hash or source-relative file key.
- `text`: exactly three poem lines joined with `\n`.
- `lines`: three line strings.
- `source_path`: repo-local path or external source id.
- `provenance`: `repo-local`, `public-domain`, `permissive`, or `external-manual`.
- `license`: known license or `unknown-repo-local`.
- `language`: initially `en` unless a source proves otherwise.
- `observer`: frontmatter observer when present.
- `timestamp`: frontmatter timestamp when present.
- `quality_flags`: parse warnings, duplicate marker, non-three-line marker.
- `split`: deterministic `train`, `dev`, or `test`.

Deduplicate by normalized poem text before splitting. Keep the held-out poem
split out of n-gram counts, neural training, retrieval examples, and threshold
tuning.

## Tokenizer And Representation

Use a character-oriented baseline with explicit multi-character control tokens:

```text
<HAIKU> <LANG=en> <OBSERVER=disk> <PROMPT> localhost latency </PROMPT>
<L1> ... <L2> ... <L3> ... <END>
```

This resolves the tokenization/model tradeoff as follows:

- Character transitions are dense enough for the current small corpus and avoid
  word-level sparsity.
- Explicit line and boundary tokens let the generator learn local diction while
  constrained decoding enforces the three-line shape.
- Prompt and observer control tokens provide a minimal conditioning path without
  pretending the repo has rich prompt/poem pairs.
- Syllable units and Japanese morphology are deferred. English syllable
  estimates are too noisy for a tokenizer contract, and Japanese segmentation
  needs a language-specific corpus and tooling.
- K-mer or de Bruijn graph representations are useful as analysis tools or a
  later constrained decoder experiment, but they are too lossy as the first
  representation because they make prompt control and line semantics harder to
  inspect.

The tokenizer task should report vocabulary size and round-trip line structure.

## First Model

The first learned baseline is a CPU-only smoothed n-gram model:

- 4-character n-gram default, with 3-gram fallback if fixtures are tiny.
- Add-k or interpolated backoff smoothing.
- Counts trained only from `train` split examples.
- Decoding constrained to `<L1>`, `<L2>`, `<L3>`, `<END>` order.
- Candidate rejection for repeated full lines, extreme line length, and exact
  normalized train-poem hash.

This is learned because it estimates corpus transition probabilities. Retrieval
or template generation may exist as a sanity fallback, but it must not be
reported as the first learned model.

Suggested command shape:

```bash
python scripts/build_local_haiku_dataset.py \
  --input haikus \
  --out data/local-haiku/dataset.jsonl \
  --manifest data/local-haiku/manifest.json

python scripts/train_local_haiku_ngram.py \
  --dataset data/local-haiku/dataset.jsonl \
  --order 4 \
  --out artifacts/local-haiku/ngram/dev/model.json.gz \
  --metrics artifacts/local-haiku/ngram/dev/metrics.json

python scripts/generate_local_haiku.py \
  --model artifacts/local-haiku/ngram/dev/model.json.gz \
  --prompt "localhost latency" \
  --seed 1 \
  --samples-out artifacts/local-haiku/ngram/dev/samples.jsonl

python scripts/evaluate_local_haiku.py \
  --dataset data/local-haiku/dataset.jsonl \
  --prompts data/prompts/dev.txt \
  --samples artifacts/local-haiku/ngram/dev/samples.jsonl \
  --report artifacts/local-haiku/ngram/dev/report.md \
  --json artifacts/local-haiku/ngram/dev/eval.json
```

## Stronger Follow-Up

After the n-gram baseline passes the first learned gate, implement a tiny
GRU/LSTM character language model that reuses the same dataset and control-token
stream:

- Embedding 64-128, hidden size 128-256, one or two layers.
- CPU PyTorch as an optional dependency.
- Same prompt/control tokens and evaluator.
- Report dev loss/perplexity and memorization checks.

Tiny Transformer, LoRA, ONNX, `llama.cpp`, and AMD GPU paths remain later
experiments. AMD ROCm is optional acceleration only; CPU must be the default
success path.

## Validation Gates

Prototype gate:

- Dataset builder parses repo-local frontmatter and poem lines.
- Evaluation fixtures run locally with no network or LLM API.
- Any sanity generator emits exactly three poem lines for fixed prompts.
- Reports include pass/fail counts and example failures.

First learned gate:

- N-gram training uses only the training poem split.
- Generation works from a prompt or observer tag with a deterministic seed.
- Exact train-set poem overlap is zero after rejection.
- At least 90% of development samples pass three-line form.
- At least 80% pass length or syllable-proxy bounds.
- Repetition hard failures are below 10% after rejection.
- Prompt-topic overlap is reported for every prompt and improves over an
  unconditioned or shuffled-prompt baseline.
- Held-out or development negative log-likelihood/perplexity is reported.

Deployable local command gate:

- One local command accepts prompt and seed and prints one accepted three-line
  poem to stdout.
- The command uses a local learned model artifact and no cloud API.
- The command runs automatic checks before success, retries locally when needed,
  and exits non-zero with diagnostics if no candidate passes.

## Deployment Shape

Runtime should be a local Python command with CPU default behavior:

```bash
python scripts/generate_local_haiku.py \
  --model artifacts/local-haiku/ngram/latest/model.json.gz \
  --prompt "disk pressure" \
  --seed 42
```

The hardware probe should classify the environment as `cpu`,
`rocm-pytorch`, `onnx-rocm`, `llama-vulkan`, or `unknown`, but missing GPU
support is a normal fallback. The n-gram path must remain standard-library
friendly except for test tooling. PyTorch belongs to the GRU/LSTM follow-up.

## Graph Alignment

Downstream tasks should fan out in this order:

```text
local-haiku-research-synthesis
  -> local-haiku-impl-corpus-dataset
    -> local-haiku-impl-tokenizer
      -> local-haiku-impl-ngram-baseline
        -> local-haiku-integration-sync
          -> local-haiku-first-baseline-validation
            -> local-haiku-local-deployment-path
            -> local-haiku-impl-first-model

local-haiku-research-synthesis
  -> implementation-local-haiku
    -> local-haiku-impl-eval-harness
  -> local-haiku-impl-hardware-probe

local-haiku-test-plan-doc
  -> implementation-deterministic-local
    -> local-haiku-integration-sync
```

`local-haiku-impl-first-model` is intentionally repurposed as the stronger
GRU/LSTM follow-up. It should not block the first end-to-end n-gram integration.
`implementation-local-haiku` owns focused fixture/test examples before the full
evaluation harness consumes them. `implementation-deterministic-local` owns
prompt split files before integration runs development prompts.

## Input Confidence

The training architecture and validation plan have durable artifacts and are
the strongest inputs. The corpus and tokenization research branches completed
but did not leave visible report artifacts in this checkout, so this synthesis
uses their task descriptions and the repo state as lower-confidence inputs.
The resulting plan is still coherent because the repo-local corpus, explicit
control-token stream, CPU n-gram baseline, and local evaluation gates agree with
the validated training and test-plan artifacts.
