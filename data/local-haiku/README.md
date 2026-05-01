# Local Haiku Dataset

## End-to-End Local N-Gram Workflow

Run the first learned baseline entirely on local files and CPU:

```bash
RUN_DIR=artifacts/local-haiku/ngram/dev

python3 scripts/build_local_haiku_dataset.py \
  --input haikus \
  --out data/local-haiku/dataset.jsonl \
  --manifest data/local-haiku/manifest.json \
  --splits data/local-haiku/splits.json

python3 scripts/build_local_haiku_tokenizer_examples.py \
  --dataset data/local-haiku/dataset.jsonl \
  --out artifacts/local-haiku/tokenizer/train-examples.jsonl \
  --report artifacts/local-haiku/tokenizer/report.json

python3 scripts/probe_local_ml_env.py

python3 scripts/train_local_haiku_ngram.py \
  --dataset data/local-haiku/dataset.jsonl \
  --order 4 \
  --out "$RUN_DIR/model.json.gz" \
  --metrics "$RUN_DIR/metrics.json"

rm -f "$RUN_DIR/samples.jsonl"
python3 scripts/generate_local_haiku.py \
  --model "$RUN_DIR/model.json.gz" \
  --prompt "localhost latency" \
  --seed 1 \
  --samples-out "$RUN_DIR/samples.jsonl"

python3 scripts/evaluate_local_haiku.py \
  --dataset data/local-haiku/dataset.jsonl \
  --prompts data/prompts/dev.txt \
  --samples "$RUN_DIR/samples.jsonl" \
  --report "$RUN_DIR/report.md" \
  --json "$RUN_DIR/eval.json"
```

This path is local-first: the dataset builder reads repo-local `.haiku` files,
the tokenizer emits explicit character/control-token streams, the trainer uses
the learned CPU-only n-gram model, and the evaluator runs deterministic local
checks with no cloud API. `scripts/probe_local_ml_env.py` reports whether
optional acceleration is visible, but missing GPU support is not a failure for
this baseline.

Build the normalized repo-local corpus with:

```bash
python3 scripts/build_local_haiku_dataset.py \
  --input haikus \
  --out data/local-haiku/dataset.jsonl \
  --manifest data/local-haiku/manifest.json \
  --splits data/local-haiku/splits.json
```

The command reads only repo-local `haikus/*.haiku` files. It strips the initial
frontmatter from training text while preserving `observer` and `timestamp`, then
truncates later observation blocks before writing poem lines. External downloads
are intentionally not performed by this command; any future external corpus must
be added through an explicit source path with reviewed provenance and licensing.

Records are deduplicated before splitting by `sha256(normalized poem text)`.
Splits are deterministic from the same normalized text hash: buckets `0..79`
are `train`, `80..89` are `dev`, and `90..99` are `test`. Downstream training
tasks must filter to `split == "train"` and must not use `dev` or `test`
records for model counts, prompt tuning, retrieval examples, or thresholds.

Build baseline tokenizer examples with:

```bash
python3 scripts/build_local_haiku_tokenizer_examples.py \
  --dataset data/local-haiku/dataset.jsonl \
  --out artifacts/local-haiku/tokenizer/train-examples.jsonl \
  --report artifacts/local-haiku/tokenizer/report.json
```

The tokenizer uses a character/control-token stream such as
`<HAIKU> <LANG=en> <OBSERVER=disk> <SOURCE=repo-local> <PROMPT> ... </PROMPT>
<L1> ... <L2> ... <L3> ... <END>`. Characters give dense transitions for the
small repo-local corpus, while explicit control tokens preserve prompt context,
language, observer/source metadata, poem boundaries, and the three line breaks
the decoder must respect. K-mer and de Bruijn graph representations are deferred
to a later experiment because they make prompt control and line semantics harder
to inspect for the first baseline.

Train the first learned baseline with the standard-library, CPU-only n-gram
trainer:

```bash
python3 scripts/train_local_haiku_ngram.py \
  --dataset data/local-haiku/dataset.jsonl \
  --order 4 \
  --out artifacts/local-haiku/ngram/dev/model.json.gz \
  --metrics artifacts/local-haiku/ngram/dev/metrics.json
```

The trainer filters to `split == "train"` for transition counts, stores learned
character/control-token n-gram counts in a gzipped JSON artifact, and reports
development-set negative log likelihood/perplexity when a dev split is present.
It uses add-alpha smoothed backoff and has no GPU, cloud API, or non-standard
runtime dependency.

Generate one constrained three-line candidate locally with:

```bash
python3 scripts/generate_local_haiku.py \
  --model artifacts/local-haiku/ngram/dev/model.json.gz \
  --prompt "disk pressure" \
  --seed 42
```

The decoder builds a prompt/observer control-token prefix, samples from learned
transition counts, constrains line markers to `<L1>`, `<L2>`, `<L3>`, `<END>`,
and retries candidates that exactly match a training poem, repeat full lines,
repeat obvious token phrases, or miss known local prompt topics.

Evaluate generated JSONL samples with:

```bash
python3 scripts/evaluate_local_haiku.py \
  --dataset data/local-haiku/dataset.jsonl \
  --prompts data/prompts/dev.txt \
  --samples artifacts/local-haiku/ngram/dev/samples.jsonl \
  --report artifacts/local-haiku/ngram/dev/report.md \
  --json artifacts/local-haiku/ngram/dev/eval.json
```

The evaluator exits non-zero when any sample fails the local checks, while still
writing the JSON and Markdown reports for debugging.
