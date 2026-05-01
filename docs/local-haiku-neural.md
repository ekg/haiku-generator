# Local Haiku Tiny GRU

The stronger local model is a tiny character/control-token GRU language model
that reuses `data/local-haiku/dataset.jsonl`, the tokenizer stream from
`src/local_haiku_tokenizer.py`, and the same JSONL sample format consumed by
`scripts/evaluate_local_haiku.py`.

Training is local by default:

```bash
python3 scripts/train_local_haiku_neural.py \
  --dataset data/local-haiku/dataset.jsonl \
  --out artifacts/local-haiku/neural/dev/model.npz \
  --metadata artifacts/local-haiku/neural/dev/metadata.json \
  --metrics artifacts/local-haiku/neural/dev/metrics.json
```

Generation loads the JSON metadata, which points at the NumPy checkpoint:

```bash
python3 scripts/generate_local_haiku_neural.py \
  --model artifacts/local-haiku/neural/dev/metadata.json \
  --prompt "localhost latency" \
  --seed 1 \
  --samples-out artifacts/local-haiku/neural/dev/samples.jsonl
```

The metrics file reports train loss, train perplexity, and dev
negative-log-likelihood/perplexity when a dev split is present, so output can be
compared with the n-gram baseline metrics. Generated JSONL can be evaluated with
the existing harness:

```bash
python3 scripts/evaluate_local_haiku.py \
  --dataset data/local-haiku/dataset.jsonl \
  --samples artifacts/local-haiku/neural/dev/samples.jsonl \
  --json artifacts/local-haiku/neural/dev/eval.json \
  --report artifacts/local-haiku/neural/dev/report.md
```

Runtime assumptions:

- CPU is the default success path.
- The command uses local Python plus NumPy and does not call any cloud API.
- PyTorch CPU or PyTorch ROCm may be used by a later compatible trainer, but
  PyTorch and AMD GPU/ROCm are not required for this command.
- Hardware probe results from `scripts/probe_local_ml_env.py` are acceleration
  guidance only; missing GPU support is not a failure.
