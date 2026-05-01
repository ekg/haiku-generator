# Local Haiku Dataset

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
