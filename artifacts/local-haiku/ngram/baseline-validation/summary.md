# Local Haiku Baseline Validation

Task: `local-haiku-first-baseline-validation`

Run directory: `artifacts/local-haiku/ngram/baseline-validation`

## Training Result

- Checkpoint: `model.json.gz`
- Runtime: CPU-only Python standard library; no cloud APIs or GPU dependencies.
- Dataset: `data/local-haiku/dataset.jsonl`
- Requested n-gram order: 4
- Effective n-gram order: 4
- Train examples: 1919
- Train poem hashes: 2009
- Vocabulary size: 87
- Dev examples: 232
- Dev token count: 20382
- Dev perplexity: 2.7281432726978925

The baseline checkpoint was created successfully. The ML environment probe is saved in
`ml-env.txt`; optional acceleration is not required for this baseline.

## Held-Out Generation

- Prompt file: `data/prompts/heldout.txt`
- Samples saved: `samples.jsonl`
- Generation failures: none
- Seeds: 100 through 108, one deterministic sample per held-out prompt.

## Evaluation Summary

- Evaluation JSON: `eval.json`
- Evaluation report: `report.md`
- Samples: 9
- Passed: 6
- Failed: 3
- Pass rate: 0.6666666666666666
- Failure counts: `prompt_topic_overlap=3`
- Warning counts: `syllable_proxy=8`

Failed prompts:

- `Write a haiku about a climate report crossing the wire.`
- `Write a haiku about a temp directory after a long build.`
- `Write a haiku about cicadas outside while tests run.`

## Qualitative Findings

The local workflow is operational, but this first learned baseline is not ready
as a quality floor. It frequently emits malformed word fragments such as
`tworkgraph`, `compilesysteady`, and `filliseconnectory`, which is consistent
with a character n-gram model trained on a small, technical corpus. Prompt
anchoring works better for known local machine topics like localhost, network,
process, and local, but fails for broader held-out subjects such as climate and
cicadas. The evaluator catches topic misses and syllable-proxy warnings, but it
does not yet score lexical coherence directly.

## Next Graph Expansion

Highest-value follow-up tasks were added at the current graph level because the
validation node is already at the configured maximum dependency depth:

- `corpus-expand-local`: expand the corpus and held-out prompt coverage for
  non-machine local imagery.
- `tokenizer-reduce-local`: add tokenizer or decoding changes that reduce
  broken word fragments.
- `evaluator-score-local`: improve evaluator diagnostics for lexical coherence
  and syllable quality.
