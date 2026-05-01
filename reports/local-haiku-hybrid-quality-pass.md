# Local Haiku Hybrid Quality Pass

## Scope

Generated a fixed local matrix under `artifacts/local-haiku/hybrid/quality-pass/` using three prompts and two seeds:

- `Write a haiku about localhost latency.`
- `Write a haiku about disk pressure.`
- `Write a haiku about process memory.`

The matrix compares the current n-gram baseline, the tiny-GRU neural baseline, and three hybrid settings. All runs used the existing local evaluator and the current local artifacts.

## Artifacts

- `artifacts/local-haiku/hybrid/quality-pass/samples.jsonl`
- `artifacts/local-haiku/hybrid/quality-pass/eval.json`
- `artifacts/local-haiku/hybrid/quality-pass/eval.md`
- `artifacts/local-haiku/hybrid/quality-pass/matrix-summary.json`
- `artifacts/local-haiku/hybrid/quality-pass/qualitative-notes.md`

## Results

The local evaluator passed every generated sample, so pass rate alone is not discriminating:

| Model/config | Samples | Eval pass rate | Mean roughness penalty | Mean accepted attempt |
| --- | ---: | ---: | ---: | ---: |
| ngram | 6 | 1.000 | 3.50 | 2.67 |
| neural | 6 | 1.000 | 7.83 | 1.00 |
| hybrid w=0.0 temp=0.9 top_k=8 pool=2 | 6 | 1.000 | 0.83 | 3.50 |
| hybrid w=0.6 temp=0.9 top_k=8 pool=2 | 6 | 1.000 | 1.17 | 4.00 |
| hybrid w=1.0 temp=0.75 top_k=5 pool=1 | 6 | 1.000 | 5.33 | 22.83 |

The roughness penalty is a deterministic qualitative diagnostic recorded in the matrix runner. It counts obvious artifacts the evaluator currently misses: very short word fragments, numeric tokens, repeated words, and line-ending fragments.

## Qualitative Findings

The tiny-GRU neural baseline is not usable as a free generator. It repeatedly emits character soup such as `mawtioaaeoaw`, `rofaowgs`, and mixed-case fragments, yet the current evaluator still passes these samples when line count, topic anchor, and basic lexical checks happen to pass.

The n-gram baseline is structurally much better than the neural baseline, but still often produces local word salad and clipped fragments, for example `sixty-s`, `sixty-t`, `Firefox fi`, and repeated numeric phrases.

Hybrid decoding helps when the graph remains dominant. The best setting in this pass was effectively graph-first with accepted-candidate selection: `neural_weight=0.0`, `temperature=0.9`, `top_k=8`, `candidate_pool=2`. `neural_weight=0.6` was close on the diagnostic score, but inspection did not show a reliable semantic improvement over `0.0`.

High neural weight is a blocker. `neural_weight=1.0` raised roughness, required far more accepted-candidate retries, and produced outputs with fragments like trailing `t`, repeated `eight`, and overloaded storage terms.

## Decoder Change

I made one narrow decoder change in `src/local_haiku_hybrid.py`: the generator now collects a small pool of already-accepted candidates and returns the one with the lowest fragment/repetition roughness penalty. This does not replace the n-gram graph constraints or the existing evaluator; it only improves candidate selection among outputs that already pass the decoder rejection checks.

Defaults were tuned to the current evidence:

- `neural_weight=0.0`
- `temperature=0.9`
- `top_k=8`
- `candidate_pool=2`

## Recommendation

Do not adopt the tiny-GRU reranker as a quality-improving component yet. Use the hybrid decoder only in graph-first fallback mode with the defaults above, and retrain or replace the neural checkpoint before increasing `neural_weight`.

The next quality work should tighten the evaluator so neural character soup and clipped line-end fragments fail instead of passing, then rerun this matrix after neural retraining.
