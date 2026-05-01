# Local Haiku Hybrid Quality Pass

## Recommendation

Tune further before adopting the hybrid decoder as the default.

The best hybrid setting in this matrix was `hybrid_w0.0_temp0.9_top8` (mean quality penalty 1.50), better than n-gram (3.50) and neural (7.83). However, the equally strong graph-only hybrid result and the high-penalty neural-weighted runs indicate that the tiny-GRU reranker is not yet adding reliable quality.

## Matrix

| model | pass rate | mean quality penalty | mean attempt | errors |
| --- | ---: | ---: | ---: | ---: |
| hybrid_w0.0_temp0.9_top8 | 1.000 | 1.50 | 3.00 | 0 |
| hybrid_w0.3_temp0.75_top8 | 1.000 | 4.00 | 3.67 | 0 |
| hybrid_w0.6_temp0.9_top8 | 1.000 | 1.50 | 3.33 | 0 |
| hybrid_w1.0_temp0.75_top5 | 1.000 | 5.33 | 22.83 | 0 |
| neural | 1.000 | 7.83 | 1.00 | 0 |
| ngram | 1.000 | 3.50 | 2.67 | 0 |

## Qualitative Failure Modes

- The evaluator pass rate is not discriminative in this matrix: every model passed structural checks.
- The neural baseline produced character-level word salad with fragments such as `mawtioaaeoaw`, `dZt`, and repeated short tokens.
- Heavier neural reranking increased retry cost and did not reduce rough lexical artifacts.
- Graph-constrained hybrid decoding reduced the worst neural failures but still emitted awkward local-metric phrases and numeric fragments.

## Suggested Defaults

- Use `neural_weight=0.0`, `temperature=0.9`, `top_k=8` for the current hybrid graph sampler when a hybrid artifact is required.
- Do not make the tiny-GRU reranker part of the default production path until it is retrained or replaced and revalidated.
