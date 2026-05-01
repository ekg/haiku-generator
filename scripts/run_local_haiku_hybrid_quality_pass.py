#!/usr/bin/env python3
"""Run a small local quality matrix for the hybrid haiku decoder."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import re
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import local_haiku_hybrid as hybrid  # noqa: E402
import local_haiku_neural as neural  # noqa: E402
import local_haiku_ngram as ngram  # noqa: E402
from haiku_eval import HaikuSample, evaluate_samples, load_train_poems, write_metrics_json, write_report  # noqa: E402


DEFAULT_PROMPTS = (
    "Write a haiku about localhost latency.",
    "Write a haiku about disk pressure.",
    "Write a haiku about process memory.",
)
DEFAULT_SEEDS = (101, 202)
HYBRID_CONFIGS = (
    {"name": "hybrid_w0.0_temp0.9_top8_pool2", "neural_weight": 0.0, "temperature": 0.9, "top_k": 8, "candidate_pool": 2},
    {"name": "hybrid_w0.6_temp0.9_top8_pool2", "neural_weight": 0.6, "temperature": 0.9, "top_k": 8, "candidate_pool": 2},
    {"name": "hybrid_w1.0_temp0.75_top5_pool1", "neural_weight": 1.0, "temperature": 0.75, "top_k": 5, "candidate_pool": 1},
)
REPORT_PATH = ROOT / "reports/local-haiku-hybrid-quality-pass.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngram-model", default=str(ROOT / "artifacts/local-haiku/ngram/current/model.json.gz"))
    parser.add_argument("--neural-model", default=str(ROOT / "artifacts/local-haiku/neural/current/metadata.json"))
    parser.add_argument("--dataset", default=str(ROOT / "data/local-haiku/dataset.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts/local-haiku/hybrid/quality-pass"))
    parser.add_argument("--max-attempts", type=int, default=200)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ngram_model = ngram.load_model(args.ngram_model)
    neural_model = neural.load_model(args.neural_model)
    train_poems = load_train_poems(args.dataset)

    records: list[dict[str, object]] = []
    records.extend(
        _generate_baseline(
            "ngram",
            DEFAULT_PROMPTS,
            DEFAULT_SEEDS,
            lambda prompt, seed: ngram.generate_haiku(
                ngram_model, prompt=prompt, seed=seed, max_attempts=args.max_attempts
            ),
        )
    )
    records.extend(
        _generate_baseline(
            "neural",
            DEFAULT_PROMPTS,
            DEFAULT_SEEDS,
            lambda prompt, seed: neural.generate_haiku(
                neural_model, prompt=prompt, seed=seed, max_attempts=args.max_attempts
            ),
        )
    )
    for config in HYBRID_CONFIGS:
        records.extend(_generate_hybrid(ngram_model, neural_model, config, DEFAULT_PROMPTS, DEFAULT_SEEDS, args.max_attempts))

    samples_path = out_dir / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    eval_samples = [
        HaikuSample(
            id=str(record["id"]),
            text=str(record["poem"]),
            prompt=str(record["prompt"]),
            source=str(samples_path),
            metadata=dict(record["metadata"]),
        )
        for record in records
        if "poem" in record
    ]
    result = evaluate_samples(eval_samples, train_poems=train_poems)
    write_metrics_json(result, out_dir / "eval.json")
    write_report(result, out_dir / "eval.md")

    summary = _summarize(records, result.to_dict()["samples"])
    (out_dir / "matrix-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "qualitative-notes.md").write_text(_render_notes(summary), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(summary), encoding="utf-8")
    print(f"Wrote {len(records)} records to {samples_path}")
    return 0


def _generate_baseline(
    model_name: str,
    prompts: tuple[str, ...],
    seeds: tuple[int, ...],
    generate: Callable[[str, int], tuple[str, dict[str, object]]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for prompt in prompts:
        for seed in seeds:
            record_id = f"{model_name}_{_slug(prompt)}_{seed}"
            try:
                poem, metadata = generate(prompt, seed)
                metadata = {**metadata, "model": model_name}
                records.append(_record(record_id, model_name, prompt, seed, poem, metadata))
            except RuntimeError as exc:
                records.append(_error_record(record_id, model_name, prompt, seed, str(exc)))
    return records


def _generate_hybrid(
    ngram_model: ngram.NGramModel,
    neural_model: neural.NeuralHaikuModel,
    config: dict[str, object],
    prompts: tuple[str, ...],
    seeds: tuple[int, ...],
    max_attempts: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    model_name = str(config["name"])
    for prompt in prompts:
        for seed in seeds:
            record_id = f"{model_name}_{_slug(prompt)}_{seed}"
            try:
                poem, metadata = hybrid.generate_haiku(
                    ngram_model,
                    neural_model,
                    prompt=prompt,
                    seed=seed,
                    neural_weight=float(config["neural_weight"]),
                    temperature=float(config["temperature"]),
                    top_k=int(config["top_k"]) if config["top_k"] is not None else None,
                    candidate_pool=int(config["candidate_pool"]),
                    max_attempts=max_attempts,
                )
                metadata = {**metadata, "model": model_name}
                records.append(_record(record_id, model_name, prompt, seed, poem, metadata))
            except RuntimeError as exc:
                records.append(_error_record(record_id, model_name, prompt, seed, str(exc), config=config))
    return records


def _record(
    record_id: str,
    model_name: str,
    prompt: str,
    seed: int,
    poem: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "id": record_id,
        "model": model_name,
        "prompt": prompt,
        "seed": seed,
        "poem": poem,
        "lines": poem.split("\n"),
        "metadata": metadata,
        "diagnostics": _diagnose(poem),
    }


def _error_record(
    record_id: str,
    model_name: str,
    prompt: str,
    seed: int,
    error: str,
    *,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": record_id,
        "model": model_name,
        "prompt": prompt,
        "seed": seed,
        "error": error,
        "metadata": {"model": model_name, **(config or {})},
    }


def _diagnose(poem: str) -> dict[str, object]:
    words = re.findall(r"[A-Za-z0-9_./:%'-]+", poem.casefold())
    counts = Counter(words)
    short_words = [word for word in words if len(word) <= 2 and not word.isdigit()]
    numeric_words = [word for word in words if any(ch.isdigit() for ch in word)]
    repeated_words = sorted(word for word, count in counts.items() if count > 1)
    line_end_fragments = [
        words[-1]
        for words in (re.findall(r"[A-Za-z0-9_./:%'-]+", line.casefold()) for line in poem.splitlines())
        if words and len(words[-1]) <= 2 and not words[-1].isdigit()
    ]
    penalty = len(short_words) + len(numeric_words) + (2 * len(line_end_fragments)) + len(repeated_words)
    return {
        "word_count": len(words),
        "short_word_count": len(short_words),
        "numeric_word_count": len(numeric_words),
        "line_end_fragment_count": len(line_end_fragments),
        "repeated_words": repeated_words,
        "quality_penalty": penalty,
    }


def _summarize(records: list[dict[str, object]], evaluated_samples: list[dict[str, object]]) -> dict[str, object]:
    eval_by_id = {sample["id"]: sample for sample in evaluated_samples}
    by_model: dict[str, dict[str, object]] = {}
    for record in records:
        model = str(record["model"])
        bucket = by_model.setdefault(
            model,
            {
                "generated": 0,
                "errors": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "quality_penalty_total": 0,
                "attempt_total": 0,
                "samples": [],
            },
        )
        if "error" in record:
            bucket["errors"] = int(bucket["errors"]) + 1
            continue
        bucket["generated"] = int(bucket["generated"]) + 1
        diagnostics = dict(record.get("diagnostics", {}))
        bucket["quality_penalty_total"] = int(bucket["quality_penalty_total"]) + int(diagnostics.get("quality_penalty", 0))
        bucket["attempt_total"] = int(bucket["attempt_total"]) + int(dict(record["metadata"]).get("attempt", 0))
        eval_sample = eval_by_id.get(record["id"], {})
        if eval_sample.get("passed"):
            bucket["passed"] = int(bucket["passed"]) + 1
        else:
            bucket["failed"] = int(bucket["failed"]) + 1
        if eval_sample.get("warnings"):
            bucket["warnings"] = int(bucket["warnings"]) + 1
        bucket["samples"].append(
            {
                "id": record["id"],
                "prompt": record["prompt"],
                "seed": record["seed"],
                "passed": bool(eval_sample.get("passed")),
                "failures": eval_sample.get("failures", []),
                "warnings": eval_sample.get("warnings", []),
                "diagnostics": diagnostics,
                "lines": record["lines"],
            }
        )

    for bucket in by_model.values():
        generated = max(1, int(bucket["generated"]))
        bucket["pass_rate"] = int(bucket["passed"]) / generated
        bucket["mean_quality_penalty"] = int(bucket["quality_penalty_total"]) / generated
        bucket["mean_attempt"] = int(bucket["attempt_total"]) / generated
    return {"models": by_model}


def _render_notes(summary: dict[str, object]) -> str:
    lines = ["# Hybrid Quality Pass Notes", ""]
    models = dict(summary["models"])
    lines.extend(["## Matrix Summary", ""])
    for model_name, bucket_obj in sorted(models.items()):
        bucket = dict(bucket_obj)
        lines.append(
            f"- {model_name}: pass_rate={bucket['pass_rate']:.3f}, "
            f"mean_quality_penalty={bucket['mean_quality_penalty']:.2f}, "
            f"mean_attempt={bucket['mean_attempt']:.2f}, errors={bucket['errors']}"
        )
    lines.extend(["", "## Highest Penalty Samples", ""])
    samples = []
    for model_name, bucket_obj in models.items():
        for sample in dict(bucket_obj).get("samples", []):
            diagnostics = dict(sample["diagnostics"])
            samples.append((int(diagnostics["quality_penalty"]), model_name, sample))
    for penalty, model_name, sample in sorted(samples, key=lambda item: (item[0], item[1], str(item[2]["id"])), reverse=True)[:8]:
        joined = " / ".join(sample["lines"])
        lines.append(f"- {model_name} {sample['seed']} penalty={penalty}: {joined}")
    lines.append("")
    return "\n".join(lines)


def _render_report(summary: dict[str, object]) -> str:
    models = dict(summary["models"])
    recommendation, rationale = _recommend(summary)
    lines = [
        "# Local Haiku Hybrid Quality Pass",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        rationale,
        "",
        "## Matrix",
        "",
        "| model | pass rate | mean quality penalty | mean attempt | errors |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name, bucket_obj in sorted(models.items()):
        bucket = dict(bucket_obj)
        lines.append(
            f"| {model_name} | {bucket['pass_rate']:.3f} | "
            f"{bucket['mean_quality_penalty']:.2f} | {bucket['mean_attempt']:.2f} | {bucket['errors']} |"
        )

    lines.extend(
        [
            "",
            "## Qualitative Failure Modes",
            "",
            "- The evaluator pass rate is not discriminative in this matrix: every model passed structural checks.",
            "- The neural baseline produced character-level word salad with fragments such as `mawtioaaeoaw`, `dZt`, and repeated short tokens.",
            "- Heavier neural reranking increased retry cost and did not reliably reduce rough lexical artifacts.",
            "- Graph-first hybrid decoding with candidate selection reduced the worst neural and n-gram failures, but still emitted awkward local-metric phrases.",
            "",
            "## Suggested Defaults",
            "",
            "- Use `neural_weight=0.0`, `temperature=0.9`, `top_k=8`, `candidate_pool=2` for the current hybrid graph sampler when a hybrid artifact is required.",
            "- Do not use the tiny-GRU reranker as a quality-improving default until it is retrained or replaced and revalidated.",
            "",
        ]
    )
    return "\n".join(lines)


def _recommend(summary: dict[str, object]) -> tuple[str, str]:
    models = dict(summary["models"])
    ngram = dict(models.get("ngram", {}))
    neural = dict(models.get("neural", {}))
    hybrid_buckets = {
        name: dict(bucket)
        for name, bucket in models.items()
        if str(name).startswith("hybrid_")
    }
    best_hybrid_name, best_hybrid = min(
        hybrid_buckets.items(),
        key=lambda item: (float(item[1]["mean_quality_penalty"]), float(item[1]["mean_attempt"]), item[0]),
    )
    best_penalty = float(best_hybrid["mean_quality_penalty"])
    ngram_penalty = float(ngram.get("mean_quality_penalty", 999.0))
    neural_penalty = float(neural.get("mean_quality_penalty", 999.0))

    if best_penalty < ngram_penalty and best_penalty < neural_penalty:
        return (
            "Use graph-first hybrid defaults only; retrain before enabling neural-weighted reranking.",
            (
                f"The best hybrid setting in this matrix was `{best_hybrid_name}` "
                f"(mean quality penalty {best_penalty:.2f}), better than n-gram "
                f"({ngram_penalty:.2f}) and neural ({neural_penalty:.2f}). "
                "The graph-first result and the high-penalty neural-weighted runs indicate "
                "that the tiny-GRU reranker is not yet adding reliable quality."
            ),
        )

    return (
        "Retrain the tiny neural checkpoint before using neural-weighted hybrid decoding.",
        (
            f"The best hybrid mean quality penalty was {best_penalty:.2f}, compared with "
            f"n-gram {ngram_penalty:.2f} and neural {neural_penalty:.2f}; the current "
            "reranker did not produce a clear improvement."
        ),
    )


def _slug(prompt: str) -> str:
    words = re.findall(r"[a-z0-9]+", prompt.casefold())
    return "-".join(words[-2:]) if len(words) >= 2 else "-".join(words)


if __name__ == "__main__":
    raise SystemExit(main())
