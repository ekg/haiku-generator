#!/usr/bin/env python3
"""Evaluate generated local haiku samples.

The command is intentionally local-only and deterministic. It accepts generated
samples from JSONL or text and can compare them with either fixture train poems
or the normalized dataset JSONL artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from haiku_eval import (  # noqa: E402
    evaluate_samples,
    load_prompt_file,
    load_samples,
    load_train_poems,
    render_report,
    write_metrics_json,
    write_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", help="Generated sample JSONL or text file.")
    parser.add_argument("--dataset", help="Normalized dataset JSONL for train overlap checks.")
    parser.add_argument("--train", help="Train poem file or directory for overlap checks.")
    parser.add_argument("--prompts", help="Prompt split file to assign to samples without prompts.")
    parser.add_argument("--json", dest="json_out", help="Write machine-readable metrics JSON.")
    parser.add_argument("--report", help="Write readable Markdown report.")
    parser.add_argument(
        "--fixtures",
        help="Evaluate all generated fixture files under this haiku_eval fixture directory.",
    )
    args = parser.parse_args(argv)

    if not args.samples and not args.fixtures:
        parser.error("one of --samples or --fixtures is required")

    prompts = load_prompt_file(args.prompts) if args.prompts else []
    samples = (
        _fixture_samples(Path(args.fixtures), prompts)
        if args.fixtures
        else load_samples(args.samples, prompts=prompts)
    )
    train_poems = []
    if args.dataset:
        train_poems.extend(load_train_poems(args.dataset))
    if args.train:
        train_poems.extend(load_train_poems(args.train))

    result = evaluate_samples(samples, train_poems=train_poems)

    if args.json_out:
        write_metrics_json(result, args.json_out)
    if args.report:
        write_report(result, args.report)
    if not args.json_out and not args.report:
        print(render_report(result), end="")

    return 1 if result.metrics["failed_count"] else 0


def _fixture_samples(fixture_dir: Path, prompts: list[str]):
    generated = fixture_dir / "generated"
    samples = []
    for path in sorted(generated.iterdir()):
        if path.suffix not in {".haiku", ".txt"}:
            continue
        samples.extend(load_samples(path, prompts=prompts))
    return samples


if __name__ == "__main__":
    raise SystemExit(main())
