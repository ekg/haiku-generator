"""End-to-end local workflow smoke coverage for the n-gram baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NETWORK_LINES = (
    "Latency hums low",
    "Loopback packets cross the dawn",
    "Ports answer softly",
)
DISK_LINES = (
    "Disk lanterns glow",
    "Quiet bytes gather at dawn",
    "Root shells answer",
)
PROCESS_LINES = (
    "Process lamps glow",
    "Local cores breathe soft and low",
    "Threads rest quietly",
)


def test_raw_data_to_evaluated_ngram_samples(tmp_path: Path):
    haikus = tmp_path / "haikus"
    haikus.mkdir()
    for index, lines in enumerate(
        [NETWORK_LINES, DISK_LINES, PROCESS_LINES, NETWORK_LINES, DISK_LINES],
        start=1,
    ):
        _write_raw_haiku(haikus / f"fixture-{index}.haiku", lines, index)

    dataset = tmp_path / "dataset.jsonl"
    manifest = tmp_path / "manifest.json"
    splits = tmp_path / "splits.json"
    examples = tmp_path / "tokenizer" / "train-examples.jsonl"
    tokenizer_report = tmp_path / "tokenizer" / "report.json"
    model = tmp_path / "ngram" / "model.json.gz"
    metrics = tmp_path / "ngram" / "metrics.json"
    samples = tmp_path / "ngram" / "samples.jsonl"
    eval_json = tmp_path / "ngram" / "eval.json"
    eval_report = tmp_path / "ngram" / "report.md"

    _run(
        "scripts/build_local_haiku_dataset.py",
        "--input",
        str(haikus),
        "--out",
        str(dataset),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--repo-root",
        str(tmp_path),
    )
    _run(
        "scripts/build_local_haiku_tokenizer_examples.py",
        "--dataset",
        str(dataset),
        "--out",
        str(examples),
        "--report",
        str(tokenizer_report),
    )
    _run(
        "scripts/train_local_haiku_ngram.py",
        "--dataset",
        str(dataset),
        "--order",
        "4",
        "--out",
        str(model),
        "--metrics",
        str(metrics),
    )
    _run(
        "scripts/generate_local_haiku.py",
        "--model",
        str(model),
        "--prompt",
        "localhost latency",
        "--seed",
        "1",
        "--samples-out",
        str(samples),
    )
    _run(
        "scripts/evaluate_local_haiku.py",
        "--dataset",
        str(dataset),
        "--samples",
        str(samples),
        "--json",
        str(eval_json),
        "--report",
        str(eval_report),
    )

    eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))
    train_metrics = json.loads(metrics.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(tokenizer_report.read_text(encoding="utf-8"))

    assert train_metrics["train_example_count"] >= 1
    assert tokenizer_payload["example_count"] == train_metrics["train_example_count"]
    assert eval_payload["metrics"]["sample_count"] == 1
    assert eval_payload["metrics"]["failed_count"] == 0


def _write_raw_haiku(path: Path, lines: tuple[str, str, str], index: int) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                "observer: network",
                f"timestamp: 20260501-000{index:02d}",
                "---",
                *lines,
                "",
                "---",
                "observations: |",
                "  fixture observation ignored by dataset builder",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
