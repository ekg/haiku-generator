"""Tests for the local tiny-GRU haiku model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_neural import (  # noqa: E402
    generate_haiku,
    generate_main,
    load_model,
    save_model,
    train_main,
    train_model,
)


def _write_fixture_dataset(path: Path) -> None:
    records = [
        {
            "id": "train-disk-1",
            "language": "en",
            "observer": "disk",
            "provenance": "repo-local",
            "split": "train",
            "lines": ["Disk hums softly", "Local logs count quiet bytes", "Morning shells answer"],
            "text": "Disk hums softly\nLocal logs count quiet bytes\nMorning shells answer",
        },
        {
            "id": "train-disk-2",
            "language": "en",
            "observer": "disk",
            "provenance": "repo-local",
            "split": "train",
            "lines": ["Root drive whispers", "Quiet bytes gather at dawn", "Shells answer softly"],
            "text": "Root drive whispers\nQuiet bytes gather at dawn\nShells answer softly",
        },
        {
            "id": "train-network",
            "language": "en",
            "observer": "network",
            "provenance": "repo-local",
            "split": "train",
            "lines": ["Packets cross dusk", "Loopback hums under glass", "Ports sleep in moonlight"],
            "text": "Packets cross dusk\nLoopback hums under glass\nPorts sleep in moonlight",
        },
        {
            "id": "dev-process",
            "language": "en",
            "observer": "process",
            "provenance": "repo-local",
            "split": "dev",
            "lines": ["Process lamps glow", "Local cores breathe soft and low", "Memory rests now"],
            "text": "Process lamps glow\nLocal cores breathe soft and low\nMemory rests now",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def test_train_save_load_and_generate_three_lines(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    checkpoint = tmp_path / "model.npz"
    metadata = tmp_path / "metadata.json"
    _write_fixture_dataset(dataset)

    model, metrics = train_model(
        dataset,
        embedding_dim=8,
        hidden_size=12,
        epochs=1,
        learning_rate=0.02,
        seed=3,
    )
    save_model(model, checkpoint, metadata)
    loaded = load_model(metadata)
    poem, sample_metadata = generate_haiku(loaded, prompt="disk pressure", seed=7)

    assert checkpoint.exists()
    assert metadata.exists()
    assert loaded.embedding_dim == 8
    assert loaded.hidden_size == 12
    assert metrics["train_example_count"] == 3
    assert metrics["dev_example_count"] == 1
    assert metrics["dev_perplexity"] is not None
    assert len(poem.splitlines()) == 3
    assert sample_metadata["attempt"] >= 1


def test_cli_training_generation_and_evaluation_shape(tmp_path: Path, capsys):
    dataset = tmp_path / "dataset.jsonl"
    checkpoint = tmp_path / "model.npz"
    metadata = tmp_path / "metadata.json"
    metrics = tmp_path / "metrics.json"
    samples = tmp_path / "samples.jsonl"
    _write_fixture_dataset(dataset)

    train_exit = train_main(
        [
            "--dataset",
            str(dataset),
            "--out",
            str(checkpoint),
            "--metadata",
            str(metadata),
            "--metrics",
            str(metrics),
            "--embedding-dim",
            "8",
            "--hidden-size",
            "12",
            "--epochs",
            "1",
            "--seed",
            "5",
        ]
    )
    train_output = capsys.readouterr().out
    generate_exit = generate_main(
        [
            "--model",
            str(metadata),
            "--prompt",
            "localhost latency",
            "--seed",
            "11",
            "--samples-out",
            str(samples),
        ]
    )
    generated = capsys.readouterr().out.strip()
    metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
    sample_payload = json.loads(samples.read_text(encoding="utf-8"))

    assert train_exit == 0
    assert generate_exit == 0
    assert "dev_perplexity=" in train_output
    assert metrics_payload["backend"] == "numpy-gru-cpu"
    assert metrics_payload["dev_negative_log_likelihood"] is not None
    assert len(generated.splitlines()) == 3
    assert len(sample_payload["lines"]) == 3
