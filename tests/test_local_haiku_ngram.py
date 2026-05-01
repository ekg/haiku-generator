"""Tests for the CPU-only local haiku n-gram baseline."""

import gzip
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_ngram import (  # noqa: E402
    DEFAULT_VALIDATED_MODEL,
    L1_TOKEN,
    _allowed_tokens,
    generate_haiku,
    generate_main,
    load_model,
    rejection_reason,
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
    model_path = tmp_path / "model.json.gz"
    _write_fixture_dataset(dataset)

    model, metrics = train_model(dataset, order=4)
    save_model(model, model_path)
    loaded = load_model(model_path)
    poem, metadata = generate_haiku(loaded, prompt="disk pressure", seed=7)

    assert model_path.exists()
    assert loaded.order == 4
    assert loaded.effective_order == 4
    assert metrics["train_example_count"] == 3
    assert metrics["dev_example_count"] == 1
    assert metrics["dev_perplexity"] is not None
    assert len(poem.splitlines()) == 3
    assert metadata["attempt"] >= 1
    assert rejection_reason(loaded, poem) is None


def test_model_artifact_contains_learned_transition_counts(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    model_path = tmp_path / "model.json.gz"
    _write_fixture_dataset(dataset)

    model, _metrics = train_model(dataset, order=4)
    save_model(model, model_path)

    with gzip.open(model_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["counts"]["1"]["[]"]["<HAIKU>"] == 3
    assert payload["counts"]["2"]['["<L1>"]']["D"] == 1
    assert payload["train_poem_count"] == 3


def test_cli_training_and_generation_shapes(tmp_path: Path, capsys):
    dataset = tmp_path / "dataset.jsonl"
    model_path = tmp_path / "model.json.gz"
    metrics_path = tmp_path / "metrics.json"
    samples_path = tmp_path / "samples.jsonl"
    _write_fixture_dataset(dataset)

    train_exit = train_main(
        [
            "--dataset",
            str(dataset),
            "--order",
            "4",
            "--out",
            str(model_path),
            "--metrics",
            str(metrics_path),
        ]
    )
    train_output = capsys.readouterr().out
    generate_exit = generate_main(
        [
            "--model",
            str(model_path),
            "--prompt",
            "disk pressure",
            "--seed",
            "11",
            "--samples-out",
            str(samples_path),
        ]
    )
    generated = capsys.readouterr().out.strip()

    assert train_exit == 0
    assert generate_exit == 0
    assert "dev_perplexity=" in train_output
    assert model_path.exists()
    assert metrics_path.exists()
    assert len(generated.splitlines()) == 3
    assert samples_path.read_text(encoding="utf-8").count("\n") == 1


def test_cli_generation_defaults_to_validated_artifact(capsys):
    assert DEFAULT_VALIDATED_MODEL.exists()

    generate_exit = generate_main(
        [
            "--prompt",
            "Write a haiku about localhost latency.",
            "--seed",
            "301",
        ]
    )
    generated = capsys.readouterr().out.strip()

    assert generate_exit == 0
    assert len(generated.splitlines()) == 3


def test_cli_generation_reports_validation_exhaustion(tmp_path: Path, capsys):
    dataset = tmp_path / "dataset.jsonl"
    model_path = tmp_path / "model.json.gz"
    _write_fixture_dataset(dataset)
    model, _metrics = train_model(dataset, order=4)
    save_model(model, model_path)

    generate_exit = generate_main(
        [
            "--model",
            str(model_path),
            "--prompt",
            "disk pressure",
            "--seed",
            "11",
            "--max-attempts",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert generate_exit == 2
    assert captured.out == ""
    assert "ERROR: no accepted candidate after 0 attempts" in captured.err


def test_tiny_fixture_falls_back_to_smaller_effective_order(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    record = {
        "id": "tiny",
        "language": "en",
        "observer": "disk",
        "provenance": "repo-local",
        "split": "train",
        "lines": ["A", "B", "C"],
        "text": "A\nB\nC",
    }
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

    model, metrics = train_model(dataset, order=200)

    assert model.effective_order < 200
    assert metrics["effective_order"] == model.effective_order


def test_word_boundary_decoder_blocks_spliced_word_fragments(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    records = [
        {
            "id": "train-process-1",
            "language": "en",
            "observer": "process",
            "provenance": "repo-local",
            "split": "train",
            "lines": ["Compile steady", "Process logs answer now", "Memory rests softly"],
            "text": "Compile steady\nProcess logs answer now\nMemory rests softly",
        },
        {
            "id": "train-process-2",
            "language": "en",
            "observer": "process",
            "provenance": "repo-local",
            "split": "train",
            "lines": ["System steady", "Compile tasks at dawn", "Local shells answer"],
            "text": "System steady\nCompile tasks at dawn\nLocal shells answer",
        },
    ]
    with dataset.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    model, metrics = train_model(dataset, order=4)
    allowed_after_complete_word = _allowed_tokens(model, [L1_TOKEN, *list("Compile")], 0, len("Compile"))
    allowed_mid_word = _allowed_tokens(model, [L1_TOKEN, *list("Compile stea")], 0, len("Compile stea"))

    assert metrics["word_vocabulary_size"] >= 8
    assert " " in allowed_after_complete_word
    assert "s" not in allowed_after_complete_word
    assert "d" in allowed_mid_word
    assert " " not in allowed_mid_word
    assert rejection_reason(
        model,
        "Compile compilesysteady\nProcess logs answer now\nMemory rests softly",
    ) == "unknown_word_fragment"
