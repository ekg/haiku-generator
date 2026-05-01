"""Tests for the local n-gram graph plus tiny-GRU hybrid decoder."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import importlib.util


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_hybrid import _candidate_quality_penalty, generate_haiku, generate_main  # noqa: E402
from local_haiku_neural import load_model as load_neural_model  # noqa: E402
from local_haiku_neural import save_model as save_neural_model  # noqa: E402
from local_haiku_neural import train_model as train_neural_model  # noqa: E402
from local_haiku_ngram import load_model as load_ngram_model  # noqa: E402
from local_haiku_ngram import rejection_reason, save_model as save_ngram_model  # noqa: E402
from local_haiku_ngram import train_model as train_ngram_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _load_quality_pass_runner():
    module_path = ROOT / "scripts" / "run_local_haiku_hybrid_quality_pass.py"
    spec = importlib.util.spec_from_file_location("run_local_haiku_hybrid_quality_pass", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _write_models(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset.jsonl"
    ngram_path = tmp_path / "ngram.json.gz"
    neural_checkpoint = tmp_path / "neural.npz"
    neural_metadata = tmp_path / "neural.json"
    _write_fixture_dataset(dataset)

    ngram_model, _ngram_metrics = train_ngram_model(dataset, order=4)
    neural_model, _neural_metrics = train_neural_model(
        dataset,
        embedding_dim=8,
        hidden_size=12,
        epochs=1,
        learning_rate=0.02,
        seed=3,
    )
    save_ngram_model(ngram_model, ngram_path)
    save_neural_model(neural_model, neural_checkpoint, neural_metadata)
    return ngram_path, neural_metadata


def test_hybrid_generates_valid_graph_constrained_haiku(tmp_path: Path):
    ngram_path, neural_metadata = _write_models(tmp_path)
    ngram_model = load_ngram_model(ngram_path)
    neural_model = load_neural_model(neural_metadata)

    poem, metadata = generate_haiku(
        ngram_model,
        neural_model,
        prompt="disk pressure",
        seed=7,
        neural_weight=0.4,
        top_k=5,
    )

    assert len(poem.splitlines()) == 3
    assert rejection_reason(ngram_model, poem) is None
    assert metadata["decoder"] == "ngram-graph-neural-rerank"
    assert metadata["neural_weight"] == 0.4
    assert metadata["accepted_candidates"] >= 1
    assert metadata["candidate_pool"] == 2


def test_hybrid_cli_writes_sample(tmp_path: Path, capsys):
    ngram_path, neural_metadata = _write_models(tmp_path)
    samples = tmp_path / "samples.jsonl"

    exit_code = generate_main(
        [
            "--ngram-model",
            str(ngram_path),
            "--neural-model",
            str(neural_metadata),
            "--prompt",
            "disk pressure",
            "--seed",
            "11",
            "--neural-weight",
            "0.5",
            "--samples-out",
            str(samples),
        ]
    )
    generated = capsys.readouterr().out.strip()
    assert exit_code == 0
    sample_payload = json.loads(samples.read_text(encoding="utf-8"))

    assert len(generated.splitlines()) == 3
    assert len(sample_payload["lines"]) == 3
    assert sample_payload["metadata"]["decoder"] == "ngram-graph-neural-rerank"
    assert sample_payload["metadata"]["top_k"] == 8
    assert sample_payload["metadata"]["candidate_pool"] == 2


def test_hybrid_candidate_pool_must_be_positive(tmp_path: Path):
    ngram_path, neural_metadata = _write_models(tmp_path)
    ngram_model = load_ngram_model(ngram_path)
    neural_model = load_neural_model(neural_metadata)

    try:
        generate_haiku(ngram_model, neural_model, prompt="disk pressure", candidate_pool=0)
    except ValueError as exc:
        assert "candidate_pool" in str(exc)
    else:
        raise AssertionError("candidate_pool=0 should be rejected")


def test_candidate_quality_penalty_prefers_complete_clean_lines():
    clean = "Latency load climbs\nLocal packets answer softly\nMorning shells answer"
    rough = "Latency to in megs awake now\nForty-eight eight memory two m\nTwo server graph the root t"

    assert _candidate_quality_penalty(clean) < _candidate_quality_penalty(rough)


def test_quality_pass_report_recommends_tuning_when_hybrid_beats_baselines():
    runner = _load_quality_pass_runner()
    summary = {
        "models": {
            "ngram": {"pass_rate": 1.0, "mean_quality_penalty": 3.5, "mean_attempt": 2.0, "errors": 0},
            "neural": {"pass_rate": 1.0, "mean_quality_penalty": 7.8, "mean_attempt": 1.0, "errors": 0},
            "hybrid_w0.0_temp0.9_top8": {
                "pass_rate": 1.0,
                "mean_quality_penalty": 1.5,
                "mean_attempt": 3.0,
                "errors": 0,
            },
            "hybrid_w1.0_temp0.75_top5": {
                "pass_rate": 1.0,
                "mean_quality_penalty": 5.3,
                "mean_attempt": 22.8,
                "errors": 0,
            },
        }
    }

    report = runner._render_report(summary)

    assert "Use graph-first hybrid defaults only; retrain before enabling neural-weighted reranking." in report
    assert "`hybrid_w0.0_temp0.9_top8`" in report
    assert "Do not use the tiny-GRU reranker as a quality-improving default" in report
