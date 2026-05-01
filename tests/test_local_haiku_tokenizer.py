"""Tests for local haiku character/control-token examples."""

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_tokenizer import (  # noqa: E402
    END_TOKEN,
    L1_TOKEN,
    L2_TOKEN,
    L3_TOKEN,
    build_training_examples,
    decode_tokens,
    encode_record,
    main,
    tokenizer_report,
    training_examples,
)


def _record(split: str = "train") -> dict[str, object]:
    return {
        "id": f"sample-{split}",
        "language": "en",
        "observer": "disk",
        "provenance": "repo-local",
        "split": split,
        "lines": [
            "Root drive softly",
            "Local logs count quiet bytes",
            "Morning shells answer",
        ],
        "text": "Root drive softly\nLocal logs count quiet bytes\nMorning shells answer",
    }


def test_encode_decode_preserves_three_line_structure():
    tokens = encode_record(_record(), prompt="disk pressure")
    decoded = decode_tokens(tokens)

    assert decoded.lines == tuple(_record()["lines"])
    assert decoded.text == _record()["text"]
    assert decoded.language == "en"
    assert decoded.observer == "disk"
    assert decoded.provenance == "repo-local"
    assert decoded.prompt == "disk pressure"


def test_line_boundaries_are_explicit_control_tokens_without_newline_chars():
    tokens = encode_record(_record())

    assert tokens.count(L1_TOKEN) == 1
    assert tokens.count(L2_TOKEN) == 1
    assert tokens.count(L3_TOKEN) == 1
    assert tokens[-1] == END_TOKEN
    assert "\n" not in tokens
    assert tokens.index(L1_TOKEN) < tokens.index(L2_TOKEN) < tokens.index(L3_TOKEN) < tokens.index(END_TOKEN)


def test_training_examples_default_to_train_split_only():
    examples = training_examples([_record("train"), _record("dev"), _record("test")])

    assert [example["id"] for example in examples] == ["sample-train"]
    assert examples[0]["split"] == "train"
    assert len(examples[0]["token_ids"]) == len(examples[0]["tokens"])


def test_build_training_examples_skips_invalid_train_records():
    invalid = {
        "id": "bad-train",
        "split": "train",
        "language": "en",
        "observer": "disk",
        "provenance": "repo-local",
        "lines": ["Only one line"],
        "text": "Only one line",
    }

    build = build_training_examples([_record("train"), invalid, _record("dev")])

    assert [example["id"] for example in build.examples] == ["sample-train"]
    assert build.skipped_invalid_count == 1


def test_report_counts_vocab_and_control_tokens():
    examples = training_examples([_record("train")])
    report = tokenizer_report(examples)

    assert report["example_count"] == 1
    assert report["vocabulary_size"] >= report["control_token_size"]
    assert "<LANG=en>" in report["control_tokens"]
    assert report["deferred"] == "k-mer/de Bruijn graph representation is deferred to a later experiment"


def test_cli_generates_examples_and_report_from_dataset(tmp_path: Path, capsys):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"id":"sample-train","language":"en","observer":"disk","provenance":"repo-local","split":"train","lines":["Root drive softly","Local logs count quiet bytes","Morning shells answer"],"text":"Root drive softly\\nLocal logs count quiet bytes\\nMorning shells answer"}',
                '{"id":"sample-dev","language":"en","observer":"network","provenance":"repo-local","split":"dev","lines":["Packets cross dusk","Loopback hums under glass","Ports sleep in moonlight"],"text":"Packets cross dusk\\nLoopback hums under glass\\nPorts sleep in moonlight"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "examples.jsonl"
    report = tmp_path / "report.json"

    exit_code = main(["--dataset", str(dataset), "--out", str(out), "--report", str(report)])

    assert exit_code == 0
    assert out.read_text(encoding="utf-8").count("\n") == 1
    assert '"sample-dev"' not in out.read_text(encoding="utf-8")
    assert '"vocabulary_size"' in report.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "control_token_size=" in captured.out
