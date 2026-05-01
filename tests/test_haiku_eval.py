"""Focused tests for local haiku checker fixtures."""

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from haiku_eval import evaluate_haiku, load_train_poems, read_poem_file


FIXTURES = Path(__file__).parent / "fixtures" / "haiku_eval"
GENERATED = FIXTURES / "generated"
TRAIN = FIXTURES / "train"


def fixture_text(name: str) -> str:
    return read_poem_file(GENERATED / name)


def train_poems() -> list[str]:
    return load_train_poems(TRAIN)


def test_fixture_files_exist():
    expected = [
        FIXTURES / "prompts" / "smoke.txt",
        TRAIN / "tiny_train.haiku",
        GENERATED / "valid.haiku",
        GENERATED / "one_line.txt",
        GENERATED / "four_line.txt",
        GENERATED / "repeated_lines.txt",
        GENERATED / "train_duplicate.haiku",
        GENERATED / "two_train_lines.haiku",
        GENERATED / "topic_miss.haiku",
        GENERATED / "topic_hit.haiku",
        GENERATED / "technical_language.haiku",
    ]

    missing = [path for path in expected if not path.exists()]
    assert missing == []


def test_three_line_form_passes_and_fails():
    passing = evaluate_haiku(fixture_text("valid.haiku"))
    one_line = evaluate_haiku(fixture_text("one_line.txt"))
    four_line = evaluate_haiku(fixture_text("four_line.txt"))

    assert passing.passed
    assert passing.details["line_count"] == 3
    assert "line_count" in one_line.failures
    assert "line_count" in four_line.failures


def test_length_proxy_and_language_sanity():
    too_long = evaluate_haiku(fixture_text("too_long.txt"))
    technical = evaluate_haiku(fixture_text("technical_language.haiku"), prompt="cpu eth0 root")
    mojibake = evaluate_haiku(fixture_text("mojibake.txt"))

    assert "length_proxy" in too_long.failures
    assert technical.passed
    assert "language_unreadable" in mojibake.failures


def test_repetition_passes_and_fails():
    passing = evaluate_haiku(fixture_text("topic_hit.haiku"), prompt="localhost latency")
    repeated = evaluate_haiku(fixture_text("repeated_lines.txt"), prompt="localhost latency")

    assert passing.passed
    assert "repeated_line" in repeated.failures


def test_prompt_topic_overlap_passes_and_fails():
    hit = evaluate_haiku(fixture_text("topic_hit.haiku"), prompt="write about localhost latency")
    miss = evaluate_haiku(fixture_text("topic_miss.haiku"), prompt="write about localhost latency")

    assert hit.passed
    assert hit.details["topic_matches"]["keywords"] or hit.details["topic_matches"]["buckets"]
    assert "prompt_topic_overlap" in miss.failures


def test_novelty_exact_duplicate_fails_and_new_output_passes():
    prior = [fixture_text("topic_hit.haiku")]
    duplicate = evaluate_haiku(fixture_text("topic_hit.haiku"), prompt="localhost latency", prior_outputs=prior)
    novel = evaluate_haiku(fixture_text("valid.haiku"), prior_outputs=prior)

    assert "novelty_exact_duplicate" in duplicate.failures
    assert novel.passed


def test_train_set_overlap_exact_duplicate_fails_and_two_lines_are_flagged():
    exact = evaluate_haiku(fixture_text("train_duplicate.haiku"), train_poems=train_poems())
    partial = evaluate_haiku(fixture_text("two_train_lines.haiku"), train_poems=train_poems())
    unrelated = evaluate_haiku(fixture_text("valid.haiku"), train_poems=train_poems())

    assert "train_exact_poem" in exact.failures
    assert partial.passed
    assert "train_two_line_overlap" in partial.warnings
    assert unrelated.passed
