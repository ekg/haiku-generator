"""Focused tests for local haiku checker fixtures."""

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from haiku_eval import (
    evaluate_haiku,
    evaluate_samples,
    load_samples,
    load_train_poems,
    read_poem_file,
    render_report,
    write_metrics_json,
    write_report,
)


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
        GENERATED / "malformed_fragments.txt",
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


def test_lexical_coherence_flags_malformed_fragments():
    malformed = evaluate_haiku(fixture_text("malformed_fragments.txt"))

    assert "lexical_coherence" in malformed.failures
    assert malformed.details["lexical_coherence"]["artifact_markers"] == ["@@"]
    assert malformed.details["lexical_coherence"]["fused_fragments"] == [
        "localhostlatency"
    ]


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


def test_sample_jsonl_harness_reports_run_level_metrics():
    samples = load_samples(FIXTURES / "samples.jsonl")
    result = evaluate_samples(samples, train_poems=train_poems())

    assert result.metrics["sample_count"] == 5
    assert result.metrics["failure_counts"]["prompt_topic_overlap"] == 1
    assert result.metrics["failure_counts"]["train_exact_poem"] == 1
    assert result.metrics["failure_counts"]["novelty_exact_duplicate"] == 1
    assert result.metrics["warning_counts"]["train_two_line_overlap"] == 1
    assert result.metrics["topic_overlap"]["prompted_count"] == 5

    duplicate = [
        sample for sample in result.samples if sample.sample.id == "duplicate-generated"
    ][0]
    assert "nearest_prior_similarity" in duplicate.check.details


def test_dataset_jsonl_train_artifact_is_consumed_for_overlap():
    dataset_train = load_train_poems(FIXTURES / "dataset.jsonl")
    samples = load_samples(FIXTURES / "dataset_overlap_sample.jsonl")
    result = evaluate_samples(samples, train_poems=dataset_train)

    assert len(dataset_train) == 1
    assert result.metrics["failure_counts"] == {"train_exact_poem": 1}


def test_plain_text_samples_can_use_prompt_file_and_render_report():
    prompts = [
        line.strip()
        for line in (FIXTURES / "prompts" / "smoke.txt").read_text().splitlines()
    ]
    samples = load_samples(FIXTURES / "plain_samples.txt", prompts=prompts)
    result = evaluate_samples(samples)
    report = render_report(result)

    assert len(samples) == 2
    assert samples[0].prompt == "localhost latency"
    assert result.metrics["failure_counts"]["prompt_topic_overlap"] == 1
    assert "# Local Haiku Evaluation Report" in report


def test_reports_include_lexical_coherence_diagnostics(tmp_path):
    samples = load_samples(GENERATED / "malformed_fragments.txt")
    result = evaluate_samples(samples)
    json_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.md"

    write_metrics_json(result, json_path)
    write_report(result, report_path)

    json_report = json_path.read_text(encoding="utf-8")
    markdown_report = report_path.read_text(encoding="utf-8")

    assert '"lexical_coherence"' in json_report
    assert '"fused_fragments"' in json_report
    assert "### Lexical Coherence" in markdown_report
    assert "- fail_count: 1" in markdown_report
