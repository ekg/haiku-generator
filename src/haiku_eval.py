"""Local-only haiku evaluation helpers.

The checks in this module are deterministic proxies for fixture and smoke tests.
They avoid network calls and external language models by design.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


LINE_CHAR_BANDS = ((8, 32), (12, 44), (8, 32))
STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "as",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "write",
    "haiku",
    "poem",
}
TOPIC_BUCKETS = {
    "storage": {"disk", "drive", "ssd", "filesystem", "file", "bytes", "root", "path"},
    "compute": {"cpu", "core", "process", "pid", "load", "ram", "memory", "thread"},
    "network": {
        "localhost",
        "latency",
        "loopback",
        "packet",
        "packets",
        "ping",
        "network",
        "eth0",
        "wlan0",
        "ms",
    },
    "machine": {"local", "host", "daemon", "shell", "kernel", "log", "logs"},
    "nature": {"rain", "moon", "moss", "pine", "river", "snow", "wind", "dawn"},
}
POEM_TEXT_FIELDS = ("poem", "text", "output", "generated", "haiku", "completion")
POEM_LINE_FIELDS = ("lines", "poem_lines", "haiku_lines")


@dataclass(frozen=True)
class HaikuCheckResult:
    """Result from the local checker."""

    lines: tuple[str, ...]
    failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class HaikuSample:
    """One generated haiku candidate with optional prompt metadata."""

    id: str
    text: str
    prompt: str = ""
    source: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HaikuSampleResult:
    """Evaluation result for one generated sample."""

    sample: HaikuSample
    check: HaikuCheckResult

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.sample.id,
            "source": self.sample.source,
            "prompt": self.sample.prompt,
            "passed": self.check.passed,
            "failures": list(self.check.failures),
            "warnings": list(self.check.warnings),
            "lines": list(self.check.lines),
            "details": self.check.details,
        }


@dataclass(frozen=True)
class HaikuRunResult:
    """Aggregate evaluation result for a generated sample file."""

    samples: tuple[HaikuSampleResult, ...]
    metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def read_poem_file(path: str | Path) -> str:
    """Read a generated poem or .haiku file."""
    return Path(path).read_text(encoding="utf-8")


def load_train_poems(path: str | Path) -> list[str]:
    """Load train poems from one file, a directory, or a normalized dataset JSONL."""
    train_path = Path(path)
    if train_path.suffix == ".jsonl":
        return load_train_poems_from_dataset(train_path)

    if train_path.is_file():
        return [train_path.read_text(encoding="utf-8")]

    poems: list[str] = []
    for child in sorted(train_path.iterdir()):
        if child.suffix in {".haiku", ".txt"} and child.is_file():
            poems.append(child.read_text(encoding="utf-8"))
    return poems


def load_train_poems_from_dataset(path: str | Path, *, split: str = "train") -> list[str]:
    """Load poem text from a normalized dataset JSONL artifact.

    Records are expected to be dicts with a train/dev/test `split` and either a
    text field or line-list field. Records without a split are treated as train
    records so tiny fixtures can stay minimal.
    """
    poems: list[str] = []
    for record in _read_jsonl_records(Path(path)):
        record_split = str(record.get("split", split))
        if record_split != split:
            continue

        poem = _record_poem_text(record)
        if poem:
            poems.append(poem)
    return poems


def load_samples(path: str | Path, *, prompts: Sequence[str] = ()) -> list[HaikuSample]:
    """Load generated samples from JSONL or plain text.

    JSONL records may use `poem`, `text`, `output`, `haiku`, or `lines` fields.
    Plain text files are split into samples by blank-line separated blocks.
    """
    sample_path = Path(path)
    if sample_path.suffix == ".jsonl":
        return _load_jsonl_samples(sample_path, prompts=prompts)

    blocks = _text_sample_blocks(sample_path.read_text(encoding="utf-8"))
    samples: list[HaikuSample] = []
    for index, block in enumerate(blocks, start=1):
        prompt = prompts[index - 1] if index <= len(prompts) else ""
        samples.append(
            HaikuSample(
                id=f"{sample_path.stem}-{index}",
                text=block,
                prompt=prompt,
                source=str(sample_path),
            )
        )
    return samples


def evaluate_samples(
    samples: Sequence[HaikuSample],
    *,
    train_poems: Iterable[str] = (),
) -> HaikuRunResult:
    """Evaluate generated samples and return per-sample results plus metrics."""
    train_poem_list = list(train_poems)
    prior_outputs: list[str] = []
    results: list[HaikuSampleResult] = []

    for sample in samples:
        check = evaluate_haiku(
            sample.text,
            prompt=sample.prompt,
            train_poems=train_poem_list,
            prior_outputs=prior_outputs,
        )
        results.append(HaikuSampleResult(sample=sample, check=check))
        prior_outputs.append(sample.text)

    return HaikuRunResult(samples=tuple(results), metrics=_run_metrics(results))


def write_metrics_json(result: HaikuRunResult, path: str | Path) -> None:
    """Write machine-readable evaluation metrics and sample details."""
    Path(path).write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_report(result: HaikuRunResult) -> str:
    """Render a compact Markdown report for humans."""
    metrics = result.metrics
    lines = [
        "# Local Haiku Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Samples: {metrics['sample_count']}",
        f"- Passed: {metrics['passed_count']}",
        f"- Failed: {metrics['failed_count']}",
        f"- Pass rate: {metrics['pass_rate']:.3f}",
        "",
        "## Checks",
        "",
    ]

    for key in ("failure_counts", "warning_counts", "topic_overlap"):
        lines.append(f"### {key.replace('_', ' ').title()}")
        values = metrics[key]
        if values:
            for name, value in values.items():
                lines.append(f"- {name}: {value}")
        else:
            lines.append("- none")
        lines.append("")

    failed = [sample for sample in result.samples if not sample.check.passed]
    if failed:
        lines.extend(["## Failed Samples", ""])
        for sample in failed[:20]:
            lines.append(f"### {sample.sample.id}")
            if sample.sample.prompt:
                lines.append(f"- Prompt: {sample.sample.prompt}")
            lines.append(f"- Failures: {', '.join(sample.check.failures)}")
            if sample.check.warnings:
                lines.append(f"- Warnings: {', '.join(sample.check.warnings)}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(result: HaikuRunResult, path: str | Path) -> None:
    """Write a readable Markdown evaluation report."""
    Path(path).write_text(render_report(result), encoding="utf-8")


def extract_poem_lines(text: str) -> tuple[str, ...]:
    """Strip optional YAML frontmatter and return non-empty poem lines."""
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    if raw_lines and raw_lines[0].strip() == "---":
        for index, line in enumerate(raw_lines[1:], start=1):
            if line.strip() == "---":
                raw_lines = raw_lines[index + 1 :]
                break

    return tuple(line.strip() for line in raw_lines if line.strip())


def normalize_poem(text: str) -> str:
    """Normalize poem text for duplicate and memorization checks."""
    return "\n".join(_normalize_line(line) for line in extract_poem_lines(text))


def evaluate_haiku(
    poem_text: str,
    *,
    prompt: str = "",
    train_poems: Iterable[str] = (),
    prior_outputs: Iterable[str] = (),
) -> HaikuCheckResult:
    """Run all local checker rules against one generated poem."""
    lines = extract_poem_lines(poem_text)
    failures: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {"line_count": len(lines)}

    if len(lines) != 3:
        failures.append("line_count")
    else:
        _check_length(lines, failures, warnings, details)
        _check_language(lines, failures, warnings, details)
        _check_repetition(lines, failures, warnings, details)
        _check_prompt_topic(prompt, lines, failures, warnings, details)
        _check_novelty(poem_text, prior_outputs, failures, warnings, details)
        _check_train_overlap(poem_text, train_poems, failures, warnings, details)

    return HaikuCheckResult(
        lines=lines,
        failures=tuple(failures),
        warnings=tuple(warnings),
        details=details,
    )


def prompt_keywords(prompt: str) -> set[str]:
    """Extract local prompt keywords after stopword removal."""
    return {token for token in _tokens(prompt) if token not in STOPWORDS}


def load_prompt_file(path: str | Path) -> list[str]:
    """Load non-empty prompts from a prompt split file."""
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _check_length(
    lines: tuple[str, ...],
    failures: list[str],
    warnings: list[str],
    details: dict[str, object],
) -> None:
    chars = [len(line) for line in lines]
    syllables = [_estimate_syllables(line) for line in lines]
    details["line_chars"] = chars
    details["estimated_syllables"] = syllables

    for index, (line, band) in enumerate(zip(lines, LINE_CHAR_BANDS)):
        low, high = band
        if len(line) < low or len(line) > high:
            failures.append("length_proxy")
            return

    if any(abs(actual - expected) > 2 for actual, expected in zip(syllables, (5, 7, 5))):
        warnings.append("syllable_proxy")


def _check_language(
    lines: tuple[str, ...],
    failures: list[str],
    warnings: list[str],
    details: dict[str, object],
) -> None:
    text = " ".join(lines)
    if any(marker in text for marker in ("�", "\ufffd")):
        failures.append("language_unreadable")
        return

    allowed = re.sub(r"[A-Za-z0-9_./:%' -]", "", text)
    if len(allowed) > max(2, len(text) // 10):
        failures.append("language_unreadable")
        return

    tokens = _tokens(text)
    alpha_or_technical = [
        token
        for token in tokens
        if re.search(r"[a-z]", token) or re.search(r"\d", token) or "/" in token or "_" in token
    ]
    ratio = len(alpha_or_technical) / max(1, len(tokens))
    details["language_token_ratio"] = ratio
    if ratio < 0.75:
        warnings.append("language_sanity")


def _check_repetition(
    lines: tuple[str, ...],
    failures: list[str],
    warnings: list[str],
    details: dict[str, object],
) -> None:
    normalized_lines = [_normalize_line(line) for line in lines]
    if len(set(normalized_lines)) != len(normalized_lines):
        failures.append("repeated_line")

    token_stream = _tokens(" ".join(lines))
    phrases = [
        tuple(token_stream[index : index + 3])
        for index in range(0, max(0, len(token_stream) - 2))
    ]
    if len(phrases) != len(set(phrases)):
        failures.append("repeated_phrase")

    if re.search(r"(.)\1{5,}", " ".join(lines).lower()):
        failures.append("repeated_characters")

    duplicate_ratio = 1 - (len(set(token_stream)) / max(1, len(token_stream)))
    details["duplicate_token_ratio"] = duplicate_ratio
    if duplicate_ratio >= 0.75:
        failures.append("duplicate_tokens")
    elif duplicate_ratio >= 0.55:
        warnings.append("duplicate_tokens")


def _check_prompt_topic(
    prompt: str,
    lines: tuple[str, ...],
    failures: list[str],
    warnings: list[str],
    details: dict[str, object],
) -> None:
    keywords = prompt_keywords(prompt)
    poem_tokens = set(_tokens(" ".join(lines)))
    matched_keywords = sorted(keywords & poem_tokens)
    matched_buckets = sorted(
        bucket
        for bucket, terms in TOPIC_BUCKETS.items()
        if keywords & terms and poem_tokens & terms
    )

    details["prompt_keywords"] = sorted(keywords)
    details["topic_matches"] = {
        "keywords": matched_keywords,
        "buckets": matched_buckets,
    }

    if keywords and not matched_keywords and not matched_buckets:
        failures.append("prompt_topic_overlap")


def _check_novelty(
    poem_text: str,
    prior_outputs: Iterable[str],
    failures: list[str],
    warnings: list[str],
    details: dict[str, object],
) -> None:
    normalized = normalize_poem(poem_text)
    nearest = 0.0
    for prior in prior_outputs:
        prior_normalized = normalize_poem(prior)
        if normalized == prior_normalized:
            failures.append("novelty_exact_duplicate")
            details["nearest_prior_similarity"] = 1.0
            return
        nearest = max(nearest, _char_ngram_jaccard(normalized, prior_normalized))

    details["nearest_prior_similarity"] = nearest
    if nearest >= 0.85:
        warnings.append("novelty_near_duplicate")


def _check_train_overlap(
    poem_text: str,
    train_poems: Iterable[str],
    failures: list[str],
    warnings: list[str],
    details: dict[str, object],
) -> None:
    normalized = normalize_poem(poem_text)
    normalized_lines = set(normalized.splitlines())
    nearest = 0.0
    two_line_overlap = False

    for train_poem in train_poems:
        train_normalized = normalize_poem(train_poem)
        if normalized == train_normalized:
            failures.append("train_exact_poem")
            details["nearest_train_similarity"] = 1.0
            return

        train_lines = set(train_normalized.splitlines())
        if len(normalized_lines & train_lines) >= 2:
            two_line_overlap = True

        nearest = max(nearest, _char_ngram_jaccard(normalized, train_normalized))

    details["nearest_train_similarity"] = nearest
    if two_line_overlap:
        warnings.append("train_two_line_overlap")
    if nearest >= 0.92:
        warnings.append("train_near_match")


def _estimate_syllables(text: str) -> int:
    count = 0
    for token in _tokens(text):
        if re.search(r"\d|[/_.]", token):
            count += 1
            continue

        word = re.sub(r"[^a-z]", "", token.lower())
        groups = re.findall(r"[aeiouy]+", word)
        syllables = len(groups)
        if word.endswith("e") and syllables > 1:
            syllables -= 1
        count += max(1, syllables)
    return count


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./:%']+", text.lower())


def _normalize_line(line: str) -> str:
    return " ".join(_tokens(line))


def _char_ngram_jaccard(left: str, right: str, n: int = 5) -> float:
    left_grams = _char_ngrams(left, n)
    right_grams = _char_ngrams(right, n)
    if not left_grams and not right_grams:
        return 1.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _char_ngrams(text: str, n: int) -> set[str]:
    collapsed = re.sub(r"\s+", " ", text.strip())
    if len(collapsed) <= n:
        return {collapsed} if collapsed else set()
    return {collapsed[index : index + n] for index in range(len(collapsed) - n + 1)}


def _load_jsonl_samples(path: Path, *, prompts: Sequence[str]) -> list[HaikuSample]:
    samples: list[HaikuSample] = []
    for index, record in enumerate(_read_jsonl_records(path), start=1):
        text = _record_poem_text(record)
        if not text:
            continue

        prompt = str(record.get("prompt") or "")
        if not prompt and index <= len(prompts):
            prompt = prompts[index - 1]

        sample_id = str(
            record.get("id") or record.get("sample_id") or f"{path.stem}-{index}"
        )
        samples.append(
            HaikuSample(
                id=sample_id,
                text=text,
                prompt=prompt,
                source=str(path),
                metadata=record,
            )
        )
    return samples


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(record)
    return records


def _record_poem_text(record: dict[str, object]) -> str:
    for field_name in POEM_LINE_FIELDS:
        value = record.get(field_name)
        if isinstance(value, list):
            return "\n".join(str(line) for line in value)

    for field_name in POEM_TEXT_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str):
            return value

    normalized = record.get("normalized_text")
    if isinstance(normalized, str):
        return normalized

    return ""


def _text_sample_blocks(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]


def _run_metrics(results: Sequence[HaikuSampleResult]) -> dict[str, object]:
    sample_count = len(results)
    passed_count = sum(1 for result in results if result.check.passed)
    failure_counts = Counter(
        failure for result in results for failure in result.check.failures
    )
    warning_counts = Counter(
        warning for result in results for warning in result.check.warnings
    )
    topic_pass = sum(
        1
        for result in results
        if result.sample.prompt and "prompt_topic_overlap" not in result.check.failures
    )
    topic_fail = sum(
        1
        for result in results
        if result.sample.prompt and "prompt_topic_overlap" in result.check.failures
    )

    return {
        "sample_count": sample_count,
        "passed_count": passed_count,
        "failed_count": sample_count - passed_count,
        "pass_rate": passed_count / sample_count if sample_count else 0.0,
        "failure_counts": dict(sorted(failure_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "topic_overlap": {
            "prompted_count": topic_pass + topic_fail,
            "pass_count": topic_pass,
            "fail_count": topic_fail,
        },
    }
