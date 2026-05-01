"""Local-only haiku evaluation helpers.

The checks in this module are deterministic proxies for fixture and smoke tests.
They avoid network calls and external language models by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable


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


def read_poem_file(path: str | Path) -> str:
    """Read a generated poem or .haiku file."""
    return Path(path).read_text(encoding="utf-8")


def load_train_poems(path: str | Path) -> list[str]:
    """Load train poems from one file or a directory of text/.haiku files."""
    train_path = Path(path)
    if train_path.is_file():
        return [train_path.read_text(encoding="utf-8")]

    poems: list[str] = []
    for child in sorted(train_path.iterdir()):
        if child.suffix in {".haiku", ".txt"} and child.is_file():
            poems.append(child.read_text(encoding="utf-8"))
    return poems


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
