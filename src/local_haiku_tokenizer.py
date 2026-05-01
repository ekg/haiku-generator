"""Character/control-token representation for local haiku training.

This is the baseline tokenizer from ``reports/local-haiku-implementation-plan.md``:
it keeps poem text as dense character transitions while representing haiku
boundaries, conditioning context, and line breaks as explicit control tokens.
Syllable, morphology, k-mer, and de Bruijn graph representations are deferred
because they are harder to inspect and less useful for prompt/line semantics in
the first CPU n-gram baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from urllib.parse import quote, unquote


HAIKU_TOKEN = "<HAIKU>"
PROMPT_TOKEN = "<PROMPT>"
PROMPT_END_TOKEN = "</PROMPT>"
L1_TOKEN = "<L1>"
L2_TOKEN = "<L2>"
L3_TOKEN = "<L3>"
END_TOKEN = "<END>"
BASE_CONTROL_TOKENS = (
    HAIKU_TOKEN,
    PROMPT_TOKEN,
    PROMPT_END_TOKEN,
    L1_TOKEN,
    L2_TOKEN,
    L3_TOKEN,
    END_TOKEN,
)
TRAIN_SPLIT = "train"


@dataclass(frozen=True)
class DecodedHaiku:
    """Decoded content from a token stream."""

    lines: tuple[str, str, str]
    language: str | None = None
    observer: str | None = None
    provenance: str | None = None
    prompt: str = ""

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class ExampleBuild:
    """Encoded examples plus records rejected before tokenization."""

    examples: list[dict[str, object]]
    skipped_invalid_count: int = 0


def encode_record(record: dict[str, object], *, prompt: str | None = None) -> list[str]:
    """Encode a normalized dataset record into a model-ready token stream."""
    lines = _record_lines(record)
    language = str(record.get("language") or "und")
    observer = str(record.get("observer") or "unknown")
    provenance = str(record.get("provenance") or "unknown")
    prompt_text = _default_prompt(record) if prompt is None else prompt

    tokens = [
        HAIKU_TOKEN,
        _metadata_token("LANG", language),
        _metadata_token("OBSERVER", observer),
        _metadata_token("SOURCE", provenance),
        PROMPT_TOKEN,
        *list(prompt_text),
        PROMPT_END_TOKEN,
        L1_TOKEN,
        *list(lines[0]),
        L2_TOKEN,
        *list(lines[1]),
        L3_TOKEN,
        *list(lines[2]),
        END_TOKEN,
    ]
    return tokens


def decode_tokens(tokens: Sequence[str]) -> DecodedHaiku:
    """Decode a token stream produced by :func:`encode_record`."""
    if not tokens or tokens[0] != HAIKU_TOKEN:
        raise ValueError("token stream must start with <HAIKU>")

    language: str | None = None
    observer: str | None = None
    provenance: str | None = None
    prompt_chars: list[str] = []
    line_chars: list[list[str]] = [[], [], []]
    section: str | None = None

    for token in tokens[1:]:
        if token == END_TOKEN:
            section = "end"
            break
        if token == PROMPT_TOKEN:
            section = "prompt"
            continue
        if token == PROMPT_END_TOKEN:
            section = None
            continue
        if token == L1_TOKEN:
            section = "l1"
            continue
        if token == L2_TOKEN:
            section = "l2"
            continue
        if token == L3_TOKEN:
            section = "l3"
            continue

        metadata = _parse_metadata_token(token)
        if metadata and section is None:
            key, value = metadata
            if key == "LANG":
                language = value
            elif key == "OBSERVER":
                observer = value
            elif key == "SOURCE":
                provenance = value
            continue

        if section == "prompt":
            prompt_chars.append(token)
        elif section == "l1":
            line_chars[0].append(token)
        elif section == "l2":
            line_chars[1].append(token)
        elif section == "l3":
            line_chars[2].append(token)
        else:
            raise ValueError(f"unexpected token outside content section: {token!r}")

    if section != "end":
        raise ValueError("token stream must end with <END>")

    return DecodedHaiku(
        lines=("".join(line_chars[0]), "".join(line_chars[1]), "".join(line_chars[2])),
        language=language,
        observer=observer,
        provenance=provenance,
        prompt="".join(prompt_chars),
    )


def iter_dataset_records(path: str | Path) -> Iterator[dict[str, object]]:
    """Yield normalized dataset records from JSONL."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            yield record


def build_vocabulary(token_streams: Iterable[Sequence[str]]) -> dict[str, int]:
    """Build deterministic token ids from encoded streams."""
    seen: set[str] = set(BASE_CONTROL_TOKENS)
    for tokens in token_streams:
        seen.update(tokens)
    ordered = sorted(seen, key=_token_sort_key)
    return {token: index for index, token in enumerate(ordered)}


def training_examples(
    records: Iterable[dict[str, object]],
    *,
    allowed_splits: set[str] | None = None,
    prompt: str | None = None,
) -> list[dict[str, object]]:
    """Return encoded examples, filtering out held-out splits by default."""
    return build_training_examples(records, allowed_splits=allowed_splits, prompt=prompt).examples


def build_training_examples(
    records: Iterable[dict[str, object]],
    *,
    allowed_splits: set[str] | None = None,
    prompt: str | None = None,
) -> ExampleBuild:
    """Return encoded examples and skip malformed non-three-line records."""
    splits = {TRAIN_SPLIT} if allowed_splits is None else set(allowed_splits)
    examples: list[dict[str, object]] = []
    streams: list[list[str]] = []
    skipped_invalid_count = 0
    for record in records:
        if str(record.get("split")) not in splits:
            continue
        try:
            tokens = encode_record(record, prompt=prompt)
        except ValueError:
            skipped_invalid_count += 1
            continue
        streams.append(tokens)
        examples.append(
            {
                "id": record.get("id"),
                "split": record.get("split"),
                "tokens": tokens,
            }
        )

    vocab = build_vocabulary(streams)
    for example in examples:
        example["token_ids"] = [vocab[token] for token in example["tokens"]]
    return ExampleBuild(examples=examples, skipped_invalid_count=skipped_invalid_count)


def tokenizer_report(
    examples: Sequence[dict[str, object]], *, skipped_invalid_count: int = 0
) -> dict[str, object]:
    """Summarize example and vocabulary sizes for reproducible reporting."""
    token_streams = [example["tokens"] for example in examples]
    vocab = build_vocabulary(token_streams)  # type: ignore[arg-type]
    control_tokens = [token for token in vocab if _is_control_token(token)]
    return {
        "example_count": len(examples),
        "skipped_invalid_count": skipped_invalid_count,
        "vocabulary_size": len(vocab),
        "control_token_size": len(control_tokens),
        "control_tokens": control_tokens,
        "baseline": "character/control-token",
        "deferred": "k-mer/de Bruijn graph representation is deferred to a later experiment",
    }


def write_examples(
    examples: Sequence[dict[str, object]],
    out_path: str | Path,
    *,
    report_path: str | Path | None = None,
    skipped_invalid_count: int = 0,
) -> dict[str, object]:
    """Write training examples as JSONL and optionally a report JSON artifact."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    report = tokenizer_report(examples, skipped_invalid_count=skipped_invalid_count)
    if report_path is not None:
        report_out = Path(report_path)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build local haiku tokenizer training examples.")
    parser.add_argument("--dataset", default="data/local-haiku/dataset.jsonl", help="Normalized dataset JSONL.")
    parser.add_argument("--out", help="Write encoded train examples JSONL.")
    parser.add_argument("--report", help="Write vocabulary/control-token report JSON.")
    parser.add_argument(
        "--splits",
        default=TRAIN_SPLIT,
        help="Comma-separated permitted splits. Defaults to train only.",
    )
    parser.add_argument("--prompt", help="Override prompt text for all examples.")
    args = parser.parse_args(argv)

    allowed_splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    if not allowed_splits:
        parser.error("--splits must include at least one split")

    records = list(iter_dataset_records(args.dataset))
    build = build_training_examples(records, allowed_splits=allowed_splits, prompt=args.prompt)
    examples = build.examples
    report = tokenizer_report(examples, skipped_invalid_count=build.skipped_invalid_count)

    if args.out:
        report = write_examples(
            examples,
            args.out,
            report_path=args.report,
            skipped_invalid_count=build.skipped_invalid_count,
        )
    elif args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        f"Built {report['example_count']} examples from splits {sorted(allowed_splits)}; "
        f"skipped_invalid_count={report['skipped_invalid_count']} "
        f"vocabulary_size={report['vocabulary_size']} control_token_size={report['control_token_size']}"
    )
    print(report["deferred"])
    return 0


def _record_lines(record: dict[str, object]) -> tuple[str, str, str]:
    raw_lines = record.get("lines")
    if isinstance(raw_lines, list):
        lines = tuple(str(line) for line in raw_lines)
    else:
        text = str(record.get("text") or "")
        lines = tuple(text.split("\n"))
    if len(lines) != 3:
        raise ValueError(f"record {record.get('id')!r} must contain exactly three lines")
    return lines  # type: ignore[return-value]


def _default_prompt(record: dict[str, object]) -> str:
    observer = record.get("observer")
    return str(observer) if observer else ""


def _metadata_token(key: str, value: str) -> str:
    return f"<{key}={quote(value, safe='._~-')}>"


def _parse_metadata_token(token: str) -> tuple[str, str] | None:
    if not (token.startswith("<") and token.endswith(">") and "=" in token):
        return None
    key, encoded = token[1:-1].split("=", 1)
    return key, unquote(encoded)


def _is_control_token(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")


def _token_sort_key(token: str) -> tuple[int, str]:
    if token in BASE_CONTROL_TOKENS:
        return (0, token)
    if _is_control_token(token):
        return (1, token)
    return (2, token)


if __name__ == "__main__":
    raise SystemExit(main())
