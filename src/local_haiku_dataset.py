"""Build normalized local haiku dataset records.

The dataset path is intentionally repo-local by default. External corpora can be
added later as explicit inputs with reviewed licensing, but this module does not
download or scrape anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SPLIT_BUCKETS = (80, 90)
SPLIT_NAMES = ("train", "dev", "test")


@dataclass(frozen=True)
class ParsedHaiku:
    """Parsed content from one `.haiku` file."""

    metadata: dict[str, str]
    lines: tuple[str, ...]
    quality_flags: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def parse_haiku_file(text: str) -> ParsedHaiku:
    """Parse frontmatter and poem lines from a repo-local `.haiku` payload."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized.split("\n")
    metadata: dict[str, str] = {}
    quality_flags: list[str] = []
    body_start = 0

    if raw_lines and raw_lines[0].strip() == "---":
        closing_index = _find_frontmatter_end(raw_lines)
        if closing_index is None:
            quality_flags.append("unterminated_frontmatter")
        else:
            metadata, metadata_flags = _parse_frontmatter(raw_lines[1:closing_index])
            quality_flags.extend(metadata_flags)
            body_start = closing_index + 1

    lines = _extract_poem_lines(raw_lines[body_start:])
    if len(lines) != 3:
        quality_flags.append("non_three_line")
    if not lines:
        quality_flags.append("empty_poem")

    return ParsedHaiku(
        metadata=metadata,
        lines=lines,
        quality_flags=tuple(dict.fromkeys(quality_flags)),
    )


def build_dataset(
    input_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    provenance: str = "repo-local",
    license_name: str = "unknown-repo-local",
    license_status: str = "unknown",
    language: str = "en",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build deduplicated dataset records and a manifest."""
    root = Path(repo_root)
    source_dir = Path(input_dir)
    paths = sorted(source_dir.glob("*.haiku"))
    records_by_norm: dict[str, dict[str, object]] = {}
    duplicate_sources: dict[str, list[str]] = {}
    parse_count = 0

    for path in paths:
        parse_count += 1
        parsed = parse_haiku_file(path.read_text(encoding="utf-8"))
        normalized_text = normalize_poem_text(parsed.text)
        source_path = _relative_path(path, root)
        duplicate_sources.setdefault(normalized_text, []).append(source_path)
        if normalized_text in records_by_norm:
            continue

        split = split_for_text(normalized_text)
        digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        records_by_norm[normalized_text] = {
            "id": f"local-haiku-{digest[:16]}",
            "text": parsed.text,
            "lines": list(parsed.lines),
            "source_path": source_path,
            "provenance": provenance,
            "license": license_name,
            "license_status": license_status,
            "language": language,
            "observer": parsed.metadata.get("observer"),
            "timestamp": parsed.metadata.get("timestamp"),
            "quality_flags": list(parsed.quality_flags),
            "split": split,
        }

    records = sorted(records_by_norm.values(), key=lambda record: str(record["id"]))
    duplicate_groups = {
        normalized: sources
        for normalized, sources in duplicate_sources.items()
        if len(sources) > 1
    }
    duplicate_source_count = sum(len(sources) - 1 for sources in duplicate_groups.values())
    split_counts = {name: 0 for name in SPLIT_NAMES}
    quality_counts: dict[str, int] = {}
    for record in records:
        split_counts[str(record["split"])] += 1
        for flag in record["quality_flags"]:
            quality_counts[str(flag)] = quality_counts.get(str(flag), 0) + 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": _relative_path(source_dir, root),
        "record_count": len(records),
        "source_file_count": parse_count,
        "dedupe": {
            "method": "sha256(normalized poem text)",
            "duplicate_source_count": duplicate_source_count,
            "duplicate_group_count": len(duplicate_groups),
        },
        "split_rule": {
            "method": "sha256(normalized poem text) first 8 hex chars modulo 100",
            "train": "0..79",
            "dev": "80..89",
            "test": "90..99",
        },
        "split_counts": split_counts,
        "quality_flag_counts": quality_counts,
        "external_downloads": "none; repo-local input only by default",
    }
    return records, manifest


def write_dataset(
    records: Sequence[dict[str, object]],
    manifest: dict[str, object],
    *,
    out_path: str | Path,
    manifest_path: str | Path | None = None,
    splits_path: str | Path | None = None,
) -> None:
    """Write JSONL records plus optional manifest and split summary."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    if manifest_path is not None:
        manifest_out = Path(manifest_path)
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if splits_path is not None:
        splits_out = Path(splits_path)
        splits_out.parent.mkdir(parents=True, exist_ok=True)
        splits = {
            split: [record["id"] for record in records if record["split"] == split]
            for split in SPLIT_NAMES
        }
        splits_out.write_text(
            json.dumps(splits, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def normalize_poem_text(text: str) -> str:
    """Normalize poem text for dedupe and deterministic split assignment."""
    lines = [_normalize_poem_line(line).casefold() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def split_for_text(normalized_text: str) -> str:
    """Return deterministic train/dev/test split for normalized poem text."""
    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < SPLIT_BUCKETS[0]:
        return "train"
    if bucket < SPLIT_BUCKETS[1]:
        return "dev"
    return "test"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the repo-local haiku dataset JSONL.")
    parser.add_argument("--input", default="haikus", help="Directory containing repo-local .haiku files.")
    parser.add_argument("--out", default="data/local-haiku/dataset.jsonl", help="Output JSONL path.")
    parser.add_argument("--manifest", default="data/local-haiku/manifest.json", help="Manifest JSON path.")
    parser.add_argument("--splits", default="data/local-haiku/splits.json", help="Split ID JSON path.")
    parser.add_argument("--repo-root", default=".", help="Repository root for relative source paths.")
    parser.add_argument("--provenance", default="repo-local", help="Provenance label for records.")
    parser.add_argument("--license", default="unknown-repo-local", help="License label for records.")
    parser.add_argument("--license-status", default="unknown", help="License review/status label for records.")
    parser.add_argument("--language", default="en", help="Language tag for records.")
    args = parser.parse_args(argv)

    records, manifest = build_dataset(
        args.input,
        repo_root=args.repo_root,
        provenance=args.provenance,
        license_name=args.license,
        license_status=args.license_status,
        language=args.language,
    )
    write_dataset(
        records,
        manifest,
        out_path=args.out,
        manifest_path=args.manifest,
        splits_path=args.splits,
    )
    print(
        f"Wrote {manifest['record_count']} records from {manifest['source_file_count']} source files "
        f"to {args.out}"
    )
    return 0


def _find_frontmatter_end(lines: Sequence[str]) -> int | None:
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return index
    return None


def _parse_frontmatter(lines: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    metadata: dict[str, str] = {}
    flags: list[str] = []
    active_block_key: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if active_block_key and (line.startswith(" ") or line.startswith("\t")):
            continue
        active_block_key = None
        if ":" not in line:
            flags.append("frontmatter_parse_warning")
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            flags.append("frontmatter_parse_warning")
            continue
        if value in {"|", ">"}:
            active_block_key = key
            continue
        metadata[key] = _strip_yaml_scalar_quotes(value)

    return metadata, flags


def _strip_yaml_scalar_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _extract_poem_lines(lines: Sequence[str]) -> tuple[str, ...]:
    poem_lines: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        if stripped == "---" or stripped.startswith("<!-- observation"):
            break
        started = True
        poem_lines.append(_normalize_poem_line(line))
    return tuple(poem_lines)


def _normalize_poem_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
