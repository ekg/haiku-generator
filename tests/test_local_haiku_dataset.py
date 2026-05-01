"""Tests for repo-local haiku dataset parsing and normalization."""

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_dataset import build_dataset, parse_haiku_file, split_for_text


def test_parse_frontmatter_and_strip_observations():
    payload = """---
observer: disk
timestamp: 20260326-151130
---
Drive sixty-two full
In the autohaiku folder
Seven point five megs

---
observations: |
  raw observation text
"""

    parsed = parse_haiku_file(payload)

    assert parsed.metadata["observer"] == "disk"
    assert parsed.metadata["timestamp"] == "20260326-151130"
    assert parsed.lines == (
        "Drive sixty-two full",
        "In the autohaiku folder",
        "Seven point five megs",
    )
    assert parsed.text == "Drive sixty-two full\nIn the autohaiku folder\nSeven point five megs"
    assert parsed.quality_flags == ()


def test_parse_html_observation_comment_after_first_stanza():
    payload = """---
observer: network
timestamp: 20260325-091655
---
Local IP one five six
Ping time thirty point eight
Firefox connects twice

<!-- observation data
=== IP Addresses ===
inet 127.0.0.1/8 scope host lo
-->
"""

    parsed = parse_haiku_file(payload)

    assert parsed.lines == (
        "Local IP one five six",
        "Ping time thirty point eight",
        "Firefox connects twice",
    )
    assert parsed.quality_flags == ()


def test_build_dataset_dedupes_before_deterministic_split(tmp_path: Path):
    haikus = tmp_path / "haikus"
    haikus.mkdir()
    first = """---
observer: disk
timestamp: 20260326-151130
---
Drive sixty-two full
In the autohaiku folder
Seven point five megs
"""
    duplicate = """---
observer: network
timestamp: 20260327-151130
---
  Drive   sixty-two full
In the autohaiku folder
Seven point five megs
"""
    second = """---
observer: process
timestamp: 20260328-151130
---
Kernel threads at dawn
Local logs breathe in soft loops
Quiet cores answer
"""
    (haikus / "a.haiku").write_text(first, encoding="utf-8")
    (haikus / "b.haiku").write_text(duplicate, encoding="utf-8")
    (haikus / "c.haiku").write_text(second, encoding="utf-8")

    records, manifest = build_dataset(haikus, repo_root=tmp_path)

    assert len(records) == 2
    assert manifest["source_file_count"] == 3
    assert manifest["dedupe"]["duplicate_source_count"] == 1

    record = next(item for item in records if item["observer"] == "disk")
    assert record["source_path"] == "haikus/a.haiku"
    assert record["provenance"] == "repo-local"
    assert record["license"] == "unknown-repo-local"
    assert record["license_status"] == "unknown"
    assert record["language"] == "en"
    assert record["timestamp"] == "20260326-151130"
    assert record["quality_flags"] == []
    assert record["split"] == split_for_text(record["text"].casefold())
