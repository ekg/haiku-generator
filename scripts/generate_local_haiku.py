#!/usr/bin/env python3
"""CLI wrapper for generating with the local CPU-only haiku n-gram model."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_ngram import generate_main


if __name__ == "__main__":
    raise SystemExit(generate_main())
