#!/usr/bin/env python3
"""CLI wrapper for building the repo-local haiku dataset."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
