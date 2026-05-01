#!/usr/bin/env python3
"""CLI wrapper for building local haiku tokenizer training examples."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_tokenizer import main


if __name__ == "__main__":
    raise SystemExit(main())
