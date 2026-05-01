#!/usr/bin/env python3
"""CLI wrapper for the local hybrid n-gram/GRU haiku decoder."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_hybrid import generate_main


if __name__ == "__main__":
    raise SystemExit(generate_main())
