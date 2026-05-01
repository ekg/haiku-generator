#!/usr/bin/env python3
"""CLI wrapper for training the local CPU tiny-GRU haiku model."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_haiku_neural import train_main


if __name__ == "__main__":
    raise SystemExit(train_main())
