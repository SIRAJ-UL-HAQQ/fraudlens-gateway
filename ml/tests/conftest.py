"""Pytest configuration for ML test suite.

Ensures repository root and backend directory are in sys.path so tests run
cleanly without requiring manual PYTHONPATH environment variable configuration.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
