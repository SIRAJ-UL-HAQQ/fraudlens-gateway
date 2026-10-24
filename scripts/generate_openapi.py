#!/usr/bin/env python3
"""Regenerate docs/openapi.json from the current FastAPI application definition."""

import json
import sys
from pathlib import Path

# Ensure backend directory is in Python path for 'app.*' imports
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app


def generate_openapi() -> None:
    """Generate docs/openapi.json from the FastAPI app."""
    openapi_schema = app.openapi()
    output_path = REPO_ROOT / "docs" / "openapi.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"Successfully generated OpenAPI specification at: {output_path}")


if __name__ == "__main__":
    generate_openapi()
