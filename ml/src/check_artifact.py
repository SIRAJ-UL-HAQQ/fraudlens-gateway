"""Artifact validation CLI tool.

Loads an artifact bundle directly using the backend's production loader (ScoringService),
executes the startup self-test, validates replicates and manifest fields,
and exits non-zero on any violation.
Source of Truth: docs/ml-artifact-contract.md.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root and backend directory are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in [str(REPO_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.scoring import ScoringService


def check_bundle(bundle_path: Path) -> None:
    """Validate ML artifact bundle using the backend's production ScoringService.load_artifact.

    Exits non-zero with error message on violation.
    """
    if not bundle_path.exists() or not bundle_path.is_dir():
        print(f"ERROR: Artifact directory does not exist: {bundle_path}", file=sys.stderr)
        sys.exit(1)

    scoring_service = ScoringService()
    try:
        scoring_service.load_artifact(str(bundle_path))
    except (RuntimeError, ValueError, OSError, TypeError, KeyError) as e:
        print(f"ERROR: Backend ScoringService.load_artifact failed on {bundle_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"SUCCESS: Artifact bundle at '{bundle_path}' loaded and validated by backend loader.\n"
        f"  - Strategy: {scoring_service.imbalance_strategy}\n"
        f"  - Model Version: {scoring_service.model_version}\n"
        f"  - Thresholds: review={scoring_service.threshold_review:.6f}, high={scoring_service.threshold_high:.6f}\n"
        f"  - Replicates: {len(scoring_service.replicates)} loaded and validated"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ML model artifact bundle via backend loader.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Path to artifact bundle directory (e.g. ml/models/class_weight)",
    )
    args = parser.parse_args()
    check_bundle(args.bundle_dir)


if __name__ == "__main__":
    main()
