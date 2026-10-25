"""Documented entry point to reproduce the full FraudLens Gateway ML results table.

Executes end-to-end:
1. PaySim dataset loading and strict chronological split
2. Variant B training across 3 imbalance strategies ('class_weight', 'none', 'smote_enn')
3. Artifact packaging: model.joblib, replicates/seed_*.joblib, manifest.json
4. Artifact validation with startup self-test
5. Reference baselines: Logistic Regression, Random Forest, Rules-only, Full-Train XGBoost, Variant A
6. Held-out test split evaluation (once, after freeze)
7. SHAP explanation stability and chance baselines on test cohort
8. Generation of ml/reports/metrics.json, comparison.csv, and environment.json

Usage:
    python -m ml.src.run_all
"""

import sys
import time
from pathlib import Path

from ml.src.train import run_stage2_training


def main() -> None:
    print("=" * 70)
    print("FraudLens Gateway: End-to-End ML Pipeline Execution (Stage 2)")
    print("=" * 70)
    start_time = time.time()

    data_path = Path("ml/data/raw/paysim.csv")
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}.", file=sys.stderr)
        print("Please place paysim.csv in ml/data/raw/ before running.", file=sys.stderr)
        sys.exit(1)

    # Execute complete training, replicate serialization, and evaluation
    run_stage2_training(
        data_path=str(data_path),
        output_dir_str="ml/models",
        is_benchmark=False,
    )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"End-to-End Pipeline Execution Completed in {elapsed:.2f} seconds.")
    print("Generated Artifacts:")
    print("  - ml/models/class_weight/ (deployed strategy bundle)")
    print("  - ml/models/none/         (undersampled 10:1 baseline bundle)")
    print("  - ml/models/smote_enn/    (SMOTE-ENN strategy bundle)")
    print("Generated Reports:")
    print("  - ml/reports/metrics.json")
    print("  - ml/reports/comparison.csv")
    print("  - ml/reports/MODEL_CARD.md")
    print("  - ml/reports/environment.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
