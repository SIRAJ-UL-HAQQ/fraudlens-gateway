# ML Artifact Contract & Directory Specification

This document defines the interface contract between the Machine Learning pipeline and the FastAPI backend service for FraudLens Gateway.

---

## 1. Directory Structure

Each trained model bundle is stored under `ml/models/<strategy>/` (e.g. `ml/models/class_weight/`):

```text
ml/models/<strategy>/
├── manifest.json
├── model.joblib
└── replicates/
    ├── seed_42.joblib
    ├── seed_43.joblib
    └── seed_44.joblib
```

---

## 2. Component Specifications

### 2.1. `manifest.json`
Metadata file containing training details, hyperparameter thresholds, and stability configurations:

```json
{
  "model_version": "v1.0.0-class_weight",
  "imbalance_strategy": "class_weight",
  "feature_names": [
    "step",
    "type",
    "amount",
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
    "balance_delta_orig",
    "balance_delta_dest"
  ],
  "thresholds": {
    "high": 0.70,
    "review": 0.40
  },
  "shap_top_k": 3,
  "stability_min_overlap": 0.67,
  "n_replicates": 3
}
```

#### Fields Description
- `model_version` (`str`): Semantic version or git SHA of the trained model.
- `imbalance_strategy` (`str`): Strategy applied (`"none"`, `"class_weight"`, or `"smote_enn"`).
- `feature_names` (`list[str]`): Ordered list of feature names expected or generated.
- `thresholds` (`dict`):
  - `high` (`float`): Calibrated probability at or above which a transaction is a candidate for High Risk.
  - `review` (`float`): Calibrated probability threshold for Analyst Review.
- `shap_top_k` (`int`): Number of top SHAP features to return (default: `3`).
- `stability_min_overlap` (`float`): Jaccard/overlap threshold across replicate pipelines ($\in [0.0, 1.0]$) to label an explanation as `"Stable"`.
- `n_replicates` (`int`): Number of replicate models stored in `replicates/`.

---

## 3. Sklearn Pipeline (`model.joblib`)
- Must be a serialized `sklearn.pipeline.Pipeline` or compatible estimator.
- **Self-Contained Feature Engineering**: The pipeline **must** include all necessary preprocessing, imputers, scalers, and encoders.
- **Input Contract**: The backend passes a `pandas.DataFrame` whose columns match the API snake_case schema (`step`, `type`, `amount`, `old_balance_orig`, `new_balance_orig`, `old_balance_dest`, `new_balance_dest`). The backend performs **no** feature engineering for the trained model.
- **Output Contract**: Implements `predict_proba(X)`. The positive class index (fraud = 1) is extracted as `predict_proba(X)[0][1]`.

---

## 4. Replicate Pipelines (`replicates/seed_*.joblib`)
- Contains $N$ replicate models trained with different random seeds on the same training split and strategy.
- Each replicate is a self-contained `Pipeline` with the same input contract.
- Used to compute local explanation stability: top-$k$ SHAP features are calculated for the main model and each replicate. If the overlap fraction across replicates $\ge \text{stability\_min\_overlap}$, `explanation_status` is `"Stable"`; otherwise `"Unstable"`.
- Stability is evaluated across random seed replicates for the **active strategy**, never across strategies at inference time.
