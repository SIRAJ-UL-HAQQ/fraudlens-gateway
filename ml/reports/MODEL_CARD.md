# Model Card: FraudLens Gateway XGBoost Model

## Model Details
- **Architecture**: Gradient Boosted Decision Trees (`xgboost.XGBClassifier`) wrapped in an end-to-end scikit-learn Pipeline with `PaySimFeaturePreprocessor`.
- **Framework & Libraries**: Python 3.13.5, `scikit-learn==1.9.0`, `xgboost==3.2.0`, `joblib==1.5.3`.
- **Primary Task**: Transaction fraud risk scoring and routing (Low Risk | High Risk | Analyst Review) under extreme class imbalance.

## Dataset & Training Data
- **Data Source**: Synthetic Financial Datasets For Fraud Detection (PaySim), located at `ml/data/raw/paysim.csv`.
- **Dataset Scale**: 6,362,620 transactions total, 8,213 fraudulent transactions (0.1291% prevalence).
- **Partitioning**: Strict chronological split based on the temporal transaction `step` column:
  - **Train Split** (steps 1–520): 6,082,007 rows, 5,781 frauds (0.0951% fraud prevalence). Subsampled 10:1 (negative-to-positive ratio) yielding 63,591 training instances (5,781 frauds + 57,810 non-frauds).
  - **Validation Split** (steps 521–595): 162,467 rows, 790 frauds (0.4863% fraud prevalence). Used strictly for Optuna hyperparameter tuning, threshold calibration, and validation importance.
  - **Test Split** (steps 596–743): 118,146 rows, 1,642 frauds (1.3898% fraud prevalence). Held-out test slice evaluated strictly once after model selection.

## Decision Record
- **Deployed Feature Variant**: **Variant B (reduced-shortcut)** with 11 features.
  - Features: `amount`, `old_balance_orig`, `new_balance_orig`, `old_balance_dest`, `new_balance_dest`, `hour_of_day`, and one-hot transaction types (`type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`).
  - Rationale: In Variant A (15 features), `balance_error_orig` (< 1e-4 for 99.45% of PaySim frauds) and `balance_delta_orig` create severe collinearity and shortcut learning ($\text{PR-AUC} = 0.9998$). Variant B excludes balance errors and deltas to promote robust learning ($\text{Validation PR-AUC} = 0.9806$).
  - **Variant A Status**: Evaluated strictly as an unsaved reference benchmark.
- **Deployed Imbalance Strategy**: **`class_weight`** (`scale_pos_weight` set dynamically to negative/positive ratio on the subsampled train set: $57810 / 5781 = 10.0$).
  - **`none` Strategy**: Labeled as "undersampled 10:1 baseline".
  - **`smote_enn` Strategy**: Resampled via `imblearn.combine.SMOTEENN` inside normalized feature space.

## Feature Importance & Shortcut Analysis (Validation Slice)
Validation evaluation of Variant B reveals that model discriminative power is concentrated in the origin account balance dynamics:

| Feature | Gain Importance (Normalized) | Permutation Importance (PR-AUC Drop) |
| :--- | :--- | :--- |
| `old_balance_orig` | 6.24% | **+0.8586** $\pm$ 0.0112 |
| `new_balance_orig` | **47.12%** | **+0.4751** $\pm$ 0.0096 |
| `amount` | 1.98% | **+0.2796** $\pm$ 0.0029 |
| `type_PAYMENT` | 25.53% | **+0.0924** $\pm$ 0.0046 |
| `new_balance_dest` | 0.85% | **+0.0718** $\pm$ 0.0027 |
| `type_CASH_OUT` | 3.98% | **+0.0275** $\pm$ 0.0015 |
| `old_balance_dest` | 0.67% | **+0.0234** $\pm$ 0.0016 |
| `hour_of_day` | 1.29% | **+0.0209** $\pm$ 0.0016 |
| `type_TRANSFER` | 3.96% | **+0.0160** $\pm$ 0.0007 |
| `type_CASH_IN` | 7.81% | **+0.0013** $\pm$ 0.0005 |
| `type_DEBIT` | 0.58% | **+0.000005** $\pm$ 0.00003 |

### Key Observations:
1. **Shortcut Retention Through Raw Columns**:
   - Variant B's signal comes overwhelmingly from `old_balance_orig`, `new_balance_orig`, and `amount`—the exact three inputs of $\text{balance\_error\_orig} = \text{old\_balance\_orig} - \text{amount} - \text{new\_balance\_orig}$.
   - **Diagnostic Run**: A model trained on **only** these 3 raw features achieves a validation PR-AUC of **0.8999** (compared to 0.9806 with all 11 features). Thus, while Variant B removes the explicit linear artifact, gradient boosted trees reconstruct the drain shortcut via joint splits on the raw columns.
2. **Gain vs. Permutation Importance Divergence**:
   - Gain and permutation metrics disagree markedly on categorical transaction types (`type_*`).
   - `type_PAYMENT` accounts for 25.53% of tree split gain because it provides high immediate reduction in impurity at root splits (ruling out thousands of non-fraud payment rows), but permuting it only causes a modest +0.0924 drop in PR-AUC.
   - Conversely, permuting `old_balance_orig` causes a catastrophic +0.8586 collapse in PR-AUC despite accounting for only 6.24% of split gain, showing that origin balance precision is essential to separate genuine transfers from account drains.

## Decision Thresholds & Gateway Routing
- **Score Definition**: The model output is an uncalibrated ranking score in [0.0, 1.0], not a calibrated posterior probability.
- **Deterministic Gateway Rule**:
  - `review` threshold: Largest $t \in \text{unique\_validation\_scores}$ such that $\text{Recall}(t) \ge 0.95$.
  - `high` threshold: Smallest $t \in \text{unique\_validation\_scores}$ with $t > \text{review}$ such that $\text{Precision}(t) \ge 0.90$ and $\text{Predicted Positives}(t) \ge 20$.
- **Validation Calibration (Variant B)**:
  - $t_{\text{review}} = 0.9557$
  - $t_{\text{high}} = 0.9712$
  - Validation Routing Breakdown: Low Risk: 99.45% (161,579 / 162,467), Analyst Review: 0.04% (70 / 162,467), High Risk: 0.50% (818 / 162,467).
  - Validation High Risk Precision: 91.56% (749 frauds / 818 alerts). Validation Combined Recall (Review + High): 95.06% (751 / 790 frauds).

## Explanation Stability
- SHAP explanation stability is evaluated using replicate models differing in random initialization seeds and subsampled majority draws.
- Stability arithmetic follows the SRS specification:
  $$\text{overlap\_count} \ge \left\lceil \text{min\_overlap} \times k - 10^{-9} \right\rceil$$
  with defaults $k = 5$ (`shap_top_k`), $\text{min\_overlap} = 0.70$ (`stability_min_overlap`).
- Per project rules, final stability claims will be reported only after Stage 2 empirical evaluation.

## Limitations & Operational Constraints
1. **Validation Re-use**: Both Optuna hyperparameter optimization and threshold calibration were determined on the same temporal validation split.
2. **Uncalibrated Output**: Risk scores reflect ranking confidence; they must not be interpreted as empirical fraud probabilities.
3. **Hyperparameter Freezing**: Hyperparameters were tuned on the `class_weight` strategy and frozen across all strategies (`none`, `class_weight`, `smote_enn`).
4. **Provisional Preprocessor**: Feature engineering implemented in `ml/src/features.py` is provisional until the data teammate's exploratory data analysis is finalized.
5. **Deployment Dependency**: The backend runtime container requires `ml/__init__.py` and `ml/src/features.py` with `PYTHONPATH` set so `PaySimFeaturePreprocessor` can be unpickled cleanly by `joblib.load()`.
6. **Strategy Comparison Schema Note**: For documentation and reporting, additional strategy comparisons (such as Logistic Regression and Random Forest baselines, confusion matrices, and PR curves) are structured inside `strategy_comparison`. Proposed schema documentation addition for `docs/ml-artifact-contract.md`:
   - `strategy_comparison`: Dictionary mapping strategy names (`none`, `class_weight`, `smote_enn`, `logistic_regression`, `random_forest`, `rules_baseline`) to their validation and test evaluation metrics, confusion matrices, and precision-recall curves.
