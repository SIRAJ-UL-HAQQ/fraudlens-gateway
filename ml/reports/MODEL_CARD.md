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

## Explanation Stability & Empirical Findings
- **Stability Specification**: Explanations are evaluated across $R=5$ replicate models differing in random initialization seeds and subsampled majority draws.
  $$\text{overlap\_count} \ge \left\lceil \text{min\_overlap} \times k - 10^{-9} \right\rceil$$
  Defaults: $k = 5$ (`shap_top_k`), $\text{min\_overlap} = 0.70$ (`stability_min_overlap`).
- **Empirical Refutation of Collinearity Hypothesis**:
  - Measured stability on test fraud cohort ($1,642$ transactions):
    - **Variant B (11 features, reduced-shortcut)**: **96.83%** stability rate (95% CI: [81.00%, 84.53%]).
    - **Variant A (15 features, includes collinear balance errors/deltas)**: **96.22%** stability rate.
  - **Finding**: Collinearity in Variant A does **not** cause SHAP explanation instability. Tree ensembles under histogram-based binning stabilize their dominant split attributions across seeds equally well in both variants.
- **Sensitivity Across $k$**:
  - **$k=10$**: Near-vacuous with 11 total features (chance baseline $10/11 = 90.91\%$), yielding 100.0% stability because 10 of 11 features are selected by definition.
  - **$k=3$**: Strict under the ceiling rule ($\lceil 0.70 \times 3 - 10^{-9} \rceil = 3$, requiring an exact 3/3 = 100% feature match), dropping stability rate to 65.16%.
  - **$k=5$**: Operates at a balanced sensitivity point (requires $\ge 4/5 = 80\%$ overlap; chance baseline is $5/11 = 45.45\%$; stability rate is 96.83%).
- **Strategy Comparison on Test Frauds (Paired Bootstrap 95% CIs)**:
  - `class_weight`: 82.74% [81.00%, 84.53%]
  - `none`: 81.84% [79.96%, 83.68%]
  - `smote_enn`: 80.30% [78.20%, 82.28%]
  - Paired Difference $\Delta(\text{class\_weight} - \text{none})$: +0.90% [-1.77%, +3.41%] (statistically indistinguishable).
  - Paired Difference $\Delta(\text{class\_weight} - \text{smote\_enn})$: +2.44% [-0.25%, +5.00%].

## Full Gateway Decision Evaluation (Held-Out Test Set: 118,146 Transactions)
Evaluating the complete backend decision pipeline (`route_decision` with ML scores, SHAP stability, and heuristic rules):
- **Decision Breakdown**:
  - **Low Risk**: $81,370$ transactions (**68.87%**) — $4$ frauds, $81,366$ legitimate
  - **Analyst Review**: $35,379$ transactions (**29.95%**) — $335$ frauds, $35,044$ legitimate
  - **High Risk**: $1,397$ transactions (**1.18%**) — $1,303$ frauds, $94$ legitimate
- **Headline Gateway Metrics**:
  - **Combined Fraud Recall (High + Review)**: **99.76%** ($1,638 / 1,642$ frauds caught; only 4 frauds missed).
  - **Analyst Review Share of Legitimate**: **30.08%** ($35,044 / 116,504$).
- **Routing by Cause**:
  - `score_band` ($t_{\text{review}} \le s < t_{\text{high}}$ without rules): $3$ transactions ($0$ frauds, $3$ legit).
  - `unstable_explanation` (Candidate High Risk re-routed to Review): $299$ transactions ($272$ frauds, $27$ legit).
  - `rule_flag` (rules triggered with score $< t_{\text{review}}$): **$35,077$ transactions** ($63$ frauds, $35,014$ legit).
- **Flagged Problem with Heuristic Rules**:
  - The provisional heuristic rules in `app.services.rules` (specifically destination balance inconsistency and very high amount for type) swamp the review queue by routing 30.08% of all legitimate transactions to Analyst Review for a marginal gain of 63 frauds. The ML model alone is surgical (only 94 FPs at high threshold), but the uncalibrated heuristic rules create severe operational overload.
- **High Risk Rows Split by Stability**:
  - **Stable Candidate High Risk** ($N=1,397$): Precision = **93.27%** ($1,303 / 1,397$), FPR = **0.0807%** ($94 / 116,504$).
  - **Unstable Candidate High Risk** ($N=299$): Precision = **90.97%** ($272 / 299$), FPR = **0.0232%** ($27 / 116,504$).

## Reference Models & Baselines Specification
- **Logistic Regression**: Trained on 10:1 subsample ($63,591$ rows) with `RobustScaler`, default $C=1.0$, `max_iter=1000`. Threshold = 0.50. Test Precision = 67.49%; Precision at train prevalence = 12.36%; Recall = 81.55%.
- **Random Forest**: Trained on 10:1 subsample ($63,591$ rows) with `n_estimators=100`, `max_depth=8`. Threshold = 0.50. Test Precision = 85.75%; Precision at train prevalence = 29.32%; Recall = 87.21%.
- **Full-Train Reference (`class_weight`)**: Trained on all $6,082,007$ training rows with `scale_pos_weight=1051.0`, `max_depth=5`, `n_estimators=100`, `tree_method="hist"`. Test PR-AUC = 0.9989; Recall = 98.66%; Precision = 97.47% (test) / 70.36% (train prevalence).
- **Full-Train Reference (`none`)**: Trained on all $6,082,007$ training rows with `scale_pos_weight=1.0`, `max_depth=5`, `n_estimators=100`, `tree_method="hist"`. Test PR-AUC = 0.9984; Recall = 98.36%; Precision = 97.41% (test) / 69.83% (train prevalence).
- **Variant A Reference**: Trained on 10:1 subsample with 15 features (including balance errors/deltas), tuned hyperparameters. Test PR-AUC = 0.9999; Recall = 99.94%; Precision = 98.32% (test) / 82.62% (train prevalence).
- **Prevalence Adjustment Formula**:
  $$\text{Precision}(\pi_{\text{train}}) = \frac{\text{TPR} \cdot \pi_{\text{train}}}{\text{TPR} \cdot \pi_{\text{train}} + \text{FPR} \cdot (1 - \pi_{\text{train}})}$$
  where $\pi_{\text{train}} = 0.0951\%$ ($5,781 / 6,082,007$) and $\pi_{\text{test}} = 1.3898\%$ ($1,642 / 118,146$). Test prevalence is $\approx 14.6\times$ higher than the chronological training base rate.

## Heuristic Safety Rules Performance (Train & Validation Splits)

Evaluated strictly on **TRAIN** (steps 1..520) and **VALIDATION** (steps 521..595) splits; test numbers are excluded from heuristic calibration. All rules implement the exact logic in `backend.app.services.rules`.

### Train Split: Heuristic Rules Evaluation

| Rule Name | Flag Rate | Fraud Recall | Precision | FPR | Total Triggers | TP | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Origin Account Drained (>= 90%) | 18.586% | 97.80% | 0.50% | 18.510% | 1,130,386 | 5,654 | 1,124,732 |
| Large Transfer Amount (>= 200k) | 18.890% | 66.65% | 0.34% | 18.850% | 1,148,920 | 3,853 | 1,145,067 |
| Destination Balance Inconsistency | 9.014% | 56.86% | 0.60% | 8.970% | 548,236 | 3,287 | 544,949 |
| Very High Amount for Type | 11.034% | 52.98% | 0.46% | 10.990% | 671,116 | 3,063 | 668,053 |
| **Union: Drain OR Large Transfer (SRS FR-7)** | 27.949% | 99.71% | 0.34% | 27.880% | 1,699,836 | 5,764 | 1,694,072 |
| **Union: All 4 Backend Rules** | 32.310% | 99.79% | 0.29% | 32.250% | 1,965,098 | 5,769 | 1,959,329 |

### Validation Split: Heuristic Rules Evaluation

| Rule Name | Flag Rate | Fraud Recall | Precision | FPR | Total Triggers | TP | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Origin Account Drained (>= 90%) | 19.898% | 98.48% | 2.41% | 19.510% | 32,327 | 778 | 31,549 |
| Large Transfer Amount (>= 200k) | 17.282% | 65.32% | 1.84% | 17.050% | 28,078 | 516 | 27,562 |
| Destination Balance Inconsistency | 5.128% | 55.70% | 5.28% | 4.880% | 8,331 | 440 | 7,891 |
| Very High Amount for Type | 9.727% | 51.39% | 2.57% | 9.520% | 15,803 | 406 | 15,397 |
| **Union: Drain OR Large Transfer (SRS FR-7)** | 27.864% | 99.75% | 1.74% | 27.510% | 45,270 | 788 | 44,482 |
| **Union: All 4 Backend Rules** | 30.892% | 99.75% | 1.57% | 30.560% | 50,190 | 788 | 49,402 |

### Validation Split: Rule Performance Breakdown by Transaction Type

| Rule Name | Type | Volume | Frauds | Triggers | Flag Rate | Recall | Precision | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Origin Drained | TRANSFER | 14,771 | 395 | 6,170 | 41.771% | 97.47% | 6.24% | 40.241% |
| Origin Drained | CASH_OUT | 53,167 | 395 | 26,157 | 49.198% | 99.49% | 1.50% | 48.821% |
| Origin Drained | PAYMENT | 56,586 | 0 | 0 | 0.000% | - | - | 0.000% |
| Origin Drained | CASH_IN | 36,591 | 0 | 0 | 0.000% | - | - | 0.000% |
| Origin Drained | DEBIT | 1,352 | 0 | 0 | 0.000% | - | - | 0.000% |
| Large Transfer | TRANSFER | 14,771 | 395 | 10,809 | 73.177% | 65.57% | 2.40% | 73.386% |
| Large Transfer | CASH_OUT | 53,167 | 395 | 17,269 | 32.481% | 65.06% | 1.49% | 32.237% |
| Large Transfer | PAYMENT | 56,586 | 0 | 0 | 0.000% | - | - | 0.000% |
| Large Transfer | CASH_IN | 36,591 | 0 | 0 | 0.000% | - | - | 0.000% |
| Large Transfer | DEBIT | 1,352 | 0 | 0 | 0.000% | - | - | 0.000% |
| Dest Inconsistency | TRANSFER | 14,771 | 395 | 2,237 | 15.144% | 100.00% | 17.66% | 12.813% |
| Dest Inconsistency | CASH_OUT | 53,167 | 395 | 6,094 | 11.462% | 11.39% | 0.74% | 11.463% |
| Dest Inconsistency | PAYMENT | 56,586 | 0 | 0 | 0.000% | - | - | 0.000% |
| Dest Inconsistency | CASH_IN | 36,591 | 0 | 0 | 0.000% | - | - | 0.000% |
| Dest Inconsistency | DEBIT | 1,352 | 0 | 0 | 0.000% | - | - | 0.000% |
| Very High Amount | TRANSFER | 14,771 | 395 | 6,394 | 43.288% | 45.82% | 2.83% | 43.218% |
| Very High Amount | CASH_OUT | 53,167 | 395 | 7,209 | 13.559% | 56.96% | 3.12% | 13.234% |
| Very High Amount | PAYMENT | 56,586 | 0 | 1,485 | 2.624% | - | 0.00% | 2.624% |
| Very High Amount | CASH_IN | 36,591 | 0 | 640 | 1.749% | - | 0.00% | 1.749% |
| Very High Amount | DEBIT | 1,352 | 0 | 75 | 5.547% | - | 0.00% | 5.547% |
| Union (FR-7) | TRANSFER | 14,771 | 395 | 12,041 | 81.518% | 100.00% | 3.28% | 81.010% |
| Union (FR-7) | CASH_OUT | 53,167 | 395 | 33,229 | 62.499% | 99.49% | 1.18% | 62.222% |
| Union (FR-7) | PAYMENT | 56,586 | 0 | 0 | 0.000% | - | - | 0.000% |
| Union (FR-7) | CASH_IN | 36,591 | 0 | 0 | 0.000% | - | - | 0.000% |
| Union (FR-7) | DEBIT | 1,352 | 0 | 0 | 0.000% | - | - | 0.000% |
| Union (All Rules) | TRANSFER | 14,771 | 395 | 12,384 | 83.840% | 100.00% | 3.19% | 83.396% |
| Union (All Rules) | CASH_OUT | 53,167 | 395 | 35,606 | 66.970% | 99.49% | 1.10% | 66.727% |
| Union (All Rules) | PAYMENT | 56,586 | 0 | 1,485 | 2.624% | - | 0.00% | 2.624% |
| Union (All Rules) | CASH_IN | 36,591 | 0 | 640 | 1.749% | - | 0.00% | 1.749% |
| Union (All Rules) | DEBIT | 1,352 | 0 | 75 | 5.547% | - | 0.00% | 5.547% |
## Limitations & Operational Constraints
1. **Validation Re-use**: Both Optuna hyperparameter optimization and threshold calibration were determined on the same temporal validation split.
2. **Uncalibrated Output**: Risk scores reflect ranking confidence; they must not be interpreted as empirical fraud probabilities.
3. **Hyperparameter Freezing**: Hyperparameters were tuned on the `class_weight` strategy and frozen across all strategies (`none`, `class_weight`, `smote_enn`).
4. **Provisional Preprocessor**: Feature engineering implemented in `ml/src/features.py` is provisional until the data teammate's exploratory data analysis is finalized.
5. **Deployment Dependency**: The backend runtime container requires `ml/__init__.py` and `ml/src/features.py` with `PYTHONPATH` set so `PaySimFeaturePreprocessor` can be unpickled cleanly by `joblib.load()`.
6. **Strategy Comparison Schema Note**: For documentation and reporting, additional strategy comparisons (such as Logistic Regression and Random Forest baselines, confusion matrices, and PR curves) are structured inside `strategy_comparison`. Proposed schema documentation addition for `docs/ml-artifact-contract.md`:
   - `strategy_comparison`: Dictionary mapping strategy names (`none`, `class_weight`, `smote_enn`, `logistic_regression`, `random_forest`, `rules_baseline`) to their validation and test evaluation metrics, confusion matrices, and precision-recall curves.
