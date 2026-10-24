# Prompt 5: ML model development (PaySim)

ROLE: ML engineer.  
MODE: Planning. This is research code: reproducibility and honest reporting matter more than raw performance.

CONSTRAINTS (also copy into .agent/rules/30-ml.md):
- Global seed config; every experiment is runnable via a CLI:
  `python -m ml.src.train --model xgboost --strategy class_weight --seed 42`.
- Split chronologically on `step` (e.g. first 70% of steps train, remainder test).
  Resample TRAIN only. No test-set information in preprocessing statistics.
- Drop leakage-prone or non-generalising columns and document why: `isFlaggedFraud`
  (it is PaySim's own rule output), and raw account IDs. If you keep ID-derived
  features such as counts, compute them using past data only.
- Report the exact fraud ratio from the data in the EDA output.

PaySim specifics to handle and document:
- Fraud appears only in TRANSFER and CASH_OUT. Decide whether to train on those types
  only and justify the choice with the EDA numbers.
- Destination balance columns are zero for merchant accounts. Handle explicitly.
- Engineered features: balance deltas and balance-error features
  (old_balance_orig - amount - new_balance_orig, and the same for the destination),
  hour-of-day from step, type encoding.
- Warn me clearly: the balance-error features can make the task nearly separable.
  Report results with and without them so the comparison stays meaningful.

TASKS:
1. ml/src/load_data.py + preprocess.py (single raw-to-API-name mapping function,
   reused by the backend at inference so training and serving cannot drift).
2. imbalance_strategies.py: none, class_weight (scale_pos_weight), SMOTE-ENN.
   Full-dataset SMOTE-ENN on millions of rows is infeasible: use a documented,
   seeded majority-class subsample, applied identically in every strategy so the
   comparison stays fair.
3. train.py: Logistic Regression and Random Forest baselines, XGBoost main model.
   Optuna tuning for XGBoost optimising PR-AUC, using time-aware validation.
4. evaluate.py: Precision, Recall, F1, PR-AUC, MCC, false-positive rate, confusion
   matrix, per model x strategy; write ml/reports/metrics.json and a comparison CSV,
   plus PR-curve plots.
5. shap_stability.py: for each strategy train N seed-replicates (N=5), compute SHAP on
   a fixed test sample, and measure top-k overlap and Spearman rank correlation.
   Per-transaction reliability design: Stable if the top-k SHAP feature overlap across
   the N replicates of the deployed strategy is >= a threshold (document the threshold
   and the reasoning). Export what the backend needs to compute this at inference.
6. Save the chosen model artifacts to ml/models (gitignored) with a model card
   (ml/reports/MODEL_CARD.md): data, split, strategy, metrics, limitations.
7. Unit tests on a tiny synthetic sample (no real data needed in CI).

ACCEPTANCE: one command reproduces the full results table; no leakage; limitations
section written. Flag any result that looks suspiciously good.
