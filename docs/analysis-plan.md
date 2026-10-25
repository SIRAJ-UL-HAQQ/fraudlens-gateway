# FraudLens Gateway: Machine Learning Analysis & Verification Plan

**Branch**: `feature/ml-pipeline`  
**Scope**: `ml/` and `scripts/` exclusively.  
**Invariants**: No modifications to `backend/` (`rules.py`, `scoring.py`, `schemas.py`, API contract). Any backend changes identified will be documented for review rather than implemented directly.

---

## 1. Executive Summary & Objective

This analysis plan governs the rigorous evaluation and artifact generation for the FraudLens Gateway machine learning pipeline. The research evaluates whether local SHAP explanations for fraud alerts remain stable across class imbalance strategies (undersampled 10:1 baseline `none`, cost-sensitive `class_weight`, and hybrid `smote_enn`) under strict chronological data splits and operational latency constraints.

---

## 2. Experimental Protocols Fixed in Advance

### A. Stability Definition & Arithmetic Parameters
- **Primary Contract Parameters**:
  - Top features: $k = 5$ (`shap_top_k = 5`).
  - Minimum overlap threshold: $\text{stability\_min\_overlap} = 0.80$.
  - Whole-Number Invariant: $k \times \text{stability\_min\_overlap} = 5 \times 0.80 = 4.0$ is an integer.
  - Decision Criterion: A transaction explanation is classified as **`Stable`** if and only if:
    $$\text{overlap\_count} \ge \left\lceil \text{stability\_min\_overlap} \times k - 10^{-9} \right\rceil = 4$$
    where overlap count is the number of intersecting features between the main model's top-$k$ SHAP features and the replicate model's top-$k$ SHAP features.
- **Chance Baseline**:
  - Theoretical chance level for selecting $k=5$ features from $p=11$ available features is:
    $$\text{Chance Level} = \frac{k}{p} = \frac{5}{11} \approx 45.45\%$$
  - Every reported stability table must report this baseline alongside empirical stability rates.
- **Secondary Measures**:
  - Full distribution of mean overlap across replicates (mean, median, IQR, min, max).
  - Spearman rank correlation of mean $|SHAP|$ attributions across all 11 features as an attribution-ordering metric.
- **Sensitivity Analyses (Reported as Supplementary Diagnostics Only)**:
  - Feature depth: $k \in \{3, 10\}$.
  - Overlap thresholds: $\text{min\_overlap} \in \{0.60, 1.00\}$.
- **Prior Run Disclosure**:
  - It is explicitly acknowledged that earlier exploratory Stage 2 runs exposed preliminary test-set stability figures under $k \in \{3, 5, 10\}$ and $\text{min\_overlap} = 0.70$. The updated primary threshold $\text{min\_overlap} = 0.80$ is fixed here in advance for the final pipeline execution.

---

### B. Gateway Routing Variant Selection Rule & Operational Queue Budget
- **Temporal Context**: In the PaySim dataset, each unit of `step` represents 1 simulated hour:
  - **Validation Split**: Steps $521 \dots 595$ ($75$ simulated hours, $162,467$ transactions).
  - **Test Split**: Steps $596 \dots 743$ ($148$ simulated hours, $118,146$ transactions).
- **Analyst Queue Operational Budget**:
  - Maximum allowable human review capacity: **40 Analyst Review alerts per simulated hour**.
  - Validation Split Capacity Cap: $40 \text{ alerts/hour} \times 75 \text{ hours} = \mathbf{3,000 \text{ alerts}}$.
  - Test Split Capacity Cap: $40 \text{ alerts/hour} \times 148 \text{ hours} = \mathbf{5,920 \text{ alerts}}$.
- **Routing Variants Evaluated**:
  - **Variant A (Current Backend Implementation - Tri-State Hybrid)**:
    - Candidate High Risk: $\text{score} \ge t_{\text{high}}$ OR $(\text{has\_rules} \land \text{score} \ge t_{\text{review}})$.
    - Stable Candidate High Risk $\to$ **High Risk**.
    - Unstable Candidate High Risk $\to$ **Analyst Review**.
    - Pure Rule Trigger ($\text{has\_rules} \land \text{score} < t_{\text{review}}$) $\to$ **Analyst Review**.
    - Borderline Score ($t_{\text{review}} \le \text{score} < t_{\text{high}}$ without rules) $\to$ **Analyst Review**.
    - Otherwise $\to$ **Low Risk**.
  - **Variant B (Score-Gated Rules - Rule Never Routes Alone)**:
    - Same as Variant A, EXCEPT that a rule trigger alone without sufficient ML risk cannot route to human review.
    - If $\text{has\_rules} \land \text{score} < t_{\text{review}}$: routes to **Low Risk** (the rule flag is preserved in the database audit log, but does not consume human queue capacity).
  - **Variant C (Train-Recalibrated Heuristic Rules)**:
    - Retains Variant A routing logic, but recalibrates the heuristic rule thresholds in `app.services.rules` strictly on the **TRAIN** split such that each heuristic rule triggers on at most a target cap ($\le 0.5\%$ flag rate on training data).
  - **Variant D (Pure Machine Learning Gateway - No Rules)**:
    - Deactivates heuristic rule routing entirely.
    - Candidate High Risk: $\text{score} \ge t_{\text{high}}$ (Stable $\to$ High Risk, Unstable $\to$ Analyst Review).
    - Review Band: $t_{\text{review}} \le \text{score} < t_{\text{high}} \to$ Analyst Review.
    - Otherwise $\to$ Low Risk.
- **Selection Decision Rule**:
  - Selected strictly on **VALIDATION SPLIT ONLY**.
  - Let $F_v$ be the combined fraud recall in $(\text{High Risk} \cup \text{Analyst Review})$ for variant $v \in \{A, B, C, D\}$.
  - Let $F^* = \max_{v} F_v$ be the best fraud recall across all candidate variants.
  - A variant is candidate-feasible if its fraud capture is within 1 percentage point of the optimum:
    $$F_v \ge F^* - 0.010$$
  - Among all candidate-feasible variants satisfying the queue budget ($\le 40$ alerts/hour), select the variant with the **smallest Analyst Review queue volume**.
  - If no variant meets the 40 alerts/hour budget, select the candidate-feasible variant that minimizes total queue volume.
  - The winning variant is frozen based on Validation results; held-out Test split metrics are computed and reported once across all four variants.

---

### C. Cross-Strategy Agreement Sampling Protocol
- **Cohort Sampling Rule**:
  - Filter all held-out test transactions whose deployed-strategy (`class_weight`) score satisfies $\text{risk\_score} \ge t_{\text{review}}$.
  - If the candidate pool exceeds 2,000 transactions, draw a deterministic random sample of exactly $N = 2,000$ transactions using seed `42`.
  - The exact same identical sample of transactions is scored and explained across all three strategies (`class_weight`, `none`, `smote_enn`).
- **Agreement Metrics**:
  - Pairwise mean top-$5$ SHAP overlap between the main models of each strategy.
  - Within-strategy seed-to-seed replicate overlap (representing the retraining noise floor).
  - Spearman rank correlation of feature attributions across all 11 features.
  - Evaluation of whether between-strategy explanation divergence exceeds the within-strategy retraining noise floor.

---

## 3. Detailed Work Breakdown

### STEP 1: Heuristic Rules Root-Cause Analysis, Vectorized Parity & Table Generation
1. **Root-Cause Investigation**:
   - Determine why earlier rule evaluation tables exhibited identical metrics between unconstrained mathematical definitions and backend rule functions.
   - Specifically verify the handling of boundary cases:
     - `old_balance_orig == 0` (drain ratio division by zero).
     - Exact equality at threshold values (`amount == 200,000.0`, `drain_ratio == 0.90`).
     - Condition `new_balance_orig == 0` in `app.services.rules.check_drain_balance`.
2. **Vectorized Parity Verification**:
   - Sample $\ge 5,000$ transactions from PaySim including all boundary conditions.
   - Assert 100% bit-exact equivalence between `ml/src/load_data.py:evaluate_rules_baseline_vectorized` and `backend.app.services.rules.evaluate_rules`.
3. **Per-Rule Metrics Table**:
   - Generate per-rule table on **TRAIN** and **VALIDATION** splits (flag rate, recall, precision, FPR, total triggers) for:
     - Large Transfer (`amount > 200,000` & `type == TRANSFER`).
     - Origin Drain (`old_balance_orig > 0`, `new_balance_orig == 0`, `amount / old_balance_orig >= 0.90`).
     - Destination Balance Inconsistency.
     - Very High Amount for Type.
     - Union of all rules.
4. **Table Automation & Grep Integrity**:
   - Regenerate all dependent figures across `ml/reports/eda.json`, `metrics.json`, `MODEL_CARD.md`, and `comparison.csv`.
   - Implement script `scripts/generate_model_card_tables.py` to populate markdown tables directly from canonical JSON reports.
   - Provide an automated test in `ml/tests/test_ml_pipeline.py` that fails if markdown tables diverge from JSON numbers.
   - Provide grep verification confirming no deprecated rule figures remain.

---

### STEP 2: Replicate Pipeline Generation & Multi-Set Stability Evaluation
1. **Model Training & Multi-Set Replicates**:
   - Train $\ge 5$ independent replicate sets with disjoint random seeds per strategy:
     - Set 1: Seeds `[101, 102, 103, 104, 105]`
     - Set 2: Seeds `[201, 202, 203, 204, 205]`
     - Set 3: Seeds `[301, 302, 303, 304, 305]`
     - Set 4: Seeds `[401, 402, 403, 404, 405]`
     - Set 5: Seeds `[501, 502, 503, 504, 505]`
   - Main model seed: `42`.
2. **Artifact Packaging & Verification**:
   - Package artifacts conforming to `docs/ml-artifact-contract.md` with updated parameters:
     - `shap_top_k: 5`
     - `stability_min_overlap: 0.80`
   - Re-run `ml/src/check_artifact.py` with startup self-test on all bundles.
3. **Statistical Reporting**:
   - Stable rate per replicate set with 95% confidence intervals across sets.
   - 1,000-sample bootstrap confidence intervals over held-out test fraud rows.
   - Chance level ($45.45\%$) displayed beside every stable rate.
   - Full distribution of mean overlap and Spearman rank correlation.
   - Remove any unsupported collinearity claims.

---

### STEP 3: Comprehensive Gateway Evaluation (`scripts/evaluate_gateway.py`)
1. **Execution Protocol**:
   - Import backend `evaluate_rules` and `route_decision` (read-only) to evaluate saved artifacts.
2. **Reporting Dimensions (Validation & Test Splits)**:
   - Decision counts and rates per 1,000 transactions.
   - Fraud counts and fraud capture rates per decision category.
   - Combined fraud recall in $(\text{High Risk} \cup \text{Analyst Review})$.
   - Decision transition flow table:
     - Candidate High Risk $\to$ High Risk (Stable).
     - Candidate High Risk $\to$ Analyst Review (Unstable).
     - Rule-only routing $\to$ Analyst Review.
     - Borderline score $\to$ Analyst Review.
3. **Four-Variant Comparison**:
   - Evaluate Variants A, B, C, and D.
   - Apply the operational queue budget rule ($\le 40$ alerts/hour) and recall threshold ($\Delta \le 1.0\text{ pp}$) on Validation.
   - Produce a documented recommendation with empirical numbers.
4. **Is `Unstable` Informative?**:
   - Evaluate candidate High Risk transactions on Validation and Test:
     - $N$, True Positives, False Positives, and Precision for `Stable` vs `Unstable` cohorts.
     - Compute Wilson score 95% confidence intervals for precision.
     - Compute AUROC of mean replicate overlap for discriminating False Positives from True Positives.
     - Negative Score-Margin Control: Compare against an equal number of transactions with the smallest score margin above $t_{\text{high}}$.
     - State plainly whether sample sizes permit conclusive statistical inference without post-hoc tuning.

---

### STEP 4: Cross-Strategy Explanation Agreement
1. **Offline Evaluation**:
   - On the cohort defined in Section 2.C ($s \ge t_{\text{review}}$, capped at 2,000 rows with seed 42), compute top-5 SHAP features from the main models of each strategy.
2. **Comparative Matrix**:
   - Pairwise mean overlap:
     - `class_weight` vs `none`
     - `class_weight` vs `smote_enn`
     - `none` vs `smote_enn`
   - Baseline: Within-strategy seed-to-seed overlap across replicate sets (retraining noise floor).
   - Compute bootstrap intervals across rows and across replicate sets.
   - Explicitly evaluate whether cross-strategy agreement is lower than within-strategy retraining noise.

---

### STEP 5: Metrics Table Layout & Model Card Documentation
1. **Primary Results Table Layout**:
   - Required Columns:
     - Model & Strategy Name
     - Training Dataset & Prevalence ($\pi_{\text{train}} = 0.09505\%$)
     - Evaluation Split & Prevalence ($\pi_{\text{split}}$)
     - Decision Threshold ($t$)
     - Precision
     - Recall (Prevalence-Independent)
     - False Positive Rate (FPR) (Prevalence-Independent)
     - F1 Score
     - Matthews Correlation Coefficient (MCC)
     - Precision-Recall AUC (PR-AUC)
     - ROC-AUC
     - Precision Reweighted to Train Prevalence: $\text{Precision}(\pi_{\text{train}}) = \frac{\text{TPR} \cdot \pi_{\text{train}}}{\text{TPR} \cdot \pi_{\text{train}} + \text{FPR} \cdot (1 - \pi_{\text{train}})}$
2. **Prevalence Ratio Investigation**:
   - Document the underlying mechanics of why test split fraud prevalence ($1.3898\%$) is $\approx 14.6\times$ higher than train split prevalence ($0.09505\%$).
   - Confirm step distribution in PaySim (steady-state simulation vs terminal bursts).
3. **Reference Model Rigor**:
   - State training data (full vs 10:1 subsample), tuning budget, and decision threshold rule for Logistic Regression, Random Forest, and Full-Train XGBoost.
   - Enforce the same validation-based threshold rule as the deployed model or flag differences.
4. **Deployed Model Rationale**:
   - Document why the 10:1 subsampled model is deployed (inference latency, retraining cost, minimal loss in PR-AUC) with full-train metrics presented side-by-side.

---

### STEP 6: Pipeline Integrity & Anti-Leakage Verification
1. **End-to-End Reproducibility**:
   - Execute `python -m ml.src.run_all` twice from a clean workspace.
   - Assert 100% bit-exact equivalence of output metrics across runs.
2. **Feature Leakage Audit**:
   - Formally verify that neither `isFraud` nor `isFlaggedFraud` exists in feature preprocessors or training matrices.
3. **Chronological Invariant**:
   - Formally verify that data splits are strictly disjoint and monotonic in step order:
     $$\max(\text{step}_{\text{train}}) = 520 < \min(\text{step}_{\text{val}}) = 521 \le \max(\text{step}_{\text{val}}) = 595 < \min(\text{step}_{\text{test}}) = 596$$
4. **Feature Set Confirmation**:
   - Confirm results for Variant A (with balance error features) and Variant B (without balance error features), explicitly identifying Variant B as the deployed feature set.

---

## 4. Next Step & Checkpoint

Per user instructions, upon committing this analysis plan to the repository on branch `feature/ml-pipeline`, all work pauses for review and explicit approval before any code or training execution begins.
