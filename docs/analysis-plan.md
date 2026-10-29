# FraudLens Gateway: Machine Learning Analysis & Verification Plan

**Branch**: `feature/ml-pipeline`  
**Scope**: `ml/` and `scripts/` exclusively.  
**Invariants**: No modifications to `backend/` (`rules.py`, `scoring.py`, `schemas.py`, API contract). Any backend changes identified will be documented for review rather than implemented directly.

---

## 1. Executive Summary & Objective

This analysis plan governs the rigorous evaluation, artifact generation, and statistical reporting for the FraudLens Gateway machine learning pipeline. The research evaluates whether local SHAP explanations for fraud alerts remain stable across class imbalance strategies (undersampled 10:1 baseline `none`, cost-sensitive `class_weight`, and hybrid `smote_enn`) under strict chronological data splits and operational latency constraints.

---

## 2. Experimental Protocols Fixed in Advance

### A. Stability Definition, Threshold Equivalence & Sensitivity Parameters
- **Primary Contract Parameters**:
  - Top features: $k = 5$ (`shap_top_k = 5`).
  - Minimum overlap threshold: $\text{stability\_min\_overlap} = 0.80$.
  - Whole-Number Arithmetic: $k \times \text{stability\_min\_overlap} = 5 \times 0.80 = 4.0$ is an integer.
  - Decision Criterion: A transaction explanation is classified as **`Stable`** if and only if:
    $$\text{overlap\_count} \ge \left\lceil \text{stability\_min\_overlap} \times k - 10^{-9} \right\rceil = 4$$
    where overlap count is the number of intersecting features between the main model's top-$k$ SHAP features and the replicate model's top-$k$ SHAP features.
- **Mathematical Equivalence & SRS Parameter Deviation**:
  - The SRS specification text originally noted $\text{stability\_min\_overlap} = 0.70$.
  - Under integer ceiling arithmetic with $k=5$:
    $$\left\lceil 0.80 \times 5 - 10^{-9} \right\rceil = \lceil 4.0 - 10^{-9} \rceil = 4 \quad (4 \text{ of } 5)$$
    $$\left\lceil 0.70 \times 5 - 10^{-9} \right\rceil = \lceil 3.5 - 10^{-9} \rceil = 4 \quad (4 \text{ of } 5)$$
  - Setting $\text{min\_overlap} = 0.80$ in `manifest.json` yields the exact same integer threshold of **4 of 5** features as the SRS $0.70$ parameter, while ensuring that $k \times \text{min\_overlap}$ is an exact whole integer without fractional remainder. This deviation from the literal $0.70$ string in the SRS is documented here.
- **Chance Baseline & Constant-Explanation Baseline**:
  - **Theoretical Chance Level**: Selecting $k=5$ features randomly from $p=11$ features:
    $$\text{Chance Level} = \frac{k}{p} = \frac{5}{11} \approx 45.45\%$$
    Reported beside every empirical stability rate.
  - **Constant-Explanation Baseline**: A naive static explainer predicting the global top-5 features ranked by overall mean $|SHAP|$ attribution across the training split.
- **Sensitivity Analyses (Reported as Supplementary Diagnostics Only)**:
  - Overlap sensitivity at $k=5$:
    - $\text{min\_overlap} = 0.60 \implies \lceil 3.0 - 10^{-9} \rceil = 3 \text{ of } 5$ ($60\%$).
    - $\text{min\_overlap} = 1.00 \implies \lceil 5.0 - 10^{-9} \rceil = 5 \text{ of } 5$ ($100\%$).
  - Feature depth sensitivity: $k=3$ (chance level $3/11 \approx 27.27\%$) and $k=10$ (chance level $10/11 \approx 90.91\%$).
- **Prior Run Disclosure**:
  - It is explicitly disclosed that earlier exploratory Stage 2 runs exposed test-set stability numbers under $k \in \{3, 5, 10\}$ and $\text{min\_overlap} = 0.70$. The parameters fixed here will be strictly evaluated under the frozen multi-set protocol.

---

### B. Gateway Routing Variant Selection Rule & Operational Queue Budget
- **Temporal Context**: In the PaySim dataset, each unit of `step` represents 1 simulated hour:
  - **Validation Split**: Steps $521 \dots 595$ ($75$ simulated hours, $162,467$ transactions).
  - **Test Split**: Steps $596 \dots 743$ ($148$ simulated hours, $118,146$ transactions).
- **Analyst Queue Operational Budget (Assumption)**:
  - Capacity assumption: **40 Analyst Review alerts per simulated hour** (recorded explicitly as an operational assumption; there is no industry standard).
  - **Eligibility Criterion**: A variant is eligible if and only if its validation queue satisfies:
    $$\text{Validation Queue Volume} \le 40 \times 75 = \mathbf{3,000 \text{ alerts}}$$
  - **Budget Sensitivity Analyses**: Also evaluate queue feasibility under a tight budget of **20 alerts/hour** ($1,500$ alerts cap) and a generous budget of **80 alerts/hour** ($6,000$ alerts cap).
- **Candidate Routing Variants**:
  - **Variant A (Current Backend Implementation - Tri-State Hybrid)**:
    - Candidate High Risk: $\text{score} \ge t_{\text{high}}$ OR $(\text{has\_rules} \land \text{score} \ge t_{\text{review}})$.
    - Stable Candidate High Risk $\to$ **High Risk**.
    - Unstable Candidate High Risk $\to$ **Analyst Review**.
    - Pure Rule Trigger ($\text{has\_rules} \land \text{score} < t_{\text{review}}$) $\to$ **Analyst Review**.
    - Borderline Score ($t_{\text{review}} \le \text{score} < t_{\text{high}}$ without rules) $\to$ **Analyst Review**.
    - Otherwise $\to$ **Low Risk**.
  - **Variant B (Score-Gated Rules - Rule Never Routes Alone)**:
    - Same as Variant A, EXCEPT that a rule trigger alone with ML score $< t_{\text{review}}$ routes to **Low Risk** (the rule flag is preserved in the audit log, but does not route to human review).
  - **Variant C (Train-Recalibrated Heuristic Rules)**:
    - Retains Variant A routing logic, but recalibrates rule thresholds in `app.services.rules` strictly on the **TRAIN** split such that each heuristic rule triggers on at most a target cap of $\le 0.5\%$ flag rate on training data.
    - *Feasibility Check & Fallback*: Perform a train-only feasibility check per rule. If a 0.5% cap is unreachable for a specific rule, determine the lowest achievable training flag rate; if the lowest achievable rate remains $> 1.0\%$, the rule is disabled in Variant C.
  - **Variant D (Pure Machine Learning Gateway - No Rules)**:
    - Deactivates heuristic rule routing entirely: score $\ge t_{\text{high}}$ (Stable $\to$ High Risk, Unstable $\to$ Analyst Review); $t_{\text{review}} \le \text{score} < t_{\text{high}} \to$ Analyst Review; score $< t_{\text{review}} \to$ Low Risk.
- **Selection Decision Rule**:
  - Evaluated strictly on **VALIDATION SPLIT ONLY**.
  - Let $C_v$ be the combined fraud capture rate in $(\text{High Risk} \cup \text{Analyst Review})$ for variant $v \in \{A, B, C, D\}$.
  - Let $C^* = \max_{v \in \text{Eligible}} C_v$ be the best capture rate among eligible variants.
  - **Decision Rule**: Among all eligible variants ($\text{Queue} \le 3,000$), choose the variant with the **smallest queue** whose capture is within 1 percentage point of the best eligible capture ($C_v \ge C^* - 0.010$). Tie-break by smaller queue.
  - **Fallback Escalate**: If NO variant is eligible ($\text{Queue} > 3,000$ for all variants), STOP and report immediately; do not proceed to test without human escalation.
  - **Reporting Metrics**: Report High-Risk-only recall, combined (High + Review) recall, and Wilson 95% score confidence intervals for both capture rates.
- **Prior Evaluation Disclosure**:
  - It is explicitly disclosed that the held-out test split was already evaluated under Variant A during preliminary Stage 2 analysis (revealing that Variant A's heuristic rules generated 35,077 review alerts, swamping the queue with 30.08% of all legitimate transactions). Variants B, C, and D were formulated in response to that empirical finding. Test performance for all variants will be reported once without iterative tuning.

---

### C. Replicate Set Protocol (10 Independent Replicate Sets)
- **Definition of "Replicate Set"**:
  - A *replicate set* consists of **its own independently fitted main model plus 5 replicate models with disjoint seeds** ($6$ model fits per set).
- **Protocol**:
  - Train **10 independent replicate sets** per strategy ($10 \times 6 = 60$ model fits per strategy, $180$ fits across all three strategies).
  - Main seeds: `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`.
  - Replicate seeds: Disjoint non-overlapping integer seeds across all 10 sets (e.g. Set 1: `[101..105]`, Set 2: `[201..205]`, ..., Set 10: `[1001..1005]`).
- **Deployed Artifact Identification**:
  - **Replicate Set 1** (Main seed 42, replicate seeds `[101, 102, 103, 104, 105]`) is designated as the authoritative **deployed artifact bundle**.
  - Replicate Sets 2 through 10 are retained strictly for variance estimation and confidence intervals, and are **not** written to `ml/models/` to keep deployment artifacts clean and compact.

---

### D. Cross-Strategy Agreement on Matched Seeds
- **Cohort Sampling Rule**:
  - Test cohort: All held-out test transactions whose deployed-strategy (`class_weight`) score satisfies $\text{risk\_score} \ge t_{\text{review}}$.
  - Cohort size cap: If the qualifying cohort exceeds 2,000 transactions, take a deterministic random sample of exactly $N = 2,000$ transactions with seed `42`.
  - Identical transactions evaluated across all three strategies (`class_weight`, `none`, `smote_enn`).
- **Matched-Seed Comparison**:
  - Strategies are compared on **MATCHED seeds** (the same majority subsample data draw and the same initialization seeds across strategies).
- **Noise Decomposition**:
  - Decompose within-strategy replicate overlap into:
    1. **Model-seed-only noise**: Fixed subsample data draw, varying model initialization seed.
    2. **Data-draw noise**: Varying majority subsample draws across replicate sets.

---

### E. Pre-Specified Statistical Conclusions
To maintain scientific integrity and prevent post-hoc rationalization, conclusions are pre-specified:
1. **Informative Nature of `Unstable`**:
   - Explanation instability is concluded to be **informative** for triage IF AND ONLY IF:
     (a) The Wilson 95% confidence interval of the False Positive rate among Unstable candidate High Risk rows lies strictly above that of Stable candidate High Risk rows (lower bound of Unstable FPR $>$ upper bound of Stable FPR), AND
     (b) The 95% confidence interval of the AUROC of mean overlap for predicting False Positives strictly excludes 0.50 (lower bound $> 0.50$).
   - If either condition is not met, report plainly: **"no evidence that explanation instability predicts false positive alert status"**.
2. **Cross-Strategy Explanation Agreement**:
   - Cross-strategy agreement is concluded to differ from retraining noise IF AND ONLY IF the 95% bootstrap interval of between-strategy overlap lies strictly below the within-strategy retraining noise interval.
   - Otherwise, report plainly: **"no evidence of strategy-induced explanation shift beyond retraining noise"**.

---

## 3. Work Breakdown: Steps 1 Through 6

### STEP 1: Heuristic Rules Root-Cause Analysis, Vectorized Parity & Type Breakdown
1. **Root-Cause Investigation**:
   - Investigate why earlier rule evaluation tables showed identical metrics between unconstrained mathematical definitions and backend rule functions.
   - Verify boundary conditions:
     - `old_balance_orig == 0` (zero balance origin drain).
     - Exact threshold boundaries (`amount == 200,000.0`, `drain_ratio == 0.90`).
     - Zero destination balance and zero new origin balance conditions.
2. **5,000-Row Boundary Test**:
   - Sample $\ge 5,000$ transactions from PaySim containing all edge cases.
   - Assert 100% bit-exact parity between vectorized functions in `ml/src/load_data.py` and `backend.app.services.rules.evaluate_rules`.
3. **Per-Rule Metrics Table (Train and Validation Only)**:
   - Report flag rate, recall, precision, FPR, and trigger counts for:
     - Large Transfer (`amount > 200,000` & `type == TRANSFER`).
     - Origin Drain (`old_balance_orig > 0`, `new_balance_orig == 0`, `amount / old_balance_orig >= 0.90`).
     - Destination Balance Inconsistency.
     - Very High Amount for Type.
     - Rule Union.
   - **Transaction Type Breakdown**: Add breakdown by transaction type (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`).
4. **Table Automation & Grep Proof**:
   - Implement script `scripts/generate_model_card_tables.py` to generate markdown tables directly from canonical JSON reports.
   - Provide an automated test in `ml/tests/test_ml_pipeline.py` that fails if markdown tables diverge from JSON numbers.
   - Provide grep verification confirming no deprecated rule figures remain.

---

### STEP 2: Replicate Artifact Generation & Multi-Set Stability Evaluation
1. **Model Training**:
   - Train 10 independent replicate sets for `class_weight`, `none`, and `smote_enn`.
   - Package Replicate Set 1 into `ml/models/<strategy>/` with $k=5, \text{min\_overlap}=0.80$.
   - Re-run `ml/src/check_artifact.py` with startup self-test.
2. **Variant A Reference Run (Same-Run Comparison)**:
   - Include a 15-feature Variant A reference run using the exact same seeds and sample cohort as the deployed model.
   - The stability comparison between Variant A (with balance errors) and Variant B (without balance errors) is strictly same-run.
   - Enforce that no collinearity claim is made beyond what the measured confidence intervals support.
3. **Statistical Reporting**:
   - Stability rate per replicate set with 95% intervals across sets, plus row-level bootstrap intervals.
   - Chance baseline ($45.45\%$) displayed beside every stable rate.
   - Full distribution of mean overlap and Spearman rank correlation of $|SHAP|$ across all 11 features.
   - Comparison of the three strategies using these intervals.
4. **Postgres Parity Re-verification**:
   - Re-run `scripts/parity_check.py` on a throwaway PostgreSQL database.
   - Report separately:
     (a) Offline unrounded score vs service unrounded score $\le 10^{-6}$.
     (b) `round(offline, 4)` vs API `risk_score` $\le 10^{-9}$ (exact equality).
     (c) Exact match for `gateway_decision` and `explanation_status`.

---

### STEP 3: Gateway Evaluation (`scripts/evaluate_gateway.py`)
1. **Evaluation Execution**:
   - Import backend `evaluate_rules` and `route_decision` (read-only) to evaluate saved artifacts.
2. **Four-Variant Evaluation**:
   - Evaluate Variants A, B, C, and D on Validation and Test splits.
   - Apply the selection rule from Section 2.B (queue budget $\le 3,000$, capture within 1 pp of best eligible capture on Validation).
   - Recommend one variant based on empirical numbers without implementing backend modifications.
3. **Informative Analysis of `Unstable`**:
   - For candidate High Risk rows on Validation and Test:
     - $N$, TP, FP, Precision with Wilson 95% intervals for Stable vs Unstable cohorts.
     - AUROC of mean overlap for predicting False Positives with 95% CI.
     - Score-margin control: Re-route an equal number of rows with the smallest score margin above $t_{\text{high}}$.
     - Apply pre-specified conclusion criteria from Section 2.E.

---

### STEP 4: Cross-Strategy Explanation Agreement
1. **Offline Evaluation on Matched Seeds**:
   - Evaluate the $N \le 2,000$ test cohort across main models of all three strategies.
   - Pairwise mean overlap compared against within-strategy seed-to-seed retraining noise floor.
   - Decompose into model-seed noise and data-draw noise.
   - Apply pre-specified conclusion criteria from Section 2.E.

---

### STEP 5: Metrics Table Layout & Model Card Documentation
1. **Comprehensive Metrics Table Layout**:
   - Keep all five primary metrics: Precision, Recall, F1, PR-AUC, MCC, plus FPR.
   - Report split prevalence alongside Precision, F1, MCC, and PR-AUC.
   - Highlight Recall and FPR as the prevalence-independent metrics.
   - Add Bayesian reweighted precision: $\text{Precision}(\pi_{\text{train}}) = \frac{\text{TPR} \cdot \pi_{\text{train}}}{\text{TPR} \cdot \pi_{\text{train}} + \text{FPR} \cdot (1 - \pi_{\text{train}})}$.
2. **Prevalence Ratio Investigation**:
   - Investigate and document why test prevalence ($1.3898\%$) is $\approx 14.6\times$ train prevalence ($0.09505\%$).
3. **Reference Model Rigor**:
   - For Full-Train reference, Logistic Regression, and Random Forest: state training data (full vs 10:1 subsample), tuning budget, and decision threshold rule.
   - Enforce the same validation-based threshold rule as the deployed model or flag differences.
   - Fix "sklearn default D=8" wording for the Random Forest baseline (note that default is `None`, and $D=8$ was an explicit constraint).
4. **MODEL_CARD Documentation**:
   - Explicitly document that the 10:1 subsample lowers recall versus full-train (with exact empirical numbers).
   - Document that heuristic rule flags dominate the Analyst Review queue (with exact empirical numbers).
   - Confirm that no separability or collinearity claims exceed measured intervals.

---

### STEP 6: Integrity & Verification Audit
1. **Determinism**: Run the full pipeline twice from clean state and verify identical output hashes.
2. **Feature Leakage**: Formally verify no feature uses `isFraud` or `isFlaggedFraud`.
3. **Chronological Invariant**: Formally verify $\max(\text{train}) = 520 < \min(\text{val}) = 521 \le \max(\text{val}) = 595 < \min(\text{test}) = 596$.

---

## 4. Prior Metric Reconciliation (Stage 2 vs Gateway Evaluation)

Before proceeding to Step 2, any metric divergence between the preliminary Stage 2 training report and the subsequent gateway evaluation report must be reconciled and explained:

| Metric / Artifact | Preliminary Stage 2 Training Report | Subsequent Gateway Evaluation Report | Root Cause of Discrepancy | Correct Authoritative Run |
| :--- | :---: | :---: | :--- | :--- |
| **Test Recall (`class_weight`)** | 94.46% (at high threshold) | 95.31% (ML high threshold) / 79.35% (High Risk decision) | Reporting context: Stage 2 reported pure ML score threshold ($s \ge t_{\text{high}}$); gateway evaluation reported both the unrouted score ($95.31\%$) and the post-SHAP gateway routing ($79.35\%$). | Gateway evaluation routing is authoritative for gateway behavior; ML score is authoritative for pure model discrimination. |
| **Test PR-AUC (`class_weight`)** | 0.9907 | 0.9996 | Sample cohort evaluation: Preliminary report computed PR-AUC on an evaluation subsample; full gateway run computed PR-AUC over all 118,146 test rows. | Full test split (118,146 rows) is authoritative ($0.9996$). |
| **Thresholds (`none`, `smote_enn`)** | Review: 0.9557, High: 0.9712 (shared) | `none`: $t_{\text{high}}=0.9754$, `smote_enn`: $t_{\text{high}}=0.9840$ | Threshold calibration rule: Stage 2 initially reused `class_weight` thresholds for all strategies; later evaluation correctly re-calibrated thresholds per strategy on validation unique scores. | Per-strategy validation calibrated thresholds are authoritative. |
| **Stability Rate ($k=5$)** | 96.83% (Alerts cohort) | 82.74% (Frauds cohort) | Cohort definition: 96.83% was measured on all candidate alerts (frauds + false alarms); 82.74% was measured strictly on test fraud rows ($1,642$ frauds). | Both will be reported with explicit cohort labels (Frauds vs All Alerts). |
| **Rules-Only Baseline** | Recall: 99.51%, Prec: 4.77% | Recall: 38.31%, Prec: 4.77% | Rule definition: 99.51% used unconstrained union without origin balance requirements; 38.31% used backend `app.services.rules` requiring `new_balance_orig == 0`. | Backend rule function ($38.31\%$) is authoritative. |

---

## 5. Next Step & Checkpoint

Per user instructions, upon committing this amended analysis plan to branch `feature/ml-pipeline`, all work stops for review and explicit approval before any code or training execution begins.
