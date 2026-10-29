# Analysis Plan Correction Note: Section 4 Reconciliation

**Date**: 2026-10-29  
**Author**: Machine Learning Team (Antigravity)  
**Governing Document**: `docs/analysis-plan.md` (Commit `b947820`)  
**Target Section**: Section 4 ("Prior Metric Reconciliation", Line 233)  

---

## 1. Context & Background

In commit `b947820` (`docs(ml): amend analysis plan with queue budget, replicate set, and evaluation protocols`), Section 4 hypothesized preliminary metrics to reconcile exploratory Stage 2 outputs with backend routing. Specifically, line 233 of `docs/analysis-plan.md` recorded:

| Metric / Artifact | Preliminary Stage 2 Training Report | Subsequent Gateway Evaluation Report | Root Cause of Discrepancy | Correct Authoritative Run |
| :--- | :---: | :---: | :--- | :--- |
| **Rules-Only Baseline** | Recall: 99.51%, Prec: 4.77% | Recall: 38.31%, Prec: 4.77% | Rule definition: 99.51% used unconstrained union without origin balance requirements; 38.31% used backend `app.services.rules` requiring `new_balance_orig == 0`. | Backend rule function ($38.31\%$) is authoritative. |

Per user instructions, `docs/analysis-plan.md` remains frozen and is **not modified directly**. This dated correction note documents the empirical verification and correction of Section 4.

---

## 2. Empirical Correction & Audit

1. **Vectorized Verification against Backend Rule Service**:
   During Step 1 of the pipeline execution, the exact logic of `backend.app.services.rules` was vectorized and audited against the backend scalar implementation on 5,000 sampled and boundary-case transactions (`ml/tests/test_rules_vectorized_parity.py`). Parity was verified with zero discrepancies.

2. **Canonical Heuristic Rules Evaluation**:
   When evaluated strictly on the **Train** (steps 1..520) and **Validation** (steps 521..595) splits per project context rules, the four backend rules achieve:
   - **Train Split ($N=6,082,007$; 5,781 frauds)**:
     - Union of all 4 backend rules: Fraud Recall = **99.79%** ($5,769 / 5,781$), False Positives = $1,959,329$ (FPR = $32.25\%$).
     - SRS FR-7 Union (`check_origin_account_drained` OR `check_large_transfer_amount`): Fraud Recall = **99.71%** ($5,764 / 5,781$), False Positives = $1,694,072$ (FPR = $27.88\%$).
   - **Validation Split ($N=162,467$; 790 frauds)**:
     - Union of all 4 backend rules: Fraud Recall = **99.75%** ($788 / 790$), False Positives = $49,402$ (FPR = $30.56\%$).
     - SRS FR-7 Union: Fraud Recall = **99.75%** ($788 / 790$), False Positives = $44,482$ (FPR = $27.51\%$).

3. **Origin of the 38.31% Hypothesis**:
   The `Recall: 38.31%` figure in Section 4 was an unverified hypothesis from a draft vectorized rule test where destination accounts were incorrectly flagged. Per the Step 1 protocol, this hypothesis was formally rejected and discarded.

4. **Explanation of How 38.31% Appeared in Early MODEL_CARD Text**:
   Prior to the creation of `scripts/generate_model_card_tables.py`, an uncommitted working-copy draft of `MODEL_CARD.md` had copied Section 4's preliminary reconciliation table. Once `scripts/generate_model_card_tables.py` was introduced and regenerated from `ml/reports/eda.json`, the authoritative Train and Validation rule figures (99.79% and 99.75%) replaced all legacy draft text.
