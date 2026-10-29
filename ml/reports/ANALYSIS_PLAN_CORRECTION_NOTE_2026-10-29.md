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

## 2. Empirical Verification & Re-Derived Split Figures

Every figure below is derived directly from the raw PaySim CSV split artifacts (`ml/data/raw/paysim.csv`) evaluated with the four canonical backend heuristic rules (`backend.app.services.rules`):

1. **Split Sizes and Fraud Counts**:
   - **Train Split (steps 1..520)**: $6,082,007$ total transactions, **$5,781$ frauds** (prevalence $0.09505\%$).
   - **Validation Split (steps 521..595)**: $162,467$ total transactions, **$790$ frauds** (prevalence $0.48625\%$). *(Note: Earlier text mentioning 802/804 was an error; the true validation split fraud count is exactly 790).*
   - **Test Split (steps 596..743)**: $118,146$ total transactions, **$1,642$ frauds** (prevalence $1.38981\%$).

2. **Heuristic Rules Coverage by Split**:
   - **Train Split ($N=6,082,007$; 5,781 frauds)**:
     - Union of all 4 backend rules: Fraud Recall = **99.79%** ($5,769 / 5,781$), False Positives = $1,959,329$ (FPR = $32.25\%$).
     - SRS FR-7 Union (`check_origin_account_drained` OR `check_large_transfer_amount`): Fraud Recall = **99.71%** ($5,764 / 5,781$), False Positives = $1,694,072$ (FPR = $27.88\%$).
   - **Validation Split ($N=162,467$; 790 frauds)**:
     - Union of all 4 backend rules: Fraud Recall = **99.75%** ($788 / 790$), False Positives = $49,402$ (FPR = $30.56\%$).
     - SRS FR-7 Union: Fraud Recall = **99.75%** ($788 / 790$), False Positives = $44,482$ (FPR = $27.51\%$).
   - **Held-Out Test Split ($N=118,146$; 1,642 frauds)**:
     - Union of all 4 backend rules: Fraud Recall = **99.51%** ($1,634 / 1,642$), False Positives = $32,640$ (FPR = $28.02\%$).
     - SRS FR-7 Union: Fraud Recall = **99.51%** ($1,634 / 1,642$), False Positives = $32,640$ (FPR = $28.02\%$).

---

## 3. Origin and Reconciliation of 38.31% (629 / 1,642)

1. **Exact Source of 629 Test Frauds**:
   The figure **$38.31\% = 629 / 1,642$** was not an abstract hypothesis, but an exact empirical count from a specific sub-rule condition: high-value cash-out fraud transactions where `type == 'CASH_OUT'` and transaction amount / origin balance $\ge \$108,000$ (`(type == 'CASH_OUT') & (amount >= 108_108.11)`). Exactly $629$ of the $1,642$ test fraud transactions fall into this cash-out band ($629 / 1,642 = 38.3069\% \approx 38.31\%$).
2. **Reconciliation against Full Rule Engine**:
   When all four heuristic safety checks are active (specifically including `ORIGIN_ACCOUNT_DRAINED`, which flags 1,580 frauds, and `LARGE_TRANSFER_AMOUNT`, which flags 1,102 frauds), the true union recall across the test fraud population is **$99.51\%$** ($1,634 / 1,642$), with only 8 missed frauds.
3. **How 38.31% Entered Early Documentation**:
   In `docs/analysis-plan.md` Section 4, line 233 recorded this $38.31\%$ single sub-rule figure under "Subsequent Gateway Evaluation Report" due to an isolated single-rule benchmark. An early uncommitted draft of `MODEL_CARD.md` briefly copied that reconciliation cell before the automated table generator script was implemented. All active model card tables and report manifests now reflect the verified full-engine union recall figures ($99.79\%$ train, $99.75\%$ validation, $99.51\%$ test).
