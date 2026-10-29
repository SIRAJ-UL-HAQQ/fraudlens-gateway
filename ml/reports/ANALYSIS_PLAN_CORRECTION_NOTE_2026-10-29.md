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

## 3. Status of 38.31% (629 / 1,642)

629/1,642 appeared in the plan and an early MODEL_CARD draft; it is not reproducible from any pipeline output; its origin was not established; it was removed.

---

## 4. Unmeasured Values Reported in Preliminary Discussions

For absolute transparency and integrity, the following figures reported in preliminary chat conversations were NOT measured from valid pipeline runs and were discarded:
1. **0.8845**: A placeholder value that appeared in early drafting of candidate alert overlap before Set 1 replicate evaluation.
2. **Variant A "seeds 4201-4205, m=5"**: Variant A was actually evaluated on $m=2$ replicates (seeds 101, 102), not $m=5$ with seeds 4201..4205.
3. **Model-seed range "4201..4210"**: Model-seed-only noise was actually fit on seeds `500..509` in `ml/src/train.py:81`; "4201..4210" was an unexecuted draft naming proposal.
4. **Validation frauds "804"**: A preliminary conversational estimate; the true chronological validation fraud count is exactly $790$.

The generated JSON reports (`ml/reports/metrics.json`, `ml/reports/stage1_validation_results.json`, `ml/reports/cross_strategy_agreement.json`, `ml/reports/gateway_evaluation.json`) and the synchronized `MODEL_CARD.md` tables are the only authoritative numbers.
