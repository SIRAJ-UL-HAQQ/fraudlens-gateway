# FraudLens Gateway — SRS Architectural Deviations

This document formally records all architectural and implementation deviations from the initial Software Requirements Specification (SRS). In accordance with project governance, the LaTeX source files of the SRS are preserved without retrospective rewriting; this living document acts as the definitive record of design adjustments agreed upon and empirically substantiated during implementation.

---

## 1. Deviation: FR-7 Hybrid Rule Decision Layer (and Dependent FR-8 / FR-9)

### 1.1. Requirement & SRS Text
- **Requirement Identifier**: `FR-7` (Hybrid Rule Evaluation Layer), with downstream dependencies `FR-8` (Tri-State Decision Matrix) and `FR-9` (Analyst Review Queue Routing).
- **Original SRS Specification**:
  > *"The system shall evaluate deterministic heuristic safety rules in addition to the ML score when computing the final gateway decision... Any triggered heuristic safety rule shall escalate the transaction to Analyst Review or High Risk."*

### 1.2. Actual Implementation (Variant B: Score-Gated Rules)
- A heuristic rule trigger raises a transaction to candidate **High Risk** only when the calibrated machine learning score meets or exceeds the review threshold (`risk_score >= threshold_review`).
- A heuristic rule trigger occurring on a transaction with low ML score (`risk_score < threshold_review`) is strictly recorded in `rule_flags` and persisted in the append-only database audit log, but **does not alter the gateway decision**; the transaction is routed to **Low Risk**.
- Pure machine learning routing applies: Candidate High Risk (`risk_score >= threshold_high` OR `has_rules and risk_score >= threshold_review`) with Stable explanation $\to$ **High Risk**; with Unstable explanation $\to$ **Analyst Review**; borderline band (`threshold_review <= risk_score < threshold_high` without rules) $\to$ **Analyst Review**; all other transactions $\to$ **Low Risk**.

### 1.3. Rationale and Empirical Evidence
During Stage 2 and Stage 3 gateway evaluation, unconstrained rule enforcement (Variant A) was shown to overwhelm operational queues by triggering on large volumes of legitimate transactions. As reported in `ml/reports/gateway_evaluation.json`:

1. **Analyst Queue Overload Under Variant A**:
   - **Validation Split** (75 simulated hours, 162,467 rows, 790 frauds):
     - `variant_evaluations.validation.variant_A.metrics.analyst_review_queue_rate_per_hour`: **`659.53`** alerts/hour (`49,465` total queue).
     - `variant_evaluations.validation.variant_A.metrics.legit_in_review_rate_pct`: **`30.51%`** (`49,325` legitimate transactions routed to review).
   - **Test Split** (148 simulated hours, 118,146 rows, 1,642 frauds):
     - `variant_evaluations.test.variant_A.metrics.analyst_review_queue_rate_per_hour`: **`239.05`** alerts/hour (`35,379` total queue).
     - `variant_evaluations.test.variant_A.metrics.legit_in_review_rate_pct`: **`30.08%`** (`35,044` legitimate transactions routed to review).

2. **Pre-Specified Selection Rule**:
   - **Operational Budget**: Queue capacity $\le 40$ alerts per simulated hour ($\le 3,000$ queue on validation, `selection_rule_validation.budget_limit_alerts: 3000`).
     *(Note: The 40 alerts/hour queue budget is an engineering operational assumption, not an external statutory or regulatory standard).*
   - **Selection Criterion**: Among all eligible variants satisfying the queue budget, select the variant with the smallest queue whose combined recall is within 1.0 percentage point of the best eligible capture (`selection_rule_validation.capture_threshold_1pp_pct: 94.06%`).
   - **Eligible Variants Comparison**:
     - **Variant B** (`variant_evaluations.validation.variant_B`): Validation queue **`145`** (`1.93` alerts/hour), legitimate in review **`0.0266%`**, combined recall **`95.06%`** (`751 / 790` frauds).
     - **Variant D** (`variant_evaluations.validation.variant_D`): Validation queue **`192`** (`2.56` alerts/hour), legitimate in review **`0.0476%`**, combined recall **`95.06%`** (`751 / 790` frauds).
     - **Outcome**: **Variant B** achieves the smallest queue (`145` alerts vs `192` for D) while meeting the capture criterion (`selection_rule_validation.smallest_queue_variant: "variant_B"`).
   - **Confirmed on Held-Out Test Split**:
     - **Variant B** (`variant_evaluations.test.variant_B`): Test queue **`302`** (`2.04` alerts/hour), legitimate in review **`0.0258%`**, combined recall **`95.92%`** (`1,575 / 1,642` frauds).
     - **Variant D** (`variant_evaluations.test.variant_D`): Test queue **`338`** (`2.28` alerts/hour), legitimate in review **`0.0395%`**, combined recall **`95.92%`** (`1,575 / 1,642` frauds).

3. **Trade-Off & Cost of Selected Variant B**:
   - **Validation Split**: `38` rule-only frauds (`rule_only_frauds_score_distribution.validation.count: 38`, score distribution median `0.9136`, max `0.9543`) with scores below $T_{rev} = 0.9557$ are no longer routed to review. Combined recall drops from `99.87%` to `95.06%` ($\Delta = -4.81$ percentage points).
   - **Test Split**: `63` rule-only frauds (`rule_only_frauds_score_distribution.test.count: 63`, score distribution median `0.8980`, max `0.9556`) with scores below $T_{rev} = 0.9557$ are no longer routed to review. Combined recall drops from `99.76%` to `95.92%` ($\Delta = -3.84$ percentage points).
   - **Justification**: In exchange for this recall cost, false positive review burden is reduced from over $30\%$ of all legitimate transactions down to less than $0.03\%$, making human analyst investigation viable without queue collapse.

---

## 2. Naming & Schema Deviations

### 2.1. Rule Flags Naming (`rule_triggers` vs `rule_flags`)
- **SRS Text (Table 6.4)**: Specifies the JSON response field as `rule_triggers: list[string]`.
- **Implementation**: The Pydantic schema in [`backend/app/models/schemas.py`](file:///backend/app/models/schemas.py) and OpenAPI contract define this field as **`rule_flags: list[str]`**.
- **Reason**: `rule_flags` was adopted in the initial API source of truth (`docs/api-contract.md`) to distinguish passive boolean rule tags from active gateway routing triggers.

---

## 3. Previously Agreed Architectural Deviations

### 3.1. Dynamic Threshold Calibration vs Fixed Constants
- **SRS Text (Section 4.2)**: Proposed fixed static decision thresholds (e.g. $0.50$ review, $0.80$ high).
- **Implementation**: Thresholds are dynamically loaded at startup from the deployed model bundle's `manifest.json` (`thresholds.high` and `thresholds.review`), calibrated empirically on the training/validation distribution to optimize precision and recall under extreme class imbalance.

### 3.2. Explanation Stability Minimum Overlap (`stability_min_overlap: 0.80`)
- **SRS Text (FR-11)**: Proposed a fractional agreement threshold of `0.70` across replicate explanations for top-$k$ feature sets.
- **Implementation**: Configured as `stability_min_overlap = 0.80` with $k = 5$.
- **Equivalence**: With $k = 5$ features and integer overlap counts, $\lceil 0.70 \times 5 - 10^{-9} \rceil = 4$ and $\lceil 0.80 \times 5 - 10^{-9} \rceil = 4$. Both thresholds require an identical minimum overlap of 4 out of 5 features to achieve `Stable` status.

### 3.3. Origin Account Drained Heuristic Definition
- **SRS Text (FR-7.1)**: Defined origin drain simply as `newbalanceOrig == 0`.
- **Implementation**: Implemented in [`backend/app/services/rules.py`](file:///backend/app/services/rules.py) with strict boundary requirements:
  `tx.old_balance_orig > 0.0 and tx.new_balance_orig == 0.0 and tx.amount >= tx.old_balance_orig * 0.90` (scoped strictly to `TRANSFER` and `CASH_OUT`).
- **Reason**: PaySim contains zero-amount and zero-balance transactions; requiring positive initial balance and a drain ratio of $\ge 90\%$ eliminates false triggers on empty accounts.

### 3.4. API Endpoint Naming and Enums
- **Implementation**: Endpoints conform strictly to RESTful conventions prefixed with `/api/v1/` (`/api/v1/predict`, `/api/v1/transactions`, `/api/v1/metrics`, `/api/v1/export`), using string enums (`"Low Risk"`, `"High Risk"`, `"Analyst Review"`, `"Stable"`, `"Unstable"`).
