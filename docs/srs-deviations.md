# FraudLens Gateway — SRS Architectural Deviations

**Audit Date**: 2026-11-01  
**Target Specification Document**: [`FraudLens_Gateway_SRS.tex`](FraudLens_Gateway_SRS.tex)  
**Target Specification SHA256**: `6EBEC864D8678362EF6C05C2DC969BE640124C6E17F2B06C1CDC2F764563CAA0`  

This document formally records all architectural, algorithmic, interface, and schema deviations between the official Software Requirements Specification ([`FraudLens_Gateway_SRS.tex`](FraudLens_Gateway_SRS.tex)) and the deployed FraudLens Gateway implementation. In accordance with project governance, the official LaTeX source file remains frozen without retrospective editing; this living document acts as the definitive record of design adjustments, operational interpretations, and empirical trade-offs established during development.

---

## 1. Variant B: Score-Gated Rules (Interpretation of FR-7)

### 1.1. Requirement & SRS Text
- **Requirement Identifier**: `FR-7` ([`FraudLens_Gateway_SRS.tex:362`](FraudLens_Gateway_SRS.tex#L362)).
- **Verbatim SRS Text**:
  > *"The system shall apply a hybrid rule-based layer (e.g., origin balance drainage $\ge 95\%$, transactions exceeding 200,000 currency units) in addition to the ML score when computing the final gateway decision."*
- **Downstream Requirements**:
  - `FR-8` ([`FraudLens_Gateway_SRS.tex:363`](FraudLens_Gateway_SRS.tex#L363)): *"The system shall classify each scored transaction into one of three categories: Low Risk, High Risk, or Analyst Review."*
  - `FR-9` ([`FraudLens_Gateway_SRS.tex:364`](FraudLens_Gateway_SRS.tex#L364)): *"A transaction classified as High Risk with an unstable explanation shall be automatically re-routed to Analyst Review rather than remaining High Risk."*

### 1.2. Implementation & Architectural Precedence
FR-7 as written does not specify the precedence or gating mechanism between heuristic safety rules and machine learning scores. FraudLens Gateway implements **Variant B (Score-Gated Rules)** in [`backend/app/services/scoring.py::ScoringService.predict`](../backend/app/services/scoring.py):
1. A heuristic rule trigger raises a transaction to candidate **High Risk** only when the calibrated machine learning score meets or exceeds the review threshold (`risk_score >= threshold_review`).
2. A heuristic rule trigger occurring on a transaction with low ML score (`risk_score < threshold_review`) is recorded in `rule_flags` and persisted directly in the database (`transactions.rule_flags`), but **does not alter the gateway decision**; the transaction routes to **Low Risk**.
3. Gateway triage executes:
   - Candidate High Risk (`risk_score >= threshold_high` OR `has_rules and risk_score >= threshold_review`):
     - If explanation is **Stable** $\to$ **High Risk**.
     - If explanation is **Unstable** $\to$ routed to **Analyst Review** (per FR-9).
   - Borderline score band (`threshold_review <= risk_score < threshold_high` without rules) $\to$ **Analyst Review**.
   - All other transactions (`risk_score < threshold_review`) $\to$ **Low Risk**.

### 1.3. Empirical Evidence: Benefit and Recall Cost
During Stage 2 and Stage 3 evaluation, unconstrained rule enforcement (Variant A: any triggered rule escalates to Review or High Risk regardless of ML score) was evaluated against Variant B across PaySim validation and held-out test splits ([`ml/reports/gateway_evaluation.json`](../ml/reports/gateway_evaluation.json)):

1. **Analyst Review Queue Reduction (Operational Benefit)**:
   - **Validation Split** (75 simulated hours, 162,467 rows, 790 frauds):
     - Variant A queue rate: **659.53 alerts/hour** (49,465 total queue; 49,325 legitimate transactions in review, or **30.51%** of all non-frauds).
     - Variant B queue rate: **1.93 alerts/hour** (145 total queue; 43 legitimate transactions in review, or **0.0266%** of non-frauds).
   - **Held-Out Test Split** (148 simulated hours, 118,146 rows, 1,642 frauds):
     - Variant A queue rate: **239.05 alerts/hour** (35,379 total queue; 35,044 legitimate transactions in review, or **30.08%** of all non-frauds).
     - Variant B queue rate: **2.04 alerts/hour** (302 total queue; 30 legitimate transactions in review, or **0.0258%** of non-frauds).

2. **Recall Trade-Off (Cost of Variant B)**:
   - **Validation Split**: 38 rule-only frauds with ML scores below review threshold ($t_{\text{review}} = 0.9557$) are no longer escalated to review. Combined fraud capture drops from **99.87%** (789/790) down to **95.06%** (751/790), representing a recall reduction of **-4.81 percentage points**.
   - **Held-Out Test Split**: 63 rule-only frauds with ML scores below review threshold ($t_{\text{review}} = 0.9557$) are no longer escalated to review. Combined fraud capture drops from **99.76%** (1,638/1,642) down to **95.92%** (1,575/1,642), representing a recall reduction of **-3.84 percentage points**.

---

## 2. Comprehensive SRS Audit Findings

The table below catalogs every requirement, design constraint, and data dictionary specification from [`FraudLens_Gateway_SRS.tex`](FraudLens_Gateway_SRS.tex).

| SRS ref (label + line) | SRS text (verbatim) | Implementation | Status | Section in this doc |
| :--- | :--- | :--- | :---: | :---: |
| **Ch. 2 / L289** | *Benchmark Dataset Finalization: The system must use the publicly available PaySim Synthetic Financial Dataset for all training, validation, and scoring demonstrations...* | Sourced from PaySim; pipeline in [`ml/src/load_data.py::load_paysim_csv`](../ml/src/load_data.py) and preprocessor in [`ml/src/features.py::PaySimFeaturePreprocessor`](../ml/src/features.py). | Implemented | N/A |
| **Ch. 2 / L290** | *Classifier Benchmarking: The primary model must be XGBoost, evaluated against Logistic Regression and Random Forest baselines; no single model may be presented as the only candidate without empirical comparison.* | Primary XGBoost benchmarked against Logistic Regression and Random Forest baselines in [`ml/src/train.py::main`](../ml/src/train.py); recorded in [`ml/reports/metrics.json`](../ml/reports/metrics.json). | Implemented | N/A |
| **Ch. 2 / L291** | *Bounded Imbalance Handling: Class-imbalance handling is strictly limited to three strategies: no resampling, class weighting, and SMOTE/SMOTE-ENN...* | Primary XGBoost trained under `none`, `class_weight`, and `smote_enn` ([`ml/src/train.py::main`](../ml/src/train.py)). | Implemented | N/A |
| **Ch. 2 / L292** | *Temporal Split Partitioning: The train/test split must be chronological (temporal ordering based on PaySim's step attribute), rather than a random shuffle, preventing lookahead data leakage.* | Chronological partition in [`ml/src/load_data.py::load_paysim_csv`](../ml/src/load_data.py): Train (steps 1..520), Val (steps 521..595), Test (steps 596..743). | Implemented | N/A |
| **Ch. 2 / L293** | *Three-Tier Containerization: The system must be containerized with Docker, using a minimal three-service Compose setup (frontend, backend, database).* | [`docker-compose.yml`](../docker-compose.yml) defines three services: `postgres`, `backend`, `frontend`. | Implemented | N/A |
| **Ch. 2 / L294** | *Timeline & Academic Scope: The project must be completed within the Software Engineering Lab course timeline... by a team of four students with mixed skill levels.* | Academic course context framing governance and moderate scope. | Implemented | N/A |
| **Ch. 2 / L305-316** | *Assumptions and Dependencies (PaySim schema stability, team Docker setups, open-source dependencies, optional Slack/email alerts).* | Dependencies pinned in `pyproject.toml` and `package.json`; Docker setups standardized across team. | Implemented | N/A |
| **Ch. 3 / L322-332** | *User Interfaces (Transaction Feed View, Flagged Transactions View, Model Performance Panel, Test Transaction Sandbox, Export Control View).* | React views implemented in [`frontend/src/components/`](../frontend/src/components/); DOM mounting via `createRoot` is pending. | Pending | Section 3.9 |
| **Ch. 3 / L386-419** | *Use Case View (UC-1 Score Transaction, UC-2 Review Flagged Transaction, UC-3 Submit Test Transaction, UC-4 Export Flagged Report, UC-5 Train Classifiers, UC-6 Manage System Deployment).* | Functional capabilities correspond to FR-1 through FR-18. | Implemented | N/A |
| **FR-1 / L352** | *The system shall train an XGBoost fraud classifier, plus Logistic Regression and Random Forest baseline classifiers, on the finalized PaySim financial transaction dataset.* | Implemented in [`ml/src/train.py::main`](../ml/src/train.py); baseline outputs recorded in [`ml/reports/metrics.json`](../ml/reports/metrics.json). | Implemented | N/A |
| **FR-2 / L353** | *The system shall train each classifier under three class-imbalance handling strategies: no resampling, class weighting, and SMOTE/SMOTE-ENN.* | Primary XGBoost trained across all 3 strategies (`none`, `class_weight`, `smote_enn`). Baseline classifiers (Logistic Regression, Random Forest) trained only under 10:1 subsampled baseline ([`ml/src/train.py::main`](../ml/src/train.py)). | Deviation | Section 3.6 |
| **FR-3 / L354** | *The system shall evaluate each trained model using Precision, Recall, F1-score, PR-AUC, and Matthews Correlation Coefficient (MCC), computed on a chronologically held-out PaySim test set.* | Computed over held-out test split via [`ml/src/evaluate.py::compute_metrics_bundle`](../ml/src/evaluate.py); recorded in [`ml/reports/metrics.json`](../ml/reports/metrics.json). | Implemented | N/A |
| **FR-4 / L355** | *The system shall perform Bayesian hyperparameter tuning (via Optuna) for the primary XGBoost model rather than using default hyperparameters.* | Executed in [`ml/src/stage1_tuning.py::objective`](../ml/src/stage1_tuning.py); optimal parameters stored in model bundle manifest. | Implemented | N/A |
| **FR-5 / L356** | *The system shall compute SHAP values for each model's predictions and measure explanation stability across repeated training runs using a top-$k$ feature overlap and Spearman rank-correlation metric.* | Top-$k$ overlap and Spearman rank correlation evaluated offline ([`ml/src/shap_stability.py::spearman_rank_correlation`](../ml/src/shap_stability.py)). Online gateway triage uses top-$k$ overlap ($k=5$, $\ge 0.80$). | Implemented | Section 3.10 |
| **FR-6 / L361** | *The system shall expose a REST endpoint (POST /api/v1/score) that accepts a PaySim transaction payload and returns a fraud risk score, the top contributing features, and an explanation-reliability status.* | Deployed endpoint is `POST /api/v1/predict` ([`backend/app/api/routes.py::predict`](../backend/app/api/routes.py)), returning [`PredictionOut`](../backend/app/models/schemas.py). | Deviation | Section 3.1 |
| **FR-7 / L362** | *The system shall apply a hybrid rule-based layer (e.g., origin balance drainage $\ge 95\%$, transactions exceeding 200,000 currency units) in addition to the ML score when computing the final gateway decision.* | Implemented as Variant B (score-gated rules). Origin drain threshold implemented as $\ge 90\%$ ([`backend/app/core/config.py::Settings`](../backend/app/core/config.py), [`backend/app/services/rules.py::check_origin_account_drained`](../backend/app/services/rules.py)). | Interpretation / Deviation | Section 1, Section 3.2 |
| **FR-8 / L363** | *The system shall classify each scored transaction into one of three categories: Low Risk, High Risk, or Analyst Review.* | Implemented via [`GatewayDecision`](../backend/app/models/schemas.py) in [`backend/app/services/scoring.py::ScoringService.predict`](../backend/app/services/scoring.py). | Implemented | Section 1 |
| **FR-9 / L364** | *A transaction classified as High Risk with an unstable explanation shall be automatically re-routed to Analyst Review rather than remaining High Risk.* | Implemented in [`backend/app/services/scoring.py::ScoringService.predict`](../backend/app/services/scoring.py). | Implemented | Section 1 |
| **FR-10 / L369** | *The system shall log every scored transaction, its decision, its SHAP explanation, its reliability status, and (once available) the analyst's final outcome to an immutable PostgreSQL audit table.* | Stored in PostgreSQL tables `transactions` and `analyst_reviews` via [`backend/app/db/repository.py::TransactionRepository`](../backend/app/db/repository.py). | Implemented | Section 3.12 |
| **FR-11 / L370** | *The system shall allow retrieval of historical transactions and their audit records via a REST endpoint with date, account, and risk filtering.* | `GET /api/v1/transactions` supports time filters (`step_min/max`, `created_from/to`) and risk (`decision`, `explanation_status`), but account filtering (`name_orig`, `name_dest`) is not implemented in [`backend/app/api/routes.py::list_transactions`](../backend/app/api/routes.py). | Deviation | Section 3.7 |
| **FR-12 / L371** | *The system shall capture and store the human fraud analyst's final adjudication verdict (Approved / Confirmed Fraud), timestamp, and commentary.* | Handled via `POST /api/v1/transactions/{id}/review`. Allowed values in [`ReviewOutcome`](../backend/app/models/schemas.py) are `"Confirmed Fraud"` and `"False Positive"`. "Approved" is represented as `"False Positive"`. | Deviation | Section 3.8 |
| **FR-13 / L376** | *The system shall display a live/recent transaction feed in the React dashboard with color-coded risk tier indicators.* | Component present in [`frontend/src/components/TransactionFeed.jsx`](../frontend/src/components/TransactionFeed.jsx); UI not integrated (pending DOM mount via `createRoot`). | Pending | Section 3.9 |
| **FR-14 / L377** | *The system shall display a filtered view of flagged (High Risk / Analyst Review) transactions with SHAP detail drawers.* | Component present in [`frontend/src/components/FlaggedView.jsx`](../frontend/src/components/FlaggedView.jsx); UI not integrated (pending DOM mount via `createRoot`). | Pending | Section 3.9 |
| **FR-15 / L378** | *The system shall display model performance metrics, PR-AUC curves, and a confusion matrix in the dashboard.* | Backend exposes metrics via `GET /api/v1/metrics`. Component [`frontend/src/components/MetricsPanel.jsx`](../frontend/src/components/MetricsPanel.jsx) renders only operational cards; PR-AUC curves and confusion matrices are not rendered in UI; UI not integrated. | Deviation / Pending | Section 3.9 |
| **FR-16 / L379** | *The system shall provide a form for submitting a test transaction and viewing its live score, SHAP explanation, and reliability status.* | Component present in [`frontend/src/components/TestTransactionForm.jsx`](../frontend/src/components/TestTransactionForm.jsx); UI not integrated (pending DOM mount via `createRoot`). | Pending | Section 3.9 |
| **FR-17 / L380** | *The system shall allow the user to export a report (PDF or CSV) of flagged transactions for a selected time range.* | Backend export endpoint implemented in [`backend/app/api/routes.py::export_transactions`](../backend/app/api/routes.py); component export link present in [`frontend/src/pages/Dashboard.jsx`](../frontend/src/pages/Dashboard.jsx); UI not integrated. | Pending | Section 3.9 |
| **FR-18 / L381** | *The system may optionally send a Slack or email alert when a High Risk transaction is flagged.* | Optional capability; no webhook relay implemented in backend codebase. | Not implemented | N/A |
| **NFR-1 / L424** | *The scoring API (/api/v1/score) shall return a risk classification, top SHAP features, and explanation-reliability status for a submitted transaction within 2 seconds under normal lab-demo load.* | Endpoint is `POST /api/v1/predict`. Latency measurement script exists in [`scripts/benchmark_latency.py`](../scripts/benchmark_latency.py). | Pending (benchmark in progress) | Section 3.1, Section 3.5 |
| **NFR-2 / L425** | *The dashboard's transaction feed shall refresh or update within 3 seconds of a new transaction being scored.* | Polling interval is set to 10 seconds (`setInterval(refreshData, 10000)`) in [`frontend/src/pages/Dashboard.jsx`](../frontend/src/pages/Dashboard.jsx); UI not integrated (pending DOM mount). | Deviation / Pending | Section 3.9 |
| **NFR-3 / L426** | *Model training (across all three candidate models and three imbalance strategies on a 100,000-sample PaySim subset) shall complete within 30 minutes on typical development hardware.* | Not measured against NFR-3's definition. The pipeline trains on a 63,591-row 10:1 subsample of the chronological train split; baselines are not trained across all 3 strategies. | Deviation / Pending | Section 3.6 |
| **NFR-4 / L427** | *The system shall support at least 10 concurrent dashboard users without noticeable performance degradation, satisfying classroom demonstration requirements.* | Concurrency capacity under multiple simultaneous browser sessions has not been evaluated in automated test suites. | Unverified | N/A |
| **NFR-5 / L428** | *Report export (PDF/CSV) for up to 5,000 transactions over a selected date range shall complete within 5 seconds.* | Capped at 5,000 rows via `EXPORT_MAX_ROWS = 5000` ([`backend/app/core/config.py::Settings`](../backend/app/core/config.py)). Latency for 5,000 rows unverified in automated tests. | Implemented / Unverified | N/A |
| **NFR-6 / L433** | *Audit Trail Immutability: The system shall not permanently delete or overwrite audit log entries; corrections or analyst remediations shall be appended as new records, preserving a tamper-evident audit history.* | Enforced at the application / repository layer ([`backend/app/db/repository.py::TransactionRepository`](../backend/app/db/repository.py)), not by database-level triggers or permissions. Cryptographic tamper-evidence is not implemented. | Implemented | Section 3.12 |
| **NFR-7 / L434** | *Graceful Failure: If the database becomes unavailable, the API shall return an explicit HTTP 503 error response rather than silently failing or returning unverified approvals.* | Handled via `DatabaseUnavailableError` returning HTTP 503 across all routes ([`backend/app/api/routes.py`](../backend/app/api/routes.py)). | Implemented | N/A |
| **NFR-8 / L435** | *Authenticated Access: Dashboard access and review submission endpoints shall require authentication (username/password or JWT bearer token) with session expiration enforced after 60 minutes of inactivity.* | Uses fixed 1-hour expiration JWT token issued at login (`JWT_EXPIRES_IN_SECONDS = 3600`, [`backend/app/core/config.py::Settings`](../backend/app/core/config.py)). Inactivity sliding-window tracking is not implemented. | Deviation | Section 3.11 |
| **NFR-9 / L436** | *Privacy & Synthetic Data Guardrail: All transaction data used for training, testing, and demonstration shall be sourced exclusively from the synthetic PaySim benchmark...* | Sourced strictly from synthetic PaySim data; no external cardholder PII is ingested. | Implemented | N/A |
| **Ch. 4 / L440-457** | *Software Quality Attributes (Reliability via Pydantic validation, Maintainability via modular container separation, Usability for analysts, Portability via Compose).* | Implemented as specified; reviewed, no findings. | Implemented | N/A |
| **Ch. 5 / L461-478** | *Other Requirements (PaySim Preprocessing Scope, Open-Source Reuse, Legal/Regulatory Synthetic Alignment).* | Implemented as specified; reviewed, no findings. | Implemented | N/A |
| **tab:paysim / L496-512** | Raw PaySim columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`. | Schema [`TransactionIn`](../backend/app/models/schemas.py) maps columns to PEP 8 `snake_case` (`name_orig`, `old_balance_orig`, etc.). Target labels `isFraud` (L514) and `isFlaggedFraud` (L516) are excluded from scoring inputs. | Implemented | N/A |
| **tab:constants / L534, L536** | `FRAUD_THRESHOLD_HIGH = 0.75`, `FRAUD_THRESHOLD_LOW = 0.20`. *(Score band between 0.20 and 0.75 is unspecified in SRS text).* | Heuristic stub mode defaults to `0.70` (high) and `0.40` (review) in [`backend/app/core/config.py::Settings`](../backend/app/core/config.py). Trained mode dynamically loads calibrated thresholds ($t_{\text{high}}=0.9712, t_{\text{review}}=0.9557$) from `manifest.json`. Scores between review and high route to Analyst Review. | Deviation | Section 3.3 |
| **tab:constants / L538** | `SHAP_STABILITY_THRESHOLD = 0.70`. | Configured as `stability_min_overlap = 0.80` with $k=5$. For $k=5$, $\lceil 0.70 \times 5 \rceil = \lceil 0.80 \times 5 \rceil = 4$; mathematically equivalent. | Implemented | Section 3.10 |
| **tab:constants / L540** | `RULE_DRAIN_BALANCE_RATIO = 0.95`. | Implemented as `RULE_DRAIN_BALANCE_RATIO = 0.90` in [`backend/app/core/config.py::Settings`](../backend/app/core/config.py) and [`backend/app/services/rules.py::check_origin_account_drained`](../backend/app/services/rules.py). | Deviation | Section 3.2 |
| **tab:constants / L542** | `RULE_LARGE_AMOUNT_LIMIT = 200,000.00`. | Implemented as `RULE_LARGE_TRANSFER_THRESHOLD = 200_000.0` in [`backend/app/core/config.py::Settings`](../backend/app/core/config.py). Value identical ($200,000.00$). | Implemented | N/A |
| **tab:constants / L544** | `JWT_EXPIRATION_MINUTES = 60`. | Implemented as `JWT_EXPIRES_IN_SECONDS = 3600` in [`backend/app/core/config.py::Settings`](../backend/app/core/config.py). | Implemented | Section 3.11 |
| **tab:constants / L546** | `TOP_K_EXPLANATION_FEATURES = 5`. | Top-5 features returned in [`PredictionOut.top_features`](../backend/app/models/schemas.py). | Implemented | N/A |
| **tab:states / L564** | `gateway_decision`: `LOW_RISK`, `HIGH_RISK`, `ANALYST_REVIEW`. | Serialized as Title Case strings (`"Low Risk"`, `"High Risk"`, `"Analyst Review"`) in [`GatewayDecision`](../backend/app/models/schemas.py). | Implemented | N/A |
| **tab:states / L566** | `explanation_status`: `STABLE`, `UNSTABLE`, `INSUFFICIENT_DATA`. | Serialized as `"Stable"`, `"Unstable"` in [`ExplanationStatus`](../backend/app/models/schemas.py). `INSUFFICIENT_DATA` is omitted; backend requires $\ge 2$ replicates at startup and fails closed to `"Unstable"` at runtime if $<2$ replicates are supplied. | Deviation | Section 3.10 |
| **tab:states / L568** | `analyst_outcome`: `PENDING`, `APPROVED_LEGITIMATE`, `CONFIRMED_FRAUD`. | In [`ReviewOutcome`](../backend/app/models/schemas.py), values are `"Confirmed Fraud"` and `"False Positive"`. Unreviewed state indicated by `latest_review: null` and `reviewed: false`. | Deviation | Section 3.8 |
| **tab:states / L570** | `resampling_regime`: `BASELINE_NONE`, `CLASS_WEIGHTED`, `SMOTE_ENN`. | Tracked as lowercase strings: `'none'`, `'class_weight'`, `'smote_enn'`. | Implemented | N/A |
| **tab:states / L572** | `alert_dispatch_status`: `NONE`, `DISPATCHED`, `FAILED`. | Not persisted in database or returned in API schemas. | Not implemented | N/A |
| **tab:api / L590** | `transaction_id`: String (UUIDv4). | Implemented as `transaction_id: str` in [`PredictionOut`](../backend/app/models/schemas.py). | Implemented | N/A |
| **tab:api / L592** | `fraud_risk_score`: Float. | Implemented as `risk_score: float` in [`PredictionOut`](../backend/app/models/schemas.py). | Deviation | Section 3.5 |
| **tab:api / L594** | `gateway_decision`: String (Enum). | Implemented as `gateway_decision: GatewayDecision` in [`PredictionOut`](../backend/app/models/schemas.py). | Implemented | N/A |
| **tab:api / L596** | `explanation_status`: String (Enum). | Implemented as `explanation_status: ExplanationStatus` in [`PredictionOut`](../backend/app/models/schemas.py). | Implemented | N/A |
| **tab:api / L598** | `top_features`: List[Object] (`feature_name`, `shap_value`, `direction`). | Implemented in [`FeatureExplanation`](../backend/app/models/schemas.py) using field name `feature` rather than `feature_name`. | Deviation | Section 3.5 |
| **tab:api / L600** | `rule_triggers`: List[String]. | Implemented as `rule_flags: list[str]` in [`PredictionOut`](../backend/app/models/schemas.py). | Deviation | Section 3.4 |
| **tab:api / L602** | `scored_at`: String (ISO 8601). | Omitted from [`PredictionOut`](../backend/app/models/schemas.py); persisted as `created_at: datetime` on parent record in database. | Deviation | Section 3.5 |
| **tab:api / L580-607** | *(Fields absent from SRS Output Schema Table tab:api)* | Telemetry addition `latency_ms: float` (hardcoded 0.0 in [`backend/app/services/scoring.py::ScoringService.predict`](../backend/app/services/scoring.py) until `fix/latency-ms-measurement` lands); metadata additions `imbalance_strategy: str` and `model_version: str`. | Addition / Pending | Section 3.5 |

---

## 3. Detailed Technical & Architectural Reconciliations

### 3.1. Endpoint Routing: `/api/v1/predict` vs `/api/v1/score`
- **SRS Text**: FR-6 ([`FraudLens_Gateway_SRS.tex:361`](FraudLens_Gateway_SRS.tex#L361)), NFR-1 ([`FraudLens_Gateway_SRS.tex:424`](FraudLens_Gateway_SRS.tex#L424)), and Table `\label{tab:api}` ([`FraudLens_Gateway_SRS.tex:580`](FraudLens_Gateway_SRS.tex#L580)) specify `POST /api/v1/score`.
- **Implementation**: Deployed on `POST /api/v1/predict` ([`backend/app/api/routes.py::predict`](../backend/app/api/routes.py)).
- **Status / Reason**: Deviation. Rationale not recorded; author to confirm.

### 3.2. Heuristic Rule Drain Ratio: `0.90` vs `0.95`
- **SRS Text**: FR-7 ([`FraudLens_Gateway_SRS.tex:362`](FraudLens_Gateway_SRS.tex#L362)) and Table `\label{tab:constants}` ([`FraudLens_Gateway_SRS.tex:540`](FraudLens_Gateway_SRS.tex#L540)) specify `RULE_DRAIN_BALANCE_RATIO = 0.95` ($\ge 95\%$).
- **Implementation**: Configured as `RULE_DRAIN_BALANCE_RATIO = 0.90` in [`backend/app/core/config.py::Settings`](../backend/app/core/config.py) and evaluated in [`backend/app/services/rules.py::check_origin_account_drained`](../backend/app/services/rules.py) (`tx.amount >= tx.old_balance_orig * 0.90`).
- **Status / Reason**: Deviation. Rationale not recorded; author to confirm.

### 3.3. Decision Thresholds and Unspecified Score Band
- **SRS Text**: Table `\label{tab:constants}` ([`FraudLens_Gateway_SRS.tex:534-536`](FraudLens_Gateway_SRS.tex#L534-L536)) specifies `FRAUD_THRESHOLD_HIGH = 0.75` and `FRAUD_THRESHOLD_LOW = 0.20`. The SRS text does not define how scores in the intermediate band $[0.20, 0.75)$ are classified.
- **Implementation**:
  - In heuristic stub mode (`MODEL_PATH` unset), defaults are `FRAUD_THRESHOLD_HIGH = 0.70` and `FRAUD_THRESHOLD_REVIEW = 0.40` ([`backend/app/core/config.py::Settings`](../backend/app/core/config.py)). Scores in $[0.40, 0.70)$ route to Analyst Review.
  - In trained model mode, thresholds are dynamically loaded from `manifest.json`, calibrated on validation unique scores (for deployed Variant B `class_weight`: $t_{\text{high}} = 0.9712$, $t_{\text{review}} = 0.9557$). Scores in $[t_{\text{review}}, t_{\text{high}})$ route to Analyst Review.
- **Status / Reason**: Deviation. Threshold calibration is derived empirically from the training/validation distribution rather than fixed static cutoffs.

### 3.4. Rule Flags Attribute Naming: `rule_flags` vs `rule_triggers`
- **SRS Text**: Table `\label{tab:api}` ([`FraudLens_Gateway_SRS.tex:600`](FraudLens_Gateway_SRS.tex#L600)) specifies `rule_triggers: List[String]`.
- **Implementation**: Named `rule_flags: list[str]` in [`backend/app/models/schemas.py::PredictionOut`](../backend/app/models/schemas.py).
- **Status / Reason**: Deviation. `rule_flags` was adopted in [`docs/api-contract.md`](api-contract.md) to distinguish passive boolean heuristic tags from active gateway decision triggers.

### 3.5. Scoring Output Telemetry and Field Differences
- **`latency_ms`**: Absent from SRS Table `\label{tab:api}` ([`FraudLens_Gateway_SRS.tex:580-607`](FraudLens_Gateway_SRS.tex#L580-L607)). Added to [`backend/app/models/schemas.py::PredictionOut`](../backend/app/models/schemas.py). Telemetry addition; hardcoded 0.0 in [`backend/app/services/scoring.py::ScoringService.predict`](../backend/app/services/scoring.py) until `fix/latency-ms-measurement` lands.
- **`imbalance_strategy` & `model_version`**: Absent from Table `\label{tab:api}`. Added to [`PredictionOut`](../backend/app/models/schemas.py) for runtime model artifact traceability.
- **`fraud_risk_score` vs `risk_score`**: Table `\label{tab:api}` specifies `fraud_risk_score`. Implemented as `risk_score` in [`PredictionOut`](../backend/app/models/schemas.py). Rationale not recorded; author to confirm.
- **`feature_name` vs `feature`**: Table `\label{tab:api}` specifies `feature_name` within `top_features` objects. Implemented as `feature` in [`backend/app/models/schemas.py::FeatureExplanation`](../backend/app/models/schemas.py).
- **`scored_at`**: Table `\label{tab:api}` specifies `scored_at` on the scoring response payload. Omitted from [`PredictionOut`](../backend/app/models/schemas.py); recorded as `created_at` on parent database records (`transactions` table).

### 3.6. Resampling Strategy Coverage for Baselines
- **SRS Text**: FR-2 ([`FraudLens_Gateway_SRS.tex:353`](FraudLens_Gateway_SRS.tex#L353)) states: *"The system shall train each classifier under three class-imbalance handling strategies: no resampling, class weighting, and SMOTE/SMOTE-ENN."*
- **Implementation**: Primary XGBoost is trained across all 3 strategies (`none`, `class_weight`, `smote_enn`). Baseline classifiers (Logistic Regression and Random Forest) were trained and evaluated strictly on the 10:1 subsampled baseline, recorded in [`ml/reports/metrics.json`](../ml/reports/metrics.json) under keys `strategy_comparison.logistic_regression` and `strategy_comparison.random_forest`. They were not fitted under `class_weight` or `smote_enn`.
- **Status / Reason**: Deviation. Baseline models serve as reference benchmarks on the subsampled baseline rather than full multi-strategy replicates.

### 3.7. Historical Transaction Filtering: Omission of Account Filter
- **SRS Text**: FR-11 ([`FraudLens_Gateway_SRS.tex:370`](FraudLens_Gateway_SRS.tex#L370)) states: *"The system shall allow retrieval of historical transactions and their audit records via a REST endpoint with date, account, and risk filtering."*
- **Implementation**: `GET /api/v1/transactions` ([`backend/app/api/routes.py::list_transactions`](../backend/app/api/routes.py)) supports time range filtering (`step_min`, `step_max`, `created_from`, `created_to`) and risk filtering (`decision`, `explanation_status`), but does not accept account identifier query parameters (`name_orig`, `name_dest`).
- **Status / Reason**: Deviation. Account filtering is not implemented in the current repository query layer.

### 3.8. Adjudication Outcomes: "False Positive" vs "Approved"
- **SRS Text**: FR-12 ([`FraudLens_Gateway_SRS.tex:371`](FraudLens_Gateway_SRS.tex#L371)) and Table `\label{tab:states}` ([`FraudLens_Gateway_SRS.tex:568`](FraudLens_Gateway_SRS.tex#L568)) specify analyst adjudication outcomes as `APPROVED_LEGITIMATE` / `Approved` and `CONFIRMED_FRAUD` / `Confirmed Fraud`. Table `\label{tab:states}` also defines a `PENDING` state.
- **Implementation**: [`backend/app/models/schemas.py::ReviewOutcome`](../backend/app/models/schemas.py) defines `"Confirmed Fraud"` and `"False Positive"`. "Approved" is represented as `"False Positive"`. Unreviewed transactions have `latest_review: null` and `reviewed: false` rather than a `"PENDING"` enum string.
- **Status / Reason**: Deviation. Rationale not recorded; author to confirm.

### 3.9. Frontend Component Integration Status
- **SRS Text**: FR-13 ([`FraudLens_Gateway_SRS.tex:376`](FraudLens_Gateway_SRS.tex#L376)), FR-14 ([`FraudLens_Gateway_SRS.tex:377`](FraudLens_Gateway_SRS.tex#L377)), FR-15 ([`FraudLens_Gateway_SRS.tex:378`](FraudLens_Gateway_SRS.tex#L378)), FR-16 ([`FraudLens_Gateway_SRS.tex:379`](FraudLens_Gateway_SRS.tex#L379)), FR-17 ([`FraudLens_Gateway_SRS.tex:380`](FraudLens_Gateway_SRS.tex#L380)), and NFR-2 ([`FraudLens_Gateway_SRS.tex:425`](FraudLens_Gateway_SRS.tex#L425)) specify dashboard views and update behavior.
- **Implementation**:
  - React components exist in [`frontend/src/components/`](../frontend/src/components/) and [`frontend/src/pages/Dashboard.jsx`](../frontend/src/pages/Dashboard.jsx). However, [`frontend/src/App.jsx`](../frontend/src/App.jsx) only exports the `App` component without invoking `createRoot` to mount into [`frontend/index.html`](../frontend/index.html) (`#root`). Thus, the UI components are present, but the live UI does not mount.
  - In [`frontend/src/components/MetricsPanel.jsx`](../frontend/src/components/MetricsPanel.jsx), only operational volume cards are rendered; PR-AUC curves and confusion matrices are not rendered.
  - In [`frontend/src/pages/Dashboard.jsx`](../frontend/src/pages/Dashboard.jsx), data polling is configured at 10 seconds (`setInterval(refreshData, 10000)`), exceeding the NFR-2 target of 3 seconds.
- **Status / Reason**: Pending integration / Deviation.

### 3.10. Explanation Status: Omission of `INSUFFICIENT_DATA`
- **SRS Text**: Table `\label{tab:states}` ([`FraudLens_Gateway_SRS.tex:566`](FraudLens_Gateway_SRS.tex#L566)) defines `explanation_status` values as `STABLE`, `UNSTABLE`, and `INSUFFICIENT_DATA`.
- **Implementation**: [`backend/app/models/schemas.py::ExplanationStatus`](../backend/app/models/schemas.py) defines only `Stable` and `Unstable`.
- **Verifiable Backend Behavior**:
  - At model bundle load time, [`backend/app/services/scoring.py::ScoringService.load_artifact`](../backend/app/services/scoring.py) requires at least 2 replicates (`len(self.replicates) >= 2`), failing application startup if fewer are present.
  - At request time, [`backend/app/services/explainer.py::RealExplainer.explain`](../backend/app/services/explainer.py) checks replicate count; if fewer than 2 replicates are supplied, it logs a warning and fails closed to `"Unstable"`, rather than emitting `"INSUFFICIENT_DATA"`.
- **Status / Reason**: Deviation.

### 3.11. Authentication Session Expiration: Fixed 1-Hour JWT vs Inactivity
- **SRS Text**: NFR-8 ([`FraudLens_Gateway_SRS.tex:435`](FraudLens_Gateway_SRS.tex#L435)) specifies: *"session expiration enforced after 60 minutes of inactivity"*. Table `\label{tab:constants}` ([`FraudLens_Gateway_SRS.tex:544`](FraudLens_Gateway_SRS.tex#L544)) specifies: `JWT_EXPIRATION_MINUTES = 60` (*"Lifetime duration of analyst JWT authentication tokens"*).
- **Implementation**: [`backend/app/core/config.py::Settings`](../backend/app/core/config.py) sets `JWT_EXPIRES_IN_SECONDS = 3600`. The token is a stateless signed JWT expiring 60 minutes after issuance (`exp = iat + 3600`).
- **Status / Reason**: Deviation from NFR-8 inactivity requirement (aligned with `tab:constants`). The system does not maintain server-side session state or a sliding-window inactivity timer.

### 3.12. Audit Trail Immutability Layer
- **SRS Text**: NFR-6 ([`FraudLens_Gateway_SRS.tex:433`](FraudLens_Gateway_SRS.tex#L433)) states: *"The system shall not permanently delete or overwrite audit log entries; corrections or analyst remediations shall be appended as new records, preserving a tamper-evident audit history."*
- **Implementation**: Immutability is enforced at the application / repository layer in [`backend/app/db/repository.py::TransactionRepository`](../backend/app/db/repository.py), which exposes only append methods (`save_prediction`, `add_review`) with no update or delete routes. Immutability is not enforced by database-level triggers, rule systems, or revoked PostgreSQL privileges. Cryptographic tamper-evidence (e.g. hash chains or signatures) is not implemented.
- **Status / Reason**: Implemented at application layer.

### 3.13. Python Runtime Pin: 3.13.5 vs 3.11+
- **SRS Text**: Chapter 2 ([`FraudLens_Gateway_SRS.tex:279`](FraudLens_Gateway_SRS.tex#L279)) and Chapter 3 ([`FraudLens_Gateway_SRS.tex:340`](FraudLens_Gateway_SRS.tex#L340)) specify `Python 3.11+`.
- **Implementation**: The repository, Dockerfiles, and virtual environments pin `Python 3.13.5`.
- **Status / Reason**: Implemented. Pinned version 3.13.5 satisfies the minimum requirement of 3.11+.
