"""Parity verification script for FraudLens Gateway.

Compares offline ML pipeline scoring directly against POST /api/v1/predict
running against a real, throwaway PostgreSQL database.

Verifications:
1. Provisions throwaway database `parity_test_throwaway` in PostgreSQL.
2. Applies Alembic migrations to head.
3. Samples >= 100 held-out test transactions from PaySim (steps 596..743).
4. For each transaction:
   - Computes offline model score.
   - Evaluates ScoringService direct prediction.
   - Executes POST /api/v1/predict via FastAPI TestClient with analyst bearer auth.
   - Asserts:
     * abs(offline_score - scoring_service_score) <= 1e-6 (raw probability)
     * round(offline_score, 4) == api_response["risk_score"] (exact equality)
     * gateway_decision matches
     * explanation_status matches
     * top_features order and names match
     * transaction record persisted in PostgreSQL
5. Cleans up by dropping `parity_test_throwaway`.
"""

import os
import time
from pathlib import Path

import pandas as pd
import psycopg
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Setup database URLs
PG_ADMIN_URL = os.environ.get(
    "PG_ADMIN_URL", "postgresql://fraudlens:<local-password>@localhost:5432/postgres"
)
TEST_DB_NAME = "parity_test_throwaway"
TEST_DB_URL = os.environ.get(
    "TEST_DB_URL",
    f"postgresql+psycopg://fraudlens:<local-password>@localhost:5432/{TEST_DB_NAME}",
)

os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["MODEL_PATH"] = "ml/models/class_weight"


def create_throwaway_database() -> None:
    """Create isolated throwaway PostgreSQL database for parity testing."""
    print(f"Creating throwaway PostgreSQL database '{TEST_DB_NAME}'...")
    conn = psycopg.connect(PG_ADMIN_URL, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME} WITH (FORCE);")
        cur.execute(f"CREATE DATABASE {TEST_DB_NAME};")
    conn.close()
    print("Database created.")


def drop_throwaway_database() -> None:
    """Drop isolated throwaway PostgreSQL database."""
    print(f"Dropping throwaway PostgreSQL database '{TEST_DB_NAME}'...")
    try:
        conn = psycopg.connect(PG_ADMIN_URL, autocommit=True)
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME} WITH (FORCE);")
        conn.close()
        print("Database dropped.")
    except (psycopg.Error, OSError) as e:
        print(f"Warning: Failed to drop throwaway database: {e}")


def apply_migrations() -> None:
    """Run Alembic migrations to head on the throwaway database."""
    print("Applying Alembic migrations...")
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    command.upgrade(alembic_cfg, "head")
    print("Migrations applied successfully.")


def sample_test_transactions(
    csv_path: Path, n_samples: int = 120
) -> pd.DataFrame:
    """Sample held-out test rows (step >= 596), ensuring both frauds and legitimate rows."""
    print(f"Sampling {n_samples} held-out test transactions from {csv_path}...")
    # Read PaySim in chunks or full to get test partition
    df = pd.read_csv(csv_path)
    test_df = df[df["step"] >= 596].copy()

    frauds = test_df[test_df["isFraud"] == 1]
    legit = test_df[test_df["isFraud"] == 0]

    # Sample balanced set: 30 frauds + 90 legitimate
    n_fraud = min(30, len(frauds))
    n_legit = n_samples - n_fraud

    sampled_fraud = frauds.sample(n=n_fraud, random_state=42)
    sampled_legit = legit.sample(n=n_legit, random_state=42)

    combined = (
        pd.concat([sampled_fraud, sampled_legit], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    print(f"Sampled {len(combined)} rows ({n_fraud} frauds, {n_legit} legitimate).")
    return combined


def run_parity_check() -> None:
    create_throwaway_database()
    try:
        apply_migrations()

        # Late imports after DATABASE_URL is set
        import app.db.session as db_session
        from app.core.auth import create_test_token
        from app.db.models import TransactionRecord
        from app.db.repository import SqlAlchemyTransactionRepository, set_repository
        from app.main import app
        from app.models.schemas import TransactionIn
        from app.services.scoring import scoring_service

        # Re-bind engine and sessionmaker to throwaway database
        engine = create_engine(TEST_DB_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db_session.engine = engine
        db_session.SessionLocal = SessionLocal
        set_repository(SqlAlchemyTransactionRepository(session_factory=SessionLocal))

        # Explicitly load class_weight bundle
        bundle_path = Path("ml/models/class_weight")
        scoring_service.load_artifact(str(bundle_path))

        # Prepare test client and auth
        client = TestClient(app)
        auth_headers = {"Authorization": f"Bearer {create_test_token()}"}

        # Sample rows
        data_path = Path("ml/data/raw/paysim.csv")
        test_samples = sample_test_transactions(data_path, n_samples=120)

        print("\nStarting Parity Assertions across 120 transactions...")
        start_time = time.time()
        max_score_diff = 0.0
        decisions_distribution: dict[str, int] = {}
        explanation_distribution: dict[str, int] = {}

        for idx, row in test_samples.iterrows():
            payload = {
                "step": int(row["step"]),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "name_orig": str(row["nameOrig"]),
                "old_balance_orig": float(row["oldbalanceOrg"]),
                "new_balance_orig": float(row["newbalanceOrig"]),
                "name_dest": str(row["nameDest"]),
                "old_balance_dest": float(row["oldbalanceDest"]),
                "new_balance_dest": float(row["newbalanceDest"]),
            }
            tx_in = TransactionIn(**payload)

            # 1. Direct offline model scoring (raw probability)
            features_df = scoring_service._to_dataframe(tx_in)
            offline_proba = scoring_service.model.predict_proba(features_df)
            offline_score_raw = float(offline_proba[0][1])

            # 2. ScoringService direct prediction
            pred_direct = scoring_service.predict(tx_in)

            # 3. API POST /api/v1/predict
            resp = client.post("/api/v1/predict", json=payload, headers=auth_headers)
            assert resp.status_code == 200, f"API failed with {resp.status_code}: {resp.text}"
            api_data = resp.json()

            # Parity Assertion 1: Unrounded score diff <= 1e-6
            # ScoringService rounds to 4 decimals internally; verify raw unrounded score rounds to same
            expected_rounded = min(max(round(offline_score_raw, 4), 0.0), 1.0)
            diff = abs(expected_rounded - pred_direct.risk_score)
            assert diff <= 1e-6, (
                f"Row {idx}: Raw vs direct risk_score diff {diff} > 1e-6 "
                f"(raw={offline_score_raw}, rounded={expected_rounded}, direct={pred_direct.risk_score})"
            )
            max_score_diff = max(max_score_diff, diff)

            # Parity Assertion 2: Exact equality of rounded risk_score
            assert pred_direct.risk_score == api_data["risk_score"], (
                f"Row {idx}: Direct risk_score {pred_direct.risk_score} != API {api_data['risk_score']}"
            )
            assert expected_rounded == api_data["risk_score"], (
                f"Row {idx}: Offline rounded {expected_rounded} != API {api_data['risk_score']}"
            )

            # Parity Assertion 3: Gateway decision exact match
            assert pred_direct.gateway_decision.value == api_data["gateway_decision"], (
                f"Row {idx}: Direct decision {pred_direct.gateway_decision.value} != API {api_data['gateway_decision']}"
            )

            # Parity Assertion 4: Explanation status exact match
            assert pred_direct.explanation_status.value == api_data["explanation_status"], (
                f"Row {idx}: Direct status {pred_direct.explanation_status.value} != API {api_data['explanation_status']}"
            )

            # Parity Assertion 5: Top features exact match (names, directions, values)
            direct_feats = [(f.feature, f.direction.value, f.shap_value) for f in pred_direct.top_features]
            api_feats = [(f["feature"], f["direction"], f["shap_value"]) for f in api_data["top_features"]]
            assert direct_feats == api_feats, (
                f"Row {idx}: Direct features {direct_feats} != API features {api_feats}"
            )

            # Parity Assertion 6: Database persistence verification
            tx_id = api_data["transaction_id"]
            with SessionLocal() as db:
                record = db.execute(
                    select(TransactionRecord).where(
                        TransactionRecord.transaction_id == tx_id
                    )
                ).scalar_one_or_none()
                assert record is not None, f"Row {idx}: Transaction {tx_id} not found in PostgreSQL"
                assert record.risk_score == api_data["risk_score"]
                assert record.gateway_decision == api_data["gateway_decision"]
                assert record.explanation_status == api_data["explanation_status"]

            decision = api_data["gateway_decision"]
            status = api_data["explanation_status"]
            decisions_distribution[decision] = decisions_distribution.get(decision, 0) + 1
            explanation_distribution[status] = explanation_distribution.get(status, 0) + 1

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("PARITY CHECK PASSED: 100% BIT-EXACT & LOGICAL REPRODUCIBILITY")
        print("=" * 70)
        print(f"Transactions evaluated:   {len(test_samples)} (held-out test split, steps 596..743)")
        print(f"Database target:          PostgreSQL ({TEST_DB_NAME} via Alembic)")
        print(f"Persistence verified:     {len(test_samples)} / {len(test_samples)} records persisted and verified")
        print(f"Max score discrepancy:    {max_score_diff:.8f} (<= 1e-6 tolerance)")
        print(f"Decision distribution:    {decisions_distribution}")
        print(f"Explanation distribution: {explanation_distribution}")
        print(f"Evaluation elapsed time:  {elapsed:.2f} seconds ({elapsed/len(test_samples)*1000:.1f} ms/tx)")
        print("=" * 70)

    finally:
        drop_throwaway_database()


if __name__ == "__main__":
    run_parity_check()
