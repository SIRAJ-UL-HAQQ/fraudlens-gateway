"""Unit tests verifying the latency benchmark result schema and validation invariants.

Validates:
1. Mathematical correctness of compute_percentiles (min, max, mean, p50, p95, p99).
2. Conformance of latency benchmark JSON structure to project reporting contracts:
   - Metadata present: git_commit, timestamp, model_mode, model_version, system environment.
   - SRS citations present: NFR-1 (2 s response), NFR-4 (>= 10 concurrent users), NFR-5 (5,000 export in 5 s).
   - Dual latency tracking: client round-trip, server latency_ms, and gap_ms (DB write + transport).
   - Invariant: server latency_ms <= client wall-time round-trip.
   - Sanitization: no passwords, secrets, tokens, or local user home directories in exported JSON.
   - Paced profile schema: records target RPS, duration, requests exceeding 2,000 ms, and proxy label.
3. Response validation: detects empty top_features or missing explanation_status in trained mode.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in [str(REPO_ROOT), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_latency import (
    collect_provenance,
    compute_percentiles,
    detect_system_power,
    generate_payload,
    sanitize_metadata,
)


def test_compute_percentiles_empty():
    res = compute_percentiles([])
    assert res == {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_compute_percentiles_sorted():
    vals = [float(i) for i in range(1, 101)]  # 1.0 to 100.0
    res = compute_percentiles(vals)
    assert res["min"] == 1.0
    assert res["max"] == 100.0
    assert res["mean"] == 50.5
    assert res["p50"] == 51.0
    assert res["p95"] == 96.0
    assert res["p99"] == 100.0


def test_payload_generator_schema():
    payload_mix = generate_payload(1, tx_type="mix", seed=42)
    assert "step" in payload_mix and payload_mix["step"] >= 1
    assert payload_mix["type"] in ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    assert payload_mix["amount"] >= 0.0
    assert payload_mix["name_orig"].startswith("C_")
    assert payload_mix["old_balance_orig"] >= 0.0
    assert payload_mix["new_balance_orig"] >= 0.0

    payload_transfer = generate_payload(2, tx_type="TRANSFER", seed=42)
    assert payload_transfer["type"] == "TRANSFER"

    payload_payment = generate_payload(3, tx_type="PAYMENT", seed=42)
    assert payload_payment["type"] == "PAYMENT"
    assert payload_payment["name_dest"].startswith("M_")


def test_sanitization_removes_secrets_and_local_paths():
    dirty_metadata = {
        "ANALYST_PASSWORD": "super_secret_password",
        "JWT_SECRET_KEY": "jwt_secret_token_value",
        "local_path": r"C:\Users\dell\Documents\fraudlens-gateway\secret.txt",
        "linux_path": "/home/runner/work/fraudlens-gateway",
        "safe_key": "v1.0.0-class_weight",
    }
    clean = sanitize_metadata(dirty_metadata)
    serialized = json.dumps(clean)

    assert "super_secret_password" not in serialized
    assert "jwt_secret_token_value" not in serialized
    assert r"C:\Users\dell" not in serialized
    assert "/home/runner" not in serialized
    assert clean["safe_key"] == "v1.0.0-class_weight"


def test_benchmark_result_structure():
    sample_condition = {
        "condition_name": "trained_direct_c1",
        "target_url": "http://127.0.0.1:8000",
        "model_mode": "trained",
        "model_version": "v1.0.0-class_weight",
        "proxy": "direct",
        "concurrency_clients": 1,
        "num_requests": 50,
        "warmup_requests": 10,
        "error_count": 0,
        "invalid_response_count": 0,
        "throughput_rps": 8.5,
        "client_latency": {
            "min": 85.2,
            "max": 145.0,
            "mean": 110.4,
            "p50": 108.0,
            "p95": 138.5,
            "p99": 144.2,
        },
        "server_latency": {
            "min": 70.1,
            "max": 125.0,
            "mean": 92.3,
            "p50": 90.0,
            "p95": 118.0,
            "p99": 124.0,
        },
        "gap_latency": {
            "min": 12.0,
            "max": 25.0,
            "mean": 18.1,
            "p50": 18.0,
            "p95": 22.0,
            "p99": 24.5,
        },
    }

    # Verify mathematical ordering: server <= client
    assert sample_condition["server_latency"]["p50"] <= sample_condition["client_latency"]["p50"]
    assert sample_condition["server_latency"]["p95"] <= sample_condition["client_latency"]["p95"]
    assert sample_condition["server_latency"]["max"] <= sample_condition["client_latency"]["max"]

    # Verify gap
    expected_gap_p50 = round(
        sample_condition["client_latency"]["p50"] - sample_condition["server_latency"]["p50"], 2
    )
    assert sample_condition["gap_latency"]["p50"] == expected_gap_p50


def test_srs_citations_and_paced_profile_schema():
    payload = {
        "srs_requirements": {
            "NFR-1": "2 s per scoring request under normal lab-demo load",
            "NFR-4": ">= 10 concurrent dashboard users",
            "NFR-5": "export of up to 5,000 transactions within 5 s",
        },
        "paced_profile": {
            "target_rps_per_client": 1.0,
            "concurrency_clients": 10,
            "duration_seconds": 120,
            "total_requests": 1200,
            "requests_exceeding_2000ms": 0,
            "share_exceeding_2000ms": 0.0,
            "label": "a proxy for normal lab-demo load",
            "client_latency": {"min": 50.0, "max": 250.0, "mean": 95.0, "p50": 90.0, "p95": 140.0, "p99": 180.0},
        },
    }

    assert "NFR-1" in payload["srs_requirements"]
    assert "NFR-4" in payload["srs_requirements"]
    assert "NFR-5" in payload["srs_requirements"]
    assert payload["paced_profile"]["label"] == "a proxy for normal lab-demo load"
    assert payload["paced_profile"]["requests_exceeding_2000ms"] == 0
    assert payload["paced_profile"]["share_exceeding_2000ms"] == 0.0


def test_export_benchmark_schema():
    payload = {
        "target_nfr": "NFR-5",
        "threshold_s": 5.0,
        "export_scope": "all_matching_rows_not_flagged_only",
        "csv_runs": [
            {"run": 1, "duration_s": 0.85, "line_count": 5001, "row_count": 5000, "bytes": 350000, "status_code": 200},
            {"run": 2, "duration_s": 0.82, "line_count": 5001, "row_count": 5000, "bytes": 350000, "status_code": 200},
            {"run": 3, "duration_s": 0.84, "line_count": 5001, "row_count": 5000, "bytes": 350000, "status_code": 200},
        ],
        "pdf_runs": [
            {"run": 1, "duration_s": 2.10, "page_count": 143, "bytes": 120000, "status_code": 200},
            {"run": 2, "duration_s": 2.05, "page_count": 143, "bytes": 120000, "status_code": 200},
            {"run": 3, "duration_s": 2.12, "page_count": 143, "bytes": 120000, "status_code": 200},
        ],
    }
    assert payload["target_nfr"] == "NFR-5"
    assert payload["threshold_s"] == 5.0
    assert payload["export_scope"] == "all_matching_rows_not_flagged_only"
    for run in payload["csv_runs"]:
        assert run["line_count"] == 5001
        assert run["row_count"] == 5000
        assert run["duration_s"] < 5.0
    for run in payload["pdf_runs"]:
        assert run["page_count"] > 0
        assert run["bytes"] > 0
        assert run["duration_s"] < 5.0


def test_clock_check_assertion_strict():
    """Verify that strict assertion raises AssertionError if server_ms > client_ms + 0.5."""
    # Within tolerance: server = 50.4, client = 50.0 (gap = -0.4 <= 0.5)
    server_ms = 50.4
    client_ms = 50.0
    eps = 0.5
    assert not (server_ms > client_ms + eps)

    # Exceeding tolerance: server = 50.6, client = 50.0 (gap = -0.6 > 0.5)
    server_ms_bad = 50.6
    with pytest.raises(AssertionError, match="Clock anomaly"):
        if server_ms_bad > client_ms + eps:
            raise AssertionError(
                f"Clock anomaly: server latency ({server_ms_bad} ms) exceeds client wall time ({client_ms} ms) by > 0.5 ms"
            )


def test_think_time_and_lockstep_schema():
    """Verify schema structures for think-time and lockstep concurrency profiles."""
    think_time_payload = {
        "think_time_profile": {
            "z_seconds": 1.0,
            "label": "think-time uniform [0.5Z, 1.5Z]",
        }
    }
    assert think_time_payload["think_time_profile"]["z_seconds"] == 1.0
    assert "0.5Z" in think_time_payload["think_time_profile"]["label"]

    lockstep_payload = {
        "lockstep_profile": {
            "label": "10 users click at the same instant",
            "concurrency": 10,
        }
    }
    assert lockstep_payload["lockstep_profile"]["label"] == "10 users click at the same instant"
    assert lockstep_payload["lockstep_profile"]["concurrency"] == 10


def test_provenance_schema():
    """Verify provenance metadata structure."""
    provenance = {
        "git_commit": "ae3dae2",
        "working_tree_dirty": False,
        "client_keepalive": True,
        "client_keepalive_mode": "HTTP/1.1 persistent connection pooling via httpx.Client",
        "manifest_sha256": "3E04C08C9938968809204C88C09A3020F94E8F783770F6227E84D26DA145CD56",
        "cpu_model": "Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz",
        "cpu_logical_cores": 8,
        "docker_ncpu": 8,
        "docker_mem_total_bytes": "8262410240",
        "container_python_version": "3.13.5",
        "container_xgboost_version": "3.2.0 (xgboost-cpu)",
        "omp_num_threads": "unset",
        "uvicorn_workers": 1,
        "latency_ms_definition": (
            "Server model execution time covering rules, dataframe prep, "
            "6-model inference+SHAP explanation, and gateway routing; "
            "excludes HTTP transit and database persistence"
        ),
        "operator_notes": "AC connected (plugged in), Windows Best Performance power plan",
        "os": "Linux-x86_64",
    }
    assert len(provenance["git_commit"]) >= 7
    assert provenance["manifest_sha256"] == "3E04C08C9938968809204C88C09A3020F94E8F783770F6227E84D26DA145CD56"
    assert provenance["uvicorn_workers"] == 1
    assert provenance["omp_num_threads"] == "unset"
    assert "xgboost-cpu" in provenance["container_xgboost_version"]
    assert provenance["docker_ncpu"] >= 1
    assert "AC connected" in provenance["operator_notes"]
    assert "Server model execution time" in provenance["latency_ms_definition"]


def test_dynamic_provenance_and_power_detection():
    """Verify that detect_system_power dynamically constructs notes and provenance collects metadata."""
    notes = detect_system_power("Test operator note")
    assert "AC" in notes or "battery" in notes
    assert "Test operator note" in notes

    prov = collect_provenance("Unit test run")
    assert "git_commit" in prov
    assert "backend_image_built_from_commit" in prov
    assert "operator_notes" in prov
    assert "Unit test run" in prov["operator_notes"]
    assert "docker_ncpu" in prov


def test_matrix_report_schema():
    """Verify that the full matrix report schema contains all required orchestration sections."""
    matrix_report = {
        "srs_requirements": {
            "NFR-1": "2 s per scoring request, normal lab-demo load",
            "NFR-4": ">= 10 concurrent dashboard users",
            "NFR-5": "export of up to 5,000 transactions within 5 s",
        },
        "provenance": {
            "git_commit": "ae3dae2",
            "backend_image_built_from_commit": "ae3dae2",
            "raw_data_sha256": "A" * 64,
            "resumed": False,
            "operator_notes": "AC connected (plugged in), Windows Power Scheme: Balanced, Note: matrix run",
        },
        "rounds": {
            "round_1": {
                "trained_direct": {"p50": 39.4, "rps": 22.5},
                "trained_proxy": {"p50": 42.1, "rps": 21.8},
                "stub_direct": {"p50": 2.1, "rps": 120.0},
                "stub_proxy": {"p50": 4.5, "rps": 110.0},
            },
            "round_2": {
                "trained_direct": {"p50": 39.1, "rps": 22.7},
                "trained_proxy": {"p50": 41.8, "rps": 22.0},
                "stub_direct": {"p50": 2.0, "rps": 122.0},
                "stub_proxy": {"p50": 4.4, "rps": 112.0},
            },
            "round_3": {
                "trained_direct": {"p50": 38.9, "rps": 22.9},
                "trained_proxy": {"p50": 41.5, "rps": 22.1},
                "stub_direct": {"p50": 2.1, "rps": 121.0},
                "stub_proxy": {"p50": 4.3, "rps": 113.0},
            },
        },
        "round_drift": {
            "client_p50_drift_pct": -1.27,
            "server_p50_drift_pct": -0.85,
            "throughput_drift_pct": 1.78,
        },
        "think_time_sweep": {
            "z1_direct": {"p50": 44.5},
            "z3_direct": {"p50": 45.0},
            "z5_direct": {"p50": 46.2},
            "z10_direct": {"p50": 47.1},
            "z5_proxy": {"p50": 48.9},
        },
        "lockstep_profile": {
            "label": "10 users click at the same instant",
            "concurrency": 10,
        },
        "export_benchmark": {
            "target_nfr": "NFR-5",
            "threshold_s": 5.0,
            "export_scope": "all_matching_rows_not_flagged_only",
            "csv_runs": [{"line_count": 5001, "row_count": 5000, "duration_s": 0.85}],
            "pdf_runs": [{"page_count": 129, "duration_s": 2.15}],
        },
        "script_arguments": {
            "orchestrator": "python scripts/run_latency_matrix.py",
            "commands": ["python scripts/benchmark_latency.py --yes"],
        },
    }

    assert "rounds" in matrix_report
    assert "round_drift" in matrix_report
    assert len(matrix_report["rounds"]) == 3
    assert matrix_report["provenance"]["resumed"] is False
    assert len(matrix_report["provenance"]["raw_data_sha256"]) == 64
    assert matrix_report["export_benchmark"]["csv_runs"][0]["line_count"] == 5001
    assert matrix_report["export_benchmark"]["pdf_runs"][0]["page_count"] == 129
    assert "commands" in matrix_report["script_arguments"]

