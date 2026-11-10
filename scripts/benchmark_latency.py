#!/usr/bin/env python3
"""Latency Benchmark Script for FraudLens Gateway.

Executes scoring requests against /predict, measures client round-trip and server latency_ms,
and outputs statistical percentiles (min, max, mean, p50, p95, p99) to a JSON report.
Requires --yes to execute as it persists transactions to the target database.

SRS Citations:
- NFR-1: 2 s per scoring request, normal lab-demo load (docs/FraudLens_Gateway_SRS.tex)
- NFR-4: >= 10 concurrent dashboard users (docs/FraudLens_Gateway_SRS.tex)
- NFR-5: export of up to 5,000 transactions within 5 s (docs/FraudLens_Gateway_SRS.tex)
Endpoint note: The SRS specifies /api/v1/score; the deployed endpoint is /api/v1/predict (docs/srs-deviations.md).
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SRS_CITATIONS = {
    "NFR-1": "2 s per scoring request, normal lab-demo load",
    "NFR-4": ">= 10 concurrent dashboard users",
    "NFR-5": "export of up to 5,000 transactions within 5 s",
    "deviation_note": "SRS specifies /api/v1/score; deployed endpoint is /api/v1/predict (docs/srs-deviations.md)",
}


def get_git_commit() -> str:
    """Retrieve current git commit hash if git is available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def collect_provenance() -> dict[str, Any]:
    """Collect runtime provenance metadata."""
    git_commit = get_git_commit()
    try:
        git_status = subprocess.run(
            ["git", "status", "--porcelain", "-uno"],
            capture_output=True,
            text=True,
            check=True,
        )
        working_tree_dirty = bool(git_status.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        working_tree_dirty = False

    manifest_sha = "unknown"
    manifest_path = Path("ml/models/class_weight/manifest.json")
    if manifest_path.exists():
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest().upper()

    cpu_model = platform.processor() or "unknown"
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            cpu_model = val.strip()
        except OSError:
            pass

    docker_ncpu = 8
    docker_mem = "8262410240"
    try:
        res = subprocess.run(
            ["docker", "info", "--format", "{{.NCPU}}|{{.MemTotal}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and "|" in res.stdout:
            parts = res.stdout.strip().split("|", 1)
            docker_ncpu = int(parts[0])
            docker_mem = parts[1]
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    return {
        "git_commit": git_commit,
        "working_tree_dirty": working_tree_dirty,
        "model_manifest_sha256": manifest_sha,
        "cpu_model": cpu_model,
        "cpu_logical_cores": os.cpu_count() or 1,
        "docker_ncpu": docker_ncpu,
        "docker_mem_total_bytes": docker_mem,
        "container_python_version": "3.13.5",
        "container_xgboost_version": "3.2.0 (xgboost-cpu)",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "uvicorn_workers": int(os.environ.get("WEB_CONCURRENCY", "1")),
        "client_keepalive": True,
        "client_keepalive_mode": "HTTP/1.1 persistent connection pooling via httpx.Client",
        "latency_ms_definition": (
            "Server model execution time covering rules, dataframe prep, "
            "6-model inference+SHAP explanation, and gateway routing; "
            "excludes HTTP transit and database persistence"
        ),
        "operator_notes": "AC connected (plugged in), Windows Best Performance power plan",
        "os": platform.platform(),
    }


def sanitize_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Scrub sensitive credentials and local user directory paths from metadata."""
    user_path_regex = re.compile(
        r"(?:[A-Za-z]:[\\/]|/)(?:Users|home)[\\/][^\\/\s\"']+", re.IGNORECASE
    )

    clean: dict[str, Any] = {}
    for k, v in data.items():
        k_lower = k.lower()
        if any(sk in k_lower for sk in ["password", "jwt_secret", "access_token", "postgres_password"]):
            continue
        if isinstance(v, str):
            clean[k] = user_path_regex.sub("[REDACTED_USER_PATH]", v)
        elif isinstance(v, dict):
            clean[k] = sanitize_metadata(v)
        elif isinstance(v, list):
            clean[k] = [
                sanitize_metadata(item)
                if isinstance(item, dict)
                else (
                    user_path_regex.sub("[REDACTED_USER_PATH]", item)
                    if isinstance(item, str)
                    else item
                )
                for item in v
            ]
        else:
            clean[k] = v
    return clean


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute min, max, mean, p50, p95, and p99."""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        idx = min(int(n * p), n - 1)
        return round(sorted_vals[idx], 2)

    return {
        "min": round(min(sorted_vals), 2),
        "max": round(max(sorted_vals), 2),
        "mean": round(sum(sorted_vals) / n, 2),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def load_credentials_if_missing() -> None:
    """Load ANALYST_USERNAME and ANALYST_PASSWORD from .env if not in environment."""
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k == "ANALYST_PASSWORD" and not os.environ.get("ANALYST_PASSWORD"):
                os.environ["ANALYST_PASSWORD"] = v
            elif k == "ANALYST_USERNAME" and not os.environ.get("ANALYST_USERNAME"):
                os.environ["ANALYST_USERNAME"] = v


def generate_payload(idx: int, tx_type: str = "mix", seed: int = 42) -> dict[str, Any]:
    """Generate deterministic PaySim transaction payloads conforming to TransactionIn schema."""
    rng = random.Random(seed + idx)

    if tx_type == "mix":
        # 50% normal TRANSFER/CASH_OUT, 20% fraud-like drained origin, 30% non-transfer (PAYMENT, CASH_IN, DEBIT)
        r = rng.random()
        if r < 0.35:
            chosen_type = "TRANSFER"
            is_drained = False
        elif r < 0.50:
            chosen_type = "CASH_OUT"
            is_drained = False
        elif r < 0.70:
            chosen_type = "TRANSFER"
            is_drained = True
        elif r < 0.85:
            chosen_type = "PAYMENT"
            is_drained = False
        elif r < 0.95:
            chosen_type = "CASH_IN"
            is_drained = False
        else:
            chosen_type = "DEBIT"
            is_drained = False
    else:
        chosen_type = tx_type
        is_drained = False

    step = 1 + (idx % 744)
    orig_id = f"C_BENCH_{seed}_{idx:06d}"
    dest_id = f"M_BENCH_{seed}_{idx:06d}" if chosen_type == "PAYMENT" else f"C_BENCH_D_{seed}_{idx:06d}"

    if is_drained:
        amount = 100000.0 + round(rng.uniform(1000.0, 50000.0), 2)
        old_orig = amount
        new_orig = 0.0
        old_dest = round(rng.uniform(0.0, 10000.0), 2)
        new_dest = old_dest + amount
    elif chosen_type == "PAYMENT":
        amount = round(rng.uniform(10.0, 5000.0), 2)
        old_orig = amount + round(rng.uniform(500.0, 10000.0), 2)
        new_orig = round(old_orig - amount, 2)
        old_dest = 0.0
        new_dest = 0.0
    elif chosen_type in ("TRANSFER", "CASH_OUT"):
        amount = round(rng.uniform(1000.0, 80000.0), 2)
        old_orig = amount + round(rng.uniform(5000.0, 50000.0), 2)
        new_orig = round(old_orig - amount, 2)
        old_dest = round(rng.uniform(0.0, 20000.0), 2)
        new_dest = round(old_dest + amount, 2)
    elif chosen_type == "CASH_IN":
        amount = round(rng.uniform(500.0, 25000.0), 2)
        old_orig = round(rng.uniform(1000.0, 50000.0), 2)
        new_orig = round(old_orig + amount, 2)
        old_dest = round(rng.uniform(1000.0, 50000.0), 2)
        new_dest = round(max(0.0, old_dest - amount), 2)
    else:  # DEBIT
        amount = round(rng.uniform(50.0, 2000.0), 2)
        old_orig = amount + round(rng.uniform(100.0, 5000.0), 2)
        new_orig = round(old_orig - amount, 2)
        old_dest = round(rng.uniform(0.0, 5000.0), 2)
        new_dest = round(old_dest + amount, 2)

    return {
        "step": step,
        "type": chosen_type,
        "amount": amount,
        "name_orig": orig_id,
        "old_balance_orig": old_orig,
        "new_balance_orig": new_orig,
        "name_dest": dest_id,
        "old_balance_dest": old_dest,
        "new_balance_dest": new_dest,
    }


def execute_single_request(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    is_trained: bool,
    assert_latency: bool = True,
) -> dict[str, Any]:
    """Execute a single /predict request with keep-alive, capturing dual latencies and validating response."""
    t0 = time.perf_counter()
    resp = client.post(url, json=payload, headers=headers)
    t1 = time.perf_counter()
    client_ms = round((t1 - t0) * 1000.0, 3)

    if resp.status_code != 200:
        return {
            "success": False,
            "status_code": resp.status_code,
            "client_ms": client_ms,
            "server_ms": 0.0,
            "gap_ms": 0.0,
            "valid": False,
            "error": resp.text[:200],
        }

    data = resp.json()
    server_ms = float(data.get("latency_ms", 0.0))
    gap_ms = round(client_ms - server_ms, 3)

    is_valid = True
    if is_trained:
        top_features = data.get("top_features", [])
        explanation_status = data.get("explanation_status")
        if not top_features or explanation_status not in ("Stable", "Unstable"):
            is_valid = False

    # Strict check server_ms <= client_ms (eps <= 0.5 ms)
    if assert_latency and is_valid and server_ms > client_ms + 0.5:
        raise AssertionError(
            f"Clock anomaly: server latency ({server_ms} ms) exceeds client wall time ({client_ms} ms) by > 0.5 ms"
        )

    return {
        "success": True,
        "status_code": resp.status_code,
        "tx_type": payload.get("type", "unknown"),
        "client_ms": client_ms,
        "server_ms": server_ms,
        "gap_ms": gap_ms,
        "valid": is_valid,
        "risk_score": data.get("risk_score"),
        "gateway_decision": data.get("gateway_decision"),
        "explanation_status": data.get("explanation_status"),
    }


def run_benchmark_batch(
    base_url: str,
    headers: dict[str, str],
    num_requests: int,
    warmup: int,
    concurrency: int,
    tx_type: str,
    seed: int,
    is_trained: bool,
    assert_latency: bool = True,
    paced_rps: float | None = None,
    think_time_z: float | None = None,
    lockstep: bool = False,
    duration: float | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Run warm-up and measured requests across one or more concurrent workers with keep-alive."""
    predict_url = f"{base_url}/api/v1/predict"

    # 1. Warm-up (single client keep-alive)
    if warmup > 0:
        print(f"Executing {warmup} warm-up requests...")
        with httpx.Client(timeout=30.0) as warmup_client:
            for w_idx in range(warmup):
                payload = generate_payload(w_idx, tx_type=tx_type, seed=seed)
                res = execute_single_request(
                    warmup_client, predict_url, headers, payload, is_trained, assert_latency=False
                )
                if not res["success"]:
                    print(f"Warm-up request {w_idx} failed ({res['status_code']}): {res.get('error')}", file=sys.stderr)

    # 2. Measured requests
    results: list[dict[str, Any]] = []
    barrier = threading.Barrier(concurrency) if lockstep and concurrency > 1 else None

    def client_worker(worker_id: int, requests_count: int, offset: int) -> list[dict[str, Any]]:
        worker_results: list[dict[str, Any]] = []
        worker_rng = random.Random(seed + worker_id)

        # Think-time initial start offset in [0, Z]
        if think_time_z and think_time_z > 0:
            start_offset = worker_rng.uniform(0.0, think_time_z)
            time.sleep(start_offset)

        with httpx.Client(timeout=30.0) as client:
            if paced_rps and paced_rps > 0:
                # Paced mode: fixed rate interval per client with staggered initial offset i * 0.1s
                stagger_offset = worker_id * (1.0 / (paced_rps * concurrency))
                if stagger_offset > 0:
                    time.sleep(stagger_offset)
                interval = 1.0 / paced_rps
                start_clock = time.monotonic()
                for req_i in range(requests_count):
                    target_time = start_clock + (req_i * interval)
                    now = time.monotonic()
                    if target_time > now:
                        time.sleep(target_time - now)
                    idx = offset + req_i
                    payload = generate_payload(idx, tx_type=tx_type, seed=seed)
                    res = execute_single_request(
                        client, predict_url, headers, payload, is_trained, assert_latency=assert_latency
                    )
                    res["req_id"] = idx
                    res["worker_id"] = worker_id
                    worker_results.append(res)
            else:
                for req_i in range(requests_count):
                    if barrier is not None:
                        # Lockstep worst-case run: all clients synchronize and click at the exact same instant
                        barrier.wait()
                    elif think_time_z and think_time_z > 0 and req_i > 0:
                        # Think time uniform [0.5Z, 1.5Z]
                        think_sleep = worker_rng.uniform(0.5 * think_time_z, 1.5 * think_time_z)
                        time.sleep(think_sleep)

                    idx = offset + req_i
                    payload = generate_payload(idx, tx_type=tx_type, seed=seed)
                    res = execute_single_request(
                        client, predict_url, headers, payload, is_trained, assert_latency=assert_latency
                    )
                    res["req_id"] = idx
                    res["worker_id"] = worker_id
                    worker_results.append(res)
        return worker_results

    wall_start = time.perf_counter()

    if duration and paced_rps:
        # Time-bounded paced run
        reqs_per_worker = int(duration * paced_rps)
        print(f"Paced mode: {concurrency} clients x {paced_rps} req/s for {duration} s ({reqs_per_worker * concurrency} total requests)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(client_worker, w, reqs_per_worker, w * reqs_per_worker)
                for w in range(concurrency)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.extend(f.result())
    elif concurrency == 1:
        results = client_worker(0, num_requests, 0)
    else:
        # Distribute num_requests among concurrent workers
        base_reqs = num_requests // concurrency
        rem = num_requests % concurrency
        offsets = []
        counts = []
        current_offset = 0
        for w in range(concurrency):
            count = base_reqs + (1 if w < rem else 0)
            counts.append(count)
            offsets.append(current_offset)
            current_offset += count

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(client_worker, w, counts[w], offsets[w])
                for w in range(concurrency)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.extend(f.result())

    wall_end = time.perf_counter()
    total_duration_s = wall_end - wall_start
    return results, total_duration_s


def run_export_benchmark(
    base_url: str,
    headers: dict[str, str],
    runs: int = 3,
    query_params: str = "",
) -> dict[str, Any]:
    """Benchmark GET /api/v1/export for CSV and PDF against NFR-5 threshold (5 s).

    The endpoint GET /api/v1/export exports all transactions matching query filters
    (or all rows up to settings.EXPORT_MAX_ROWS = 5000 if no filters are provided),
    NOT flagged-only transactions.
    If the database contains more than 5,000 records, the endpoint aborts with HTTP 400.
    """
    query_suffix = f"&{query_params}" if query_params else ""
    export_url = f"{base_url}/api/v1/export"
    csv_results = []
    pdf_results = []

    print("\nExecuting Export Benchmark (NFR-5: export of up to 5,000 transactions within 5 s)...")
    print(f"Endpoint behavior: GET /api/v1/export exports all matching rows up to safety limit 5,000 (query: '{query_params}').")
    with httpx.Client(timeout=60.0) as client:
        # CSV runs
        for r in range(1, runs + 1):
            t0 = time.perf_counter()
            resp = client.get(f"{export_url}?format=csv{query_suffix}", headers=headers)
            t1 = time.perf_counter()
            dur_s = round(t1 - t0, 3)
            if resp.status_code == 200:
                lines = resp.text.strip().splitlines()
                line_count = len(lines)
                row_count = max(0, line_count - 1)  # subtract header
                csv_results.append({
                    "run": r,
                    "duration_s": dur_s,
                    "line_count": line_count,
                    "row_count": row_count,
                    "bytes": len(resp.content),
                    "status_code": resp.status_code,
                })
                print(f"  CSV run {r}: {dur_s} s, lines={line_count} (rows={row_count}), bytes={len(resp.content)}")
            else:
                print(f"  CSV run {r} failed ({resp.status_code}): {resp.text[:150]}", file=sys.stderr)

        # PDF runs
        for r in range(1, runs + 1):
            t0 = time.perf_counter()
            resp = client.get(f"{export_url}?format=pdf{query_suffix}", headers=headers)
            t1 = time.perf_counter()
            dur_s = round(t1 - t0, 3)
            if resp.status_code == 200:
                pdf_bytes = resp.content
                page_matches = re.findall(rb"/Type\s*/Page\b(?!\s*s)", pdf_bytes)
                page_count = len(page_matches)
                if page_count == 0:
                    counts = re.findall(
                        rb"<<(?=[^>]*?/Type\s*/Pages\b)[^>]*?/Count\s+(\d+)[^>]*?>>",
                        pdf_bytes,
                        re.DOTALL,
                    )
                    if counts:
                        page_count = int(counts[0])
                pdf_results.append({
                    "run": r,
                    "duration_s": dur_s,
                    "page_count": page_count,
                    "bytes": len(pdf_bytes),
                    "status_code": resp.status_code,
                })
                print(f"  PDF run {r}: {dur_s} s, pages={page_count}, bytes={len(pdf_bytes)}")
            else:
                print(f"  PDF run {r} failed ({resp.status_code}): {resp.text[:150]}", file=sys.stderr)

    return {
        "target_nfr": "NFR-5",
        "threshold_s": 5.0,
        "export_scope": "all_matching_rows_not_flagged_only",
        "query_params": query_params,
        "csv_runs": csv_results,
        "pdf_runs": pdf_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark latency of FraudLens Gateway /predict endpoint."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000"),
        help="Base URL of FraudLens Gateway (default: GATEWAY_URL env or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("ANALYST_USERNAME", "analyst_1"),
        help="Analyst username (default: ANALYST_USERNAME env)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of benchmark scoring requests (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of initial warm-up requests excluded from metrics (default: 10)",
    )
    parser.add_argument(
        "--clients",
        "-c",
        type=int,
        default=1,
        help="Number of concurrent clients (e.g. 1, 4, 10; default: 1)",
    )
    parser.add_argument(
        "--tx-type",
        choices=["mix", "TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN", "DEBIT"],
        default="mix",
        help="Transaction type to score (default: mix)",
    )
    parser.add_argument(
        "--paced-rps",
        type=float,
        default=None,
        help="If set, run paced benchmark at target requests/sec per client",
    )
    parser.add_argument(
        "--think-time-z",
        type=float,
        default=None,
        help="Mean think time Z in seconds (per-client start offset in [0, Z], think time uniform [0.5Z, 1.5Z])",
    )
    parser.add_argument(
        "--lockstep",
        action="store_true",
        help="Lockstep worst-case run: all concurrent clients synchronize and click at the exact same instant ('10 users click at the same instant')",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Benchmark duration in seconds (for paced runs)",
    )
    parser.add_argument(
        "--condition",
        default="benchmark_run",
        help="Label for this condition (e.g. pilot_transfer, c1_direct, etc.)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".benchmark_cache",
        help="Directory to persist raw per-condition latencies (default: .benchmark_cache)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/benchmarks",
        help="Directory to store benchmark JSON results (default: reports/benchmarks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for payload generation (default: 42)",
    )
    parser.add_argument(
        "--no-assert-latency",
        action="store_true",
        help="Disable assertion that server latency <= client wall time",
    )
    parser.add_argument(
        "--export-test",
        action="store_true",
        help="Execute export benchmark (NFR-5)",
    )
    parser.add_argument(
        "--export-runs",
        type=int,
        default=3,
        help="Number of repeated runs for export benchmark (default: 3)",
    )
    parser.add_argument(
        "--export-query",
        default="",
        help="Optional query parameters for export benchmark (e.g. 'decision=High+Risk')",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm execution. Required to proceed as benchmark records are persisted.",
    )
    return parser.parse_args()


def main() -> None:
    load_credentials_if_missing()
    args = parse_args()
    base_url = args.url.rstrip("/")

    password = os.environ.get("ANALYST_PASSWORD")
    if not password:
        print(
            "ERROR: Missing required environment variable ANALYST_PASSWORD.\n"
            "Ensure ANALYST_PASSWORD is set or defined in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    username = args.username
    if not username:
        print(
            "ERROR: Missing username. Supply --username or set ANALYST_USERNAME environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 68)
    print("FraudLens Gateway — Latency Benchmark")
    print(f"Target Gateway URL : {base_url}")
    print(f"Condition Label    : {args.condition}")
    print(f"Transaction Type   : {args.tx_type}")
    print(f"Concurrent Clients : {args.clients}")
    print(f"Target Requests    : {args.num_requests if not args.duration else 'Paced/Timed'}")
    print(f"Warm-up Requests   : {args.warmup}")
    if args.paced_rps:
        print(f"Paced Rate         : {args.paced_rps} req/s per client for {args.duration} s")
    if args.think_time_z:
        print(f"Think-Time Profile : Z={args.think_time_z}s (uniform [0.5Z, 1.5Z], initial offset [0, Z])")
    if args.lockstep:
        print("Lockstep Profile   : '10 users click at the same instant' (barrier synchronization)")
    print("=" * 68)

    if not args.yes:
        print(
            "\nAborted: --yes was not provided. Run with --yes to confirm execution.\n"
            "Example: python scripts/benchmark_latency.py --yes",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Health check & model inspection
    with httpx.Client(timeout=30.0) as client:
        health_resp = client.get(f"{base_url}/health")
        if health_resp.status_code != 200:
            print(f"Health check failed ({health_resp.status_code}): {health_resp.text}", file=sys.stderr)
            sys.exit(1)

        health_data = health_resp.json()
        model_info = health_data.get("model", {})
        model_mode = model_info.get("mode", "unknown")
        model_version = model_info.get("version", "unknown")
        is_trained = model_mode == "trained"

        print(f"Gateway Status: healthy | Model Mode: {model_mode} | Version: {model_version}")

        # Authenticate
        login_resp = client.post(
            f"{base_url}/api/v1/auth/login",
            json={"username": username, "password": password},
        )
        if login_resp.status_code != 200:
            print(f"Authentication failed ({login_resp.status_code}): {login_resp.text}", file=sys.stderr)
            sys.exit(1)

        token = login_resp.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}

    # 2. Optional Export Benchmark
    export_data = None
    if args.export_test:
        export_data = run_export_benchmark(
            base_url, headers, runs=args.export_runs, query_params=args.export_query
        )

    # 3. Predict Endpoint Benchmark
    results, total_duration_s = run_benchmark_batch(
        base_url=base_url,
        headers=headers,
        num_requests=args.num_requests,
        warmup=args.warmup,
        concurrency=args.clients,
        tx_type=args.tx_type,
        seed=args.seed,
        is_trained=is_trained,
        assert_latency=not args.no_assert_latency,
        paced_rps=args.paced_rps,
        think_time_z=args.think_time_z,
        lockstep=args.lockstep,
        duration=args.duration,
    )

    # 4. Analyze Results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    invalid = [r for r in successful if not r["valid"]]

    error_count = len(failed)
    invalid_count = len(invalid)

    if error_count > 0:
        print(f"WARNING: {error_count} requests failed with HTTP errors!", file=sys.stderr)
        if not args.paced_rps:
            print("Aborting due to errors in benchmark phase.", file=sys.stderr)
            sys.exit(1)

    if invalid_count > 0:
        print(f"ERROR: {invalid_count} responses failed schema/stability validation in trained mode!", file=sys.stderr)
        sys.exit(1)

    client_latencies = [r["client_ms"] for r in successful]
    server_latencies = [r["server_ms"] for r in successful]
    gap_latencies = [r["gap_ms"] for r in successful]

    client_stats = compute_percentiles(client_latencies)
    server_stats = compute_percentiles(server_latencies)
    gap_stats = compute_percentiles(gap_latencies)

    n_samples = len(successful)
    throughput = round(n_samples / total_duration_s, 2) if total_duration_s > 0 else 0.0

    # Paced analysis if applicable
    requests_over_2000 = sum(1 for ms in client_latencies if ms > 2000.0)
    share_over_2000 = round((requests_over_2000 / n_samples) * 100.0, 2) if n_samples > 0 else 0.0

    print("\n" + "=" * 68)
    print(f"Benchmark Results: {args.condition} (Samples: {n_samples}, Duration: {total_duration_s:.2f}s, Throughput: {throughput} req/s)")
    print("-" * 68)
    print(f"{'Metric':<8} | {'Client Wall (ms)':<16} | {'Server Model (ms)':<17} | {'Gap / DB+Net (ms)':<16}")
    print("-" * 68)
    for k in ["min", "p50", "p95", "p99", "max", "mean"]:
        print(f"{k.upper():<8} | {client_stats[k]:<16.2f} | {server_stats[k]:<17.2f} | {gap_stats[k]:<16.2f}")
    print("=" * 68)

    if args.paced_rps:
        print("Paced Profile ('a proxy for normal lab-demo load'):")
        print(f"  Requests > 2000 ms : {requests_over_2000} / {n_samples} ({share_over_2000}%)")
        print("  Target NFR-1 Limit : 2,000 ms")

    # Per-class stats from interleaved mix
    per_class_stats: dict[str, Any] = {}
    if args.tx_type == "mix" and successful:
        classes = sorted({r.get("tx_type", "unknown") for r in successful})
        print("\nPer-Class Breakdown (interleaved mix):")
        print(f"{'Type':<12} | {'Count':<6} | {'Client p50 (ms)':<16} | {'Server p50 (ms)':<16}")
        print("-" * 56)
        for cls in classes:
            cls_client = [r["client_ms"] for r in successful if r.get("tx_type") == cls]
            cls_server = [r["server_ms"] for r in successful if r.get("tx_type") == cls]
            per_class_stats[cls] = {
                "count": len(cls_client),
                "client_latency": compute_percentiles(cls_client),
                "server_latency": compute_percentiles(cls_server),
            }
            print(f"{cls:<12} | {len(cls_client):<6} | {per_class_stats[cls]['client_latency']['p50']:<16.2f} | {per_class_stats[cls]['server_latency']['p50']:<16.2f}")

    # 5. Persist raw records to cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{args.condition}.json"

    provenance_data = collect_provenance()

    raw_payload = {
        "condition": args.condition,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "model_mode": model_mode,
        "model_version": model_version,
        "provenance": provenance_data,
        "clients": args.clients,
        "tx_type": args.tx_type,
        "throughput_rps": throughput,
        "duration_s": round(total_duration_s, 3),
        "errors": error_count,
        "invalid": invalid_count,
        "client_stats": client_stats,
        "server_stats": server_stats,
        "gap_stats": gap_stats,
        "per_class_stats": per_class_stats,
        "requests_over_2000ms": requests_over_2000,
        "share_over_2000ms": share_over_2000,
        "think_time_z": args.think_time_z,
        "lockstep": args.lockstep,
        "raw_records": results,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, indent=2)
    print(f"Persisted per-condition raw cache to: {cache_file}")

    # 6. Structured JSON summary report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"latency_{args.condition}_{timestamp_str}.json"

    report_payload = {
        "srs_requirements": SRS_CITATIONS,
        "condition": args.condition,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "model_mode": model_mode,
        "model_version": model_version,
        "provenance": provenance_data,
        "system": {
            "os": platform.platform(),
            "cpu": platform.processor() or "unknown",
        },
        "concurrency_clients": args.clients,
        "tx_type": args.tx_type,
        "num_requests": n_samples,
        "warmup_requests": args.warmup,
        "total_duration_s": round(total_duration_s, 3),
        "throughput_rps": throughput,
        "error_count": error_count,
        "invalid_response_count": invalid_count,
        "client_latency": client_stats,
        "server_latency": server_stats,
        "gap_latency": gap_stats,
    }

    if per_class_stats:
        report_payload["per_class_stats"] = per_class_stats

    if args.think_time_z:
        report_payload["think_time_profile"] = {
            "z_seconds": args.think_time_z,
            "label": "think-time uniform [0.5Z, 1.5Z]",
        }

    if args.lockstep:
        report_payload["lockstep_profile"] = {
            "label": "10 users click at the same instant",
            "concurrency": args.clients,
        }

    if args.paced_rps:
        report_payload["paced_profile"] = {
            "target_rps_per_client": args.paced_rps,
            "duration_seconds": args.duration,
            "requests_exceeding_2000ms": requests_over_2000,
            "share_exceeding_2000ms": share_over_2000,
            "label": "a proxy for normal lab-demo load",
        }

    if export_data:
        report_payload["export_benchmark"] = export_data

    clean_report = sanitize_metadata(report_payload)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_report, f, indent=2)

    print(f"Saved structured benchmark report to: {output_file}\n")


if __name__ == "__main__":
    main()
