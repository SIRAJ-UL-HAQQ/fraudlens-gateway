#!/usr/bin/env python3
"""Latency Benchmark Script for FraudLens Gateway.

Executes N scoring requests against /predict, measures client round-trip and server latency_ms,
and outputs statistical percentiles (min, max, mean, p50, p95, p99) to JSON report.
Requires --yes to execute as it persists transactions to the target database.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark latency of FraudLens Gateway /predict endpoint."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("GATEWAY_URL", "http://localhost:8000"),
        help="Base URL of FraudLens Gateway (default: GATEWAY_URL env or http://localhost:8000)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("ANALYST_USERNAME"),
        help="Analyst username (default: ANALYST_USERNAME env)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of benchmark scoring requests (default: 1000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of initial warm-up requests excluded from metrics (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/benchmarks",
        help="Directory to store benchmark JSON results (default: reports/benchmarks)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm execution. Required to proceed as benchmark records are persisted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")

    # Security: Password must come from environment only
    password = os.environ.get("ANALYST_PASSWORD")
    if not password:
        print(
            "ERROR: Missing required environment variable ANALYST_PASSWORD.\n"
            "Passwords must never be passed as CLI arguments.",
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

    print("=" * 65)
    print("FraudLens Gateway — Latency Benchmark")
    print(f"Target Gateway URL : {base_url}")
    print(f"Requests Count (N) : {args.num_requests}")
    print(f"Warm-up Requests   : {args.warmup}")
    print("=" * 65)
    print("WARNING: This benchmark persists transactions to the target database.")
    print("Ensure this command is directed at a throwaway or dedicated test instance.")

    if not args.yes:
        print(
            "\nAborted: --yes was not provided. Run with --yes to confirm execution.\n"
            "Example: python scripts/benchmark_latency.py --yes",
            file=sys.stderr,
        )
        sys.exit(1)

    with httpx.Client(timeout=30.0) as client:
        # Check /health and inspect model info
        health_resp = client.get(f"{base_url}/health")
        if health_resp.status_code != 200:
            print(f"Health check failed ({health_resp.status_code}): {health_resp.text}", file=sys.stderr)
            sys.exit(1)

        health_data = health_resp.json()
        model_info = health_data.get("model", {})
        model_mode = model_info.get("mode", "unknown")
        model_version = model_info.get("version", "unknown")

        if model_mode == "stub":
            print(
                "\n" + "!" * 65 + "\n"
                "WARNING: Gateway is running in STUB model mode.\n"
                "Latency measurements reflect deterministic heuristic execution, not trained model inference.\n"
                + "!" * 65 + "\n"
            )

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
        predict_url = f"{base_url}/api/v1/predict"

        sample_tx = {
            "step": 12,
            "type": "TRANSFER",
            "amount": 75000.0,
            "name_orig": "C_BENCH_01",
            "old_balance_orig": 100000.0,
            "new_balance_orig": 25000.0,
            "name_dest": "C_BENCH_02",
            "old_balance_dest": 1000.0,
            "new_balance_dest": 76000.0,
        }

        # 1. Warm-up Phase
        if args.warmup > 0:
            print(f"Executing {args.warmup} warm-up requests...")
            for _ in range(args.warmup):
                client.post(predict_url, json=sample_tx, headers=headers)

        # 2. Benchmark Phase
        client_latencies_ms: list[float] = []
        server_latencies_ms: list[float] = []

        print(f"Benchmarking {args.num_requests} requests...")
        for i in range(1, args.num_requests + 1):
            t0 = time.perf_counter()
            resp = client.post(predict_url, json=sample_tx, headers=headers)
            t1 = time.perf_counter()

            if resp.status_code == 200:
                client_rt = (t1 - t0) * 1000.0
                client_latencies_ms.append(client_rt)
                data = resp.json()
                server_lat = float(data.get("latency_ms", 0.0))
                server_latencies_ms.append(server_lat)
            else:
                print(f"Request {i} failed ({resp.status_code}): {resp.text}", file=sys.stderr)

    client_stats = compute_percentiles(client_latencies_ms)
    server_stats = compute_percentiles(server_latencies_ms)

    # Display results
    print("\n" + "=" * 65)
    print("Benchmark Results Summary (milliseconds):")
    print("-" * 65)
    print(f"{'Metric':<10} | {'Client Round-Trip':<20} | {'Server latency_ms':<20}")
    print("-" * 65)
    for k in ["min", "max", "mean", "p50", "p95", "p99"]:
        print(f"{k.upper():<10} | {client_stats[k]:<20.2f} | {server_stats[k]:<20.2f}")
    print("=" * 65)

    # Prepare structured JSON report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"latency_{timestamp}.json"

    result_payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "model_mode": model_mode,
        "model_version": model_version,
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "n_requests": len(client_latencies_ms),
        "warmup_requests": args.warmup,
        "client_latency": client_stats,
        "server_latency": server_stats,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print(f"\nSaved structured benchmark report to: {output_file}")


if __name__ == "__main__":
    main()
