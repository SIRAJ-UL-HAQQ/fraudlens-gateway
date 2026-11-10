#!/usr/bin/env python3
r"""Latency Matrix Orchestration Script for FraudLens Gateway.

Automates the complete multi-condition benchmark matrix across:
1. 3 Rounds of 4 configurations:
   - Trained Direct (http://127.0.0.1:8000)
   - Trained Reverse Proxy (http://127.0.0.1:5173)
   - Stub Direct (http://127.0.0.1:8000)
   - Stub Reverse Proxy (http://127.0.0.1:5173)
2. Think-time profile sweep ($Z \in \{1, 3, 5, 10\}$ direct, $Z=5$ proxy)
3. Paced profile (10 clients @ 1 req/s for 120 s, proxy for normal lab-demo load)
4. Lockstep worst-case profile ("10 users click at the same instant")
5. Isolated export compliance benchmark (5,000 seeded rows in clean stub mode)
6. Resource profiling pass with background docker stats telemetry
7. Per-round drift analysis (Round 1 vs Round 3)
8. Raw cache archival to external path with SHA-256 recording in canonical JSON

Strict operational constraints:
- Per-condition `docker compose down -v`, `up -d --wait` with MODEL_PATH passed via env (no .env edits)
- Mode assertion on /health before every condition
- 60 s thermal/load cooldown between container restarts
- Abort on any HTTP or validation error; flag resumed runs `resumed: true`
- Overall timeout of 90 minutes
- Never commits raw latencies or zip archives to git
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in [str(REPO_ROOT), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_latency import (
    SRS_CITATIONS,
    collect_provenance,
    generate_payload,
    load_credentials_if_missing,
    sanitize_metadata,
)


def run_command(
    cmd: list[str],
    env: dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    """Execute a shell command with merged environment and timeout."""
    merged_env = {**os.environ}
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def switch_compose_environment(
    model_path: str,
    expected_mode: str,
    target_url: str = "http://127.0.0.1:8000",
    max_wait_s: float = 120.0,
) -> None:
    """Tear down existing volumes and bring up containers with specified MODEL_PATH."""
    print("\n[Compose] Bringing down stack: docker compose down -v ...")
    run_command(["docker", "compose", "down", "-v"])

    print(f"[Compose] Bringing up stack with MODEL_PATH='{model_path}' ...")
    cmd_env = {"MODEL_PATH": model_path}
    run_command(["docker", "compose", "up", "-d", "--wait"], env=cmd_env)

    # Healthcheck verification
    print(f"[Compose] Verifying /health on {target_url} (expecting mode='{expected_mode}') ...")
    t0 = time.monotonic()
    last_err = ""
    while time.monotonic() - t0 < max_wait_s:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{target_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    current_mode = data.get("model", {}).get("mode")
                    if current_mode == expected_mode:
                        print(f"[Compose] Health OK: mode='{current_mode}', version='{data.get('model', {}).get('version')}'")
                        return
                    last_err = f"Unexpected mode: {current_mode} != {expected_mode}"
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:100]}"
        except (httpx.HTTPError, OSError) as e:
            last_err = str(e)
        time.sleep(2.0)

    raise RuntimeError(f"Failed to achieve healthy state for mode '{expected_mode}' within {max_wait_s}s. Error: {last_err}")


def authenticate_analyst(base_url: str) -> dict[str, str]:
    """Authenticate analyst and return Authorization header dict."""
    username = os.environ.get("ANALYST_USERNAME", "analyst_1")
    password = os.environ.get("ANALYST_PASSWORD", "")
    if not password:
        raise ValueError("Missing ANALYST_PASSWORD environment variable.")

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(
            f"{base_url}/api/v1/auth/login",
            json={"username": username, "password": password},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Authentication failed on {base_url} ({resp.status_code}): {resp.text}")
        token = resp.json().get("access_token")
        return {"Authorization": f"Bearer {token}"}


def seed_fast_stub_transactions(
    base_url: str,
    headers: dict[str, str],
    target_count: int = 5000,
    concurrency: int = 20,
) -> None:
    """Seed exactly target_count transactions in stub mode via concurrent keep-alive workers."""
    print(f"\n[Seed] Fast-seeding {target_count} transactions into clean stub database...")
    predict_url = f"{base_url}/api/v1/predict"
    reqs_per_worker = target_count // concurrency
    remainder = target_count % concurrency

    def worker(worker_id: int, count: int, offset: int) -> int:
        success_count = 0
        with httpx.Client(timeout=30.0) as client:
            for i in range(count):
                idx = offset + i
                payload = generate_payload(idx, tx_type="mix", seed=42)
                resp = client.post(predict_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    success_count += 1
                else:
                    print(f"Seed worker {worker_id} request {idx} failed: {resp.status_code}", file=sys.stderr)
        return success_count

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        curr_offset = 0
        for w in range(concurrency):
            c = reqs_per_worker + (1 if w < remainder else 0)
            futures.append(executor.submit(worker, w, c, curr_offset))
            curr_offset += c

        total_inserted = sum(f.result() for f in futures)
    t1 = time.perf_counter()
    print(f"[Seed] Successfully seeded {total_inserted}/{target_count} records in {t1 - t0:.2f} s ({total_inserted / (t1 - t0):.1f} req/s).")

    # Verify line count via CSV export
    with httpx.Client(timeout=30.0) as client:
        export_resp = client.get(f"{base_url}/api/v1/export?format=csv", headers=headers)
        if export_resp.status_code != 200:
            raise RuntimeError(f"CSV Export verification failed: {export_resp.status_code}")
        lines = export_resp.text.strip().splitlines()
        actual_rows = len(lines) - 1
        print(f"[Seed] Database record audit: {actual_rows} rows (CSV line count: {len(lines)}).")
        if actual_rows != target_count:
            raise AssertionError(f"Expected {target_count} rows, found {actual_rows}!")


def sample_docker_stats(
    stop_event: threading.Event,
    samples: list[dict[str, Any]],
    interval: float = 1.0,
) -> None:
    """Background sampler thread collecting docker stats metrics."""
    while not stop_event.is_set():
        try:
            res = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                for line in res.stdout.strip().splitlines():
                    if "|" in line:
                        parts = line.split("|")
                        c_name = parts[0].strip()
                        cpu_str = parts[1].replace("%", "").strip()
                        mem_str = parts[2].strip()
                        try:
                            cpu_val = float(cpu_str)
                        except ValueError:
                            cpu_val = 0.0

                        samples.append({
                            "timestamp": time.time(),
                            "container": c_name,
                            "cpu_pct": cpu_val,
                            "mem_usage": mem_str,
                        })
        except (subprocess.SubprocessError, OSError):
            pass
        time.sleep(interval)


def zip_cache_archive(cache_dir: Path, target_zip: Path) -> str:
    """Zip all files in cache_dir to external target_zip and return SHA-256."""
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(cache_dir.glob("*.json")):
            zf.write(file, arcname=file.name)

    sha256 = hashlib.sha256(target_zip.read_bytes()).hexdigest().upper()
    print(f"[Archive] Raw cache files zipped to: {target_zip}")
    print(f"[Archive] Raw cache SHA-256: {sha256}")
    return sha256


def execute_condition_cli(
    condition_name: str,
    target_url: str,
    clients: int,
    num_requests: int,
    warmup: int,
    cache_dir: Path,
    extra_flags: list[str] | None = None,
    operator_note: str = "",
) -> tuple[dict[str, Any], str]:
    """Execute benchmark_latency.py for a condition and return parsed cache data and command string."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "benchmark_latency.py"),
        "--url",
        target_url,
        "--condition",
        condition_name,
        "--clients",
        str(clients),
        "--num-requests",
        str(num_requests),
        "--warmup",
        str(warmup),
        "--cache-dir",
        str(cache_dir),
        "--yes",
    ]
    if operator_note:
        cmd.extend(["--operator-note", operator_note])
    if extra_flags:
        cmd.extend(extra_flags)

    cmd_str = " ".join(cmd)
    print(f"\n[Execute] {cmd_str}")
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print(f"[Execute] Condition '{condition_name}' failed ({res.returncode}):\n{res.stderr}\n{res.stdout}", file=sys.stderr)
        raise RuntimeError(f"Condition '{condition_name}' failed with return code {res.returncode}")

    cache_file = cache_dir / f"{condition_name}.json"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cache output: {cache_file}")

    data = json.loads(cache_file.read_text(encoding="utf-8"))
    return data, cmd_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete FraudLens Gateway latency benchmark matrix."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of full matrix rounds (default: 3)",
    )
    parser.add_argument(
        "--scoring-requests",
        type=int,
        default=500,
        help="Measured requests per scoring concurrency level (default: 500)",
    )
    parser.add_argument(
        "--scoring-warmup",
        type=int,
        default=10,
        help="Warmup requests per scoring condition (default: 10)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=60,
        help="Thermal cooldown in seconds between conditions (default: 60)",
    )
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=5400.0,
        help="Maximum allowable wall time in seconds (default: 5400s / 90min)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing cached condition files if valid",
    )
    parser.add_argument(
        "--operator-note",
        default="",
        help="Free-text operator note included in provenance metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run short dry run (N=10, 1 round, short think times) to verify matrix logic",
    )
    parser.add_argument(
        "--output",
        default="ml/reports/latency_benchmark.json",
        help="Path to canonical benchmark JSON report",
    )
    parser.add_argument(
        "--cache-dir",
        default=".benchmark_cache",
        help="Directory storing raw per-condition latencies",
    )
    parser.add_argument(
        "--raw-zip-path",
        default=None,
        help="External path for raw cache zip archive (default: tempdir/fraudlens_raw_latencies_<ts>.zip)",
    )
    return parser.parse_args()


def main() -> None:
    load_credentials_if_missing()
    args = parse_args()

    overall_start = time.monotonic()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("\n*** RUNNING IN DRY-RUN MODE: Scaled down request counts and rounds ***\n")
        rounds_count = 1
        req_count = 10
        warmup_count = 2
        cooldown_s = 5
        paced_duration = 5
        seed_count = 200
    else:
        rounds_count = args.rounds
        req_count = args.scoring_requests
        warmup_count = args.scoring_warmup
        cooldown_s = args.cooldown
        paced_duration = 120
        seed_count = 5000

    direct_url = "http://127.0.0.1:8000"
    proxy_url = "http://127.0.0.1:5173"

    executed_commands: list[str] = []
    matrix_rounds_data: dict[str, Any] = {}
    think_sweep_data: dict[str, Any] = {}
    paced_profile_data: dict[str, Any] = {}
    export_benchmark_data: dict[str, Any] = {}
    profiling_stats: dict[str, Any] = {}
    any_resumed = False

    def check_timeout() -> None:
        elapsed = time.monotonic() - overall_start
        if elapsed > args.max_wall_time:
            raise TimeoutError(f"Benchmark aborted: exceeded maximum wall time of {args.max_wall_time} s ({elapsed:.1f} s elapsed).")

    print("=" * 72)
    print("FraudLens Gateway — Latency & Throughput Matrix Orchestrator")
    print(f"Rounds: {rounds_count} | Requests per level: {req_count} | Cooldown: {cooldown_s} s")
    print(f"Max Wall Time: {args.max_wall_time} s (90 min) | Resume: {args.resume}")
    print(f"Operator Note: {args.operator_note or 'None'}")
    print("=" * 72)

    # ---------------------------------------------------------
    # PART 1: 3-ROUND SCORING MATRIX
    # ---------------------------------------------------------
    for r in range(1, rounds_count + 1):
        round_key = f"round_{r}"
        matrix_rounds_data[round_key] = {}
        print(f"\n>>>>>>>>>>>>>>> STARTING ROUND {r}/{rounds_count} <<<<<<<<<<<<<<<")

        # Configurations order per round:
        # 1. Trained Direct
        # 2. Trained Proxy
        # 3. Stub Direct
        # 4. Stub Proxy
        configs = [
            ("trained_direct", "ml/models/class_weight", "trained", direct_url),
            ("trained_proxy", "ml/models/class_weight", "trained", proxy_url),
            ("stub_direct", "", "stub", direct_url),
            ("stub_proxy", "", "stub", proxy_url),
        ]

        for cfg_name, model_path, expected_mode, target_url in configs:
            check_timeout()
            cond_prefix = f"r{r}_{cfg_name}"
            print(f"\n--- Round {r} Condition: {cfg_name} (Mode: {expected_mode}, URL: {target_url}) ---")

            # Check if all subconditions are already cached
            c1_file = cache_dir / f"{cond_prefix}_c1.json"
            c4_file = cache_dir / f"{cond_prefix}_c4.json"
            c10_file = cache_dir / f"{cond_prefix}_c10.json"
            c_lock_file = cache_dir / f"{cond_prefix}_lockstep.json"

            if (
                args.resume
                and c1_file.exists()
                and c4_file.exists()
                and c10_file.exists()
                and c_lock_file.exists()
            ):
                print(f"[Resume] Reusing cached results for {cond_prefix} ...")
                any_resumed = True
                d1 = json.loads(c1_file.read_text(encoding="utf-8"))
                d4 = json.loads(c4_file.read_text(encoding="utf-8"))
                d10 = json.loads(c10_file.read_text(encoding="utf-8"))
                d_lock = json.loads(c_lock_file.read_text(encoding="utf-8"))
            else:
                switch_compose_environment(model_path, expected_mode, target_url=direct_url)
                print(f"[Cooldown] Thermal stabilization cooldown for {cooldown_s} s ...")
                time.sleep(cooldown_s)

                d1, cmd1 = execute_condition_cli(
                    f"{cond_prefix}_c1", target_url, 1, req_count, warmup_count, cache_dir, operator_note=args.operator_note
                )
                executed_commands.append(cmd1)

                d4, cmd4 = execute_condition_cli(
                    f"{cond_prefix}_c4", target_url, 4, req_count, warmup_count, cache_dir, operator_note=args.operator_note
                )
                executed_commands.append(cmd4)

                d10, cmd10 = execute_condition_cli(
                    f"{cond_prefix}_c10", target_url, 10, req_count, warmup_count, cache_dir, operator_note=args.operator_note
                )
                executed_commands.append(cmd10)

                d_lock, cmd_lock = execute_condition_cli(
                    f"{cond_prefix}_lockstep",
                    target_url,
                    10,
                    10 if args.dry_run else 50,
                    0,
                    cache_dir,
                    extra_flags=["--lockstep"],
                    operator_note=args.operator_note,
                )
                executed_commands.append(cmd_lock)

            matrix_rounds_data[round_key][cfg_name] = {
                "c1": d1["client_stats"],
                "c4": d4["client_stats"],
                "c10": d10["client_stats"],
                "server_c1": d1["server_stats"],
                "server_c4": d4["server_stats"],
                "server_c10": d10["server_stats"],
                "gap_c1": d1["gap_stats"],
                "gap_c4": d4["gap_stats"],
                "gap_c10": d10["gap_stats"],
                "throughput_c1": d1["throughput_rps"],
                "throughput_c4": d4["throughput_rps"],
                "throughput_c10": d10["throughput_rps"],
                "lockstep": d_lock["client_stats"],
            }

            # In Round 1, execute think-time sweep and paced profile during trained conditions
            if r == 1 and cfg_name == "trained_direct":
                check_timeout()
                # Paced Profile (10 clients @ 1 req/s)
                paced_file = cache_dir / "r1_trained_direct_paced.json"
                if args.resume and paced_file.exists():
                    print("[Resume] Reusing cached paced profile ...")
                    any_resumed = True
                    d_paced = json.loads(paced_file.read_text(encoding="utf-8"))
                else:
                    d_paced, cmd_paced = execute_condition_cli(
                        "r1_trained_direct_paced",
                        direct_url,
                        10,
                        10,
                        0,
                        cache_dir,
                        extra_flags=["--paced-rps", "1.0", "--duration", str(paced_duration)],
                        operator_note=args.operator_note,
                    )
                    executed_commands.append(cmd_paced)
                paced_profile_data = {
                    "target_rps_per_client": 1.0,
                    "concurrency_clients": 10,
                    "duration_seconds": paced_duration,
                    "requests_exceeding_2000ms": d_paced.get("requests_over_2000ms", 0),
                    "share_exceeding_2000ms": d_paced.get("share_over_2000ms", 0.0),
                    "client_stats": d_paced["client_stats"],
                    "server_stats": d_paced["server_stats"],
                    "label": "a proxy for normal lab-demo load",
                }

                # Think-time sweep Z in {1, 3, 5, 10}
                z_values = [1.0, 3.0, 5.0, 10.0]
                for z in z_values:
                    check_timeout()
                    z_name = f"r1_trained_direct_z{int(z)}"
                    z_file = cache_dir / f"{z_name}.json"
                    if args.resume and z_file.exists():
                        print(f"[Resume] Reusing cached think-time Z={z} ...")
                        any_resumed = True
                        dz = json.loads(z_file.read_text(encoding="utf-8"))
                    else:
                        dz, cmdz = execute_condition_cli(
                            z_name,
                            direct_url,
                            10,
                            20 if args.dry_run else req_count,
                            warmup_count,
                            cache_dir,
                            extra_flags=["--think-time-z", str(z)],
                            operator_note=args.operator_note,
                        )
                        executed_commands.append(cmdz)
                    think_sweep_data[f"z_{int(z)}_direct"] = {
                        "z_seconds": z,
                        "client_stats": dz["client_stats"],
                        "server_stats": dz["server_stats"],
                        "gap_stats": dz["gap_stats"],
                        "throughput_rps": dz["throughput_rps"],
                    }

            if r == 1 and cfg_name == "trained_proxy":
                check_timeout()
                # Think-time proxy run (Z=5)
                z_proxy_name = "r1_trained_proxy_z5"
                z_proxy_file = cache_dir / f"{z_proxy_name}.json"
                if args.resume and z_proxy_file.exists():
                    print("[Resume] Reusing cached think-time proxy Z=5 ...")
                    any_resumed = True
                    dz_proxy = json.loads(z_proxy_file.read_text(encoding="utf-8"))
                else:
                    dz_proxy, cmdz_proxy = execute_condition_cli(
                        z_proxy_name,
                        proxy_url,
                        10,
                        20 if args.dry_run else req_count,
                        warmup_count,
                        cache_dir,
                        extra_flags=["--think-time-z", "5.0"],
                        operator_note=args.operator_note,
                    )
                    executed_commands.append(cmdz_proxy)
                think_sweep_data["z_5_proxy"] = {
                    "z_seconds": 5.0,
                    "client_stats": dz_proxy["client_stats"],
                    "server_stats": dz_proxy["server_stats"],
                    "gap_stats": dz_proxy["gap_stats"],
                    "throughput_rps": dz_proxy["throughput_rps"],
                }

    # ---------------------------------------------------------
    # PART 2: COMPLIANCE EXPORT BENCHMARK (NFR-5)
    # ---------------------------------------------------------
    check_timeout()
    print("\n>>>>>>>>>>>>>>> STARTING COMPLIANCE EXPORT BENCHMARK (NFR-5) <<<<<<<<<<<<<<<")
    export_cache_file = cache_dir / "export_compliance_audit.json"
    if args.resume and export_cache_file.exists():
        print("[Resume] Reusing cached export compliance benchmark ...")
        any_resumed = True
        export_benchmark_data = json.loads(export_cache_file.read_text(encoding="utf-8"))
    else:
        # Export condition has its own clean stack in stub mode
        switch_compose_environment("", "stub", target_url=direct_url)
        headers = authenticate_analyst(direct_url)
        seed_fast_stub_transactions(direct_url, headers, target_count=seed_count, concurrency=20)

        # Run direct export benchmark
        d_exp_direct, cmd_exp_dir = execute_condition_cli(
            "export_direct_condition",
            direct_url,
            1,
            0,
            0,
            cache_dir,
            extra_flags=["--export-test", "--export-runs", "3"],
            operator_note=args.operator_note,
        )
        executed_commands.append(cmd_exp_dir)

        # Run proxy export benchmark
        d_exp_proxy, cmd_exp_proxy = execute_condition_cli(
            "export_proxy_condition",
            proxy_url,
            1,
            0,
            0,
            cache_dir,
            extra_flags=["--export-test", "--export-runs", "3"],
            operator_note=args.operator_note,
        )
        executed_commands.append(cmd_exp_proxy)

        export_benchmark_data = {
            "target_nfr": "NFR-5",
            "threshold_s": 5.0,
            "seeded_records": seed_count,
            "export_scope": "all_matching_rows_not_flagged_only",
            "direct": d_exp_direct.get("export_benchmark", {}),
            "proxy": d_exp_proxy.get("export_benchmark", {}),
        }
        export_cache_file.write_text(json.dumps(export_benchmark_data, indent=2), encoding="utf-8")

    # ---------------------------------------------------------
    # PART 3: RESOURCE PROFILING PASS (DOCKER STATS)
    # ---------------------------------------------------------
    check_timeout()
    print("\n>>>>>>>>>>>>>>> STARTING RESOURCE PROFILING PASS <<<<<<<<<<<<<<<")
    prof_file = cache_dir / "resource_profiling_pass.json"
    if args.resume and prof_file.exists():
        print("[Resume] Reusing cached resource profiling pass ...")
        any_resumed = True
        profiling_stats = json.loads(prof_file.read_text(encoding="utf-8"))
    else:
        switch_compose_environment("ml/models/class_weight", "trained", target_url=direct_url)
        time.sleep(5)
        stop_sampler = threading.Event()
        stats_samples: list[dict[str, Any]] = []
        sampler_thread = threading.Thread(
            target=sample_docker_stats, args=(stop_sampler, stats_samples, 1.0)
        )
        sampler_thread.daemon = True
        sampler_thread.start()

        # Run 4-client scoring load for profiling
        d_prof, cmd_prof = execute_condition_cli(
            "resource_profile_load",
            direct_url,
            4,
            50 if args.dry_run else 300,
            5,
            cache_dir,
            operator_note=args.operator_note,
        )
        executed_commands.append(cmd_prof)

        stop_sampler.set()
        sampler_thread.join(timeout=5.0)

        # Analyze peak CPU and memory
        peak_cpu_backend = max([s["cpu_pct"] for s in stats_samples if "backend" in s["container"]] or [0.0])
        peak_cpu_postgres = max([s["cpu_pct"] for s in stats_samples if "postgres" in s["container"]] or [0.0])
        last_mem_backend = [s["mem_usage"] for s in stats_samples if "backend" in s["container"]]
        last_mem_postgres = [s["mem_usage"] for s in stats_samples if "postgres" in s["container"]]

        profiling_stats = {
            "duration_s": d_prof.get("duration_s", 0.0),
            "throughput_rps": d_prof.get("throughput_rps", 0.0),
            "peak_backend_cpu_pct": peak_cpu_backend,
            "peak_postgres_cpu_pct": peak_cpu_postgres,
            "sample_backend_memory": last_mem_backend[-1] if last_mem_backend else "unknown",
            "sample_postgres_memory": last_mem_postgres[-1] if last_mem_postgres else "unknown",
        }
        prof_file.write_text(json.dumps(profiling_stats, indent=2), encoding="utf-8")

    # ---------------------------------------------------------
    # PART 4: ROUND DRIFT ANALYSIS (ROUND 1 vs ROUND 3)
    # ---------------------------------------------------------
    r1_trained = matrix_rounds_data.get("round_1", {}).get("trained_direct", {})
    r3_trained = matrix_rounds_data.get(f"round_{rounds_count}", {}).get("trained_direct", {})

    r1_p50 = r1_trained.get("c1", {}).get("p50", 0.0)
    r3_p50 = r3_trained.get("c1", {}).get("p50", 0.0)
    r1_rps = r1_trained.get("throughput_c1", 0.0)
    r3_rps = r3_trained.get("throughput_c1", 0.0)

    p50_drift_pct = round(((r3_p50 - r1_p50) / r1_p50) * 100.0, 2) if r1_p50 > 0 else 0.0
    rps_drift_pct = round(((r3_rps - r1_rps) / r1_rps) * 100.0, 2) if r1_rps > 0 else 0.0

    round_drift_data = {
        "round_1_p50_ms": r1_p50,
        f"round_{rounds_count}_p50_ms": r3_p50,
        "p50_drift_pct": p50_drift_pct,
        "round_1_throughput_rps": r1_rps,
        f"round_{rounds_count}_throughput_rps": r3_rps,
        "throughput_drift_pct": rps_drift_pct,
    }

    # ---------------------------------------------------------
    # PART 5: ARCHIVE RAW CACHES OUTSIDE REPO & COMPUTE SHA-256
    # ---------------------------------------------------------
    ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if args.raw_zip_path:
        zip_target_path = Path(args.raw_zip_path)
    else:
        zip_target_path = Path(tempfile.gettempdir()) / f"fraudlens_raw_latencies_{ts_str}.zip"

    raw_sha256 = zip_cache_archive(cache_dir, zip_target_path)

    # ---------------------------------------------------------
    # PART 6: ASSEMBLE CANONICAL LATENCY BENCHMARK JSON
    # ---------------------------------------------------------
    provenance_meta = collect_provenance(operator_note=args.operator_note)
    provenance_meta["raw_data_sha256"] = raw_sha256
    provenance_meta["resumed"] = any_resumed

    canonical_report = {
        "srs_requirements": SRS_CITATIONS,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": provenance_meta["git_commit"],
        "provenance": provenance_meta,
        "rounds": matrix_rounds_data,
        "round_drift": round_drift_data,
        "think_time_sweep": think_sweep_data,
        "paced_profile": paced_profile_data,
        "lockstep_profile": {
            "label": "10 users click at the same instant",
            "concurrency": 10,
            "round_1_direct_max_ms": matrix_rounds_data.get("round_1", {}).get("trained_direct", {}).get("lockstep", {}).get("max", 0.0),
        },
        "export_benchmark": export_benchmark_data,
        "resource_profiling": profiling_stats,
        "script_arguments": {
            "orchestrator": " ".join(sys.argv),
            "commands": executed_commands,
        },
    }

    clean_report = sanitize_metadata(canonical_report)
    out_file = REPO_ROOT / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(clean_report, indent=2), encoding="utf-8")
    print(f"\n[Canonical Report] Committed report generated at: {out_file}")

    total_wall_s = time.monotonic() - overall_start
    print(f"\nTotal matrix execution wall time: {total_wall_s / 60.0:.2f} minutes.")
    print("Matrix run completed successfully.")


if __name__ == "__main__":
    main()
