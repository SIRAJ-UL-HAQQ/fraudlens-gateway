# FraudLens Gateway — Latency & Throughput Benchmark Report

### SRS Requirements Referenced
- **NFR-1**: 2 s per scoring request under normal lab-demo load (`docs/FraudLens_Gateway_SRS.tex`).
- **NFR-4**: $\ge$ 10 concurrent dashboard users (`docs/FraudLens_Gateway_SRS.tex`).
- **NFR-5**: Export of up to 5,000 transactions within 5 s (`docs/FraudLens_Gateway_SRS.tex`).
- **Endpoint Reference Note**: The SRS specifies `/api/v1/score`; the deployed endpoint is `/api/v1/predict` (see `docs/srs-deviations.md`).
- **Export Behavior Note**: `/api/v1/export` exports all transactions matching query filters (default: all records up to 5,000 rows), not flagged-only transactions.

---

## 1. Executive Summary & Observations

- **Scoring Latency (NFR-1)**: All tested scoring requests completed well within the 2,000 ms ceiling under both sequential and concurrent profiles.
- **Dashboard Concurrency (NFR-4)**: Tested under 1, 4, and 10 concurrent clients, as well as a paced profile (10 clients @ 1 req/s) representing normal lab-demo load.
- **Export Latency (NFR-5)**: Export of 5,000 database transactions tested in CSV and PDF formats across direct (:8000) and reverse-proxy (:5173) routes.

---

## 2. Before vs After Optimization (Tree Explainer Cache)

Tree explainers were previously constructed per request (6 TreeExplainers per call). After pre-caching them at startup, server latency dropped from ~245 ms to ~28 ms ($p_{50}$).

| Evaluation State | Client Wall $p_{50}$ (ms) | Server Model $p_{50}$ (ms) | Throughput (req/s) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Before Fix** (Uncached TreeExplainer) | 268.12 | 244.81 | 3.30 | 6 TreeExplainers rebuilt per request |
| **After Fix** (Cached TreeExplainer) | 39.13 | 26.75 | 23.79 | TreeExplainers cached at model bundle load |

---

## 3. Concurrency & Latency Benchmark Matrix

**Summary Table Definition:** The primary matrix below reports the **Round 1 baseline** across concurrency profiles and routes under initial steady-state execution. Section 4 provides the **multi-round stability analysis** (cross-round min, median, and max across all 3 rounds) as well as the Round 1 vs Round 3 drift table across all four configurations.

Dual latency measurements: **Client Wall Time** captures full end-to-end round trip (including network, HTTP keep-alive, JSON serialization, and PostgreSQL transaction commit); **Server Model Time** captures backend inference + explanation.

| Profile / Condition | Clients | Client Wall $p_{50}$ (ms) | Client Wall $p_{95}$ (ms) | Client Max (ms) | Server $p_{50}$ (ms) | Gap $p_{50}$ (ms) | Throughput (req/s) | Share > 2,000 ms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Sequential (Trained, Direct) | 1 | 39.13 | 53.87 | 109.66 | 26.75 | 12.39 | 23.79 | 0.0% |
| Stress 4-Client (Trained, Direct) | 4 | 151.47 | 212.26 | 285.60 | 86.81 | 64.04 | 24.54 | 0.0% |
| Stress 10-Client (Trained, Direct) | 10 | 395.60 | 498.49 | 587.66 | 140.27 | 251.88 | 23.02 | 0.0% |
| Sequential (Trained, Proxy) | 1 | 45.24 | 62.27 | 134.70 | 30.29 | 14.97 | 20.98 | 0.0% |
| Stress 4-Client (Trained, Proxy) | 4 | 164.84 | 212.29 | 295.80 | 87.76 | 76.46 | 22.87 | 0.0% |
| Stress 10-Client (Trained, Proxy) | 10 | 429.85 | 567.50 | 688.26 | 144.58 | 280.26 | 21.15 | 0.0% |
| Sequential (Stub, Direct) | 1 | 14.88 | 19.80 | 66.09 | 1.09 | 13.74 | 60.51 | 0.0% |
| Stress 4-Client (Stub, Direct) | 4 | 50.05 | 73.60 | 130.72 | 1.75 | 48.09 | 68.94 | 0.0% |
| Stress 10-Client (Stub, Direct) | 10 | 161.53 | 219.35 | 278.65 | 2.36 | 158.34 | 49.77 | 0.0% |
| Sequential (Stub, Proxy) | 1 | 15.49 | 19.30 | 79.21 | 1.10 | 14.35 | 58.39 | 0.0% |
| Stress 4-Client (Stub, Proxy) | 4 | 58.41 | 90.20 | 134.55 | 2.04 | 56.29 | 57.60 | 0.0% |
| Stress 10-Client (Stub, Proxy) | 10 | 167.48 | 226.36 | 269.44 | 2.52 | 164.93 | 48.78 | 0.0% |
| Paced (10 clients @ 1 req/s, proxy for lab load) | 10 | 53.94 | 96.75 | 221.35 | 36.73 | 17.21 | 10.00 | 0.0% |
| Think-Time (Z=1.0s, Direct) | 10 | 53.99 | 135.08 | 319.62 | 36.88 | 17.66 | 8.57 | 0.0% |
| Think-Time (Z=3.0s, Direct) | 10 | 49.93 | 94.18 | 155.88 | 32.33 | 17.77 | 3.05 | 0.0% |
| Think-Time (Z=5.0s, Direct) | 10 | 48.64 | 76.94 | 540.29 | 30.39 | 18.11 | 1.85 | 0.0% |
| Think-Time (Z=10.0s, Direct) | 10 | 47.76 | 74.54 | 701.75 | 27.78 | 19.61 | 0.93 | 0.0% |
| Think-Time (Z=5.0s, Proxy) | 10 | 50.24 | 80.50 | 1286.96 | 30.67 | 19.23 | 1.85 | 0.0% |
| Lockstep ('10 users click at same instant') | 10 | 385.72 | 438.85 | 448.09 | 233.52 | 134.81 | 13.47 | 0.0% |

---

## 4. Stability & Round Drift Analysis (Round 1 vs Round 3)

Compares sequential single-client baseline ($c=1$) across cold stack initialization (Round 1) and warm repeat execution (Round 3) across all four configurations.

| Configuration | Route | Model Mode | Round 1 $p_{50}$ | Round 3 $p_{50}$ | $p_{50}$ Drift (%) | Round 1 Throughput | Round 3 Throughput | Throughput Drift (%) | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| trained_direct | direct | trained | 39.13 ms | 42.13 ms | +7.67% | 23.79 req/s | 22.38 req/s | -5.93% | Consistent across cold/warm restarts |
| trained_proxy | proxy | trained | 45.24 ms | 43.34 ms | -4.20% | 20.98 req/s | 21.49 req/s | +2.43% | Reverse proxy overhead steady |
| stub_direct | direct | stub | 14.88 ms | 14.24 ms | -4.30% | 60.51 req/s | 64.41 req/s | +6.45% | Minimal drift under fast stub |
| stub_proxy | proxy | stub | 15.49 ms | 14.78 ms | -4.58% | 58.39 req/s | 61.14 req/s | +4.71% | Consistent proxy forwarding under stub |

### Multi-Round Cross-Run Statistics (Min / Median / Max Across 3 Rounds)

Summarizes client $p_{50}$, client $p_{95}$, server $p_{50}$, and throughput aggregated across the 3 independent stack life cycles for all four configurations.

| Configuration | Profile | Clients | Median Client $p_{50}$ [Min, Max] | Median Client $p_{95}$ [Min, Max] | Median Server $p_{50}$ [Min, Max] | Median Throughput [Min, Max] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| trained_direct | Sequential (1 Client) | 1 | 41.66 ms [39.13, 42.13] | 59.24 ms [53.87, 65.42] | 27.61 ms [26.75, 27.82] | 22.38 req/s [22.34, 23.79] |
| trained_direct | Stress 4-Client | 4 | 151.47 ms [150.46, 155.87] | 209.48 ms [196.01, 212.26] | 84.76 ms [84.09, 86.81] | 24.54 req/s [24.06, 24.85] |
| trained_direct | Stress 10-Client | 10 | 412.39 ms [395.60, 423.31] | 530.47 ms [498.49, 566.71] | 147.62 ms [140.27, 153.44] | 22.15 req/s [21.51, 23.02] |
| trained_direct | Lockstep (10 Clients) | 10 | 385.72 ms [383.04, 426.53] | 438.85 ms [423.46, 451.89] | 237.33 ms [233.52, 270.14] | 13.47 req/s [12.32, 13.73] |
| trained_proxy | Sequential (1 Client) | 1 | 43.34 ms [42.60, 45.24] | 61.95 ms [56.62, 62.27] | 28.51 ms [28.01, 30.29] | 21.49 req/s [20.98, 22.41] |
| trained_proxy | Stress 4-Client | 4 | 154.81 ms [152.58, 164.84] | 212.29 ms [193.86, 219.34] | 83.20 ms [81.39, 87.76] | 24.15 req/s [22.87, 24.49] |
| trained_proxy | Stress 10-Client | 10 | 416.74 ms [414.87, 429.85] | 541.72 ms [535.65, 567.50] | 144.90 ms [144.58, 146.64] | 21.90 req/s [21.15, 22.09] |
| trained_proxy | Lockstep (10 Clients) | 10 | 403.56 ms [395.52, 411.96] | 451.64 ms [439.99, 662.84] | 252.82 ms [249.77, 255.41] | 12.56 req/s [11.78, 12.60] |
| stub_direct | Sequential (1 Client) | 1 | 14.24 ms [13.69, 14.88] | 17.70 ms [16.64, 19.80] | 1.00 ms [0.98, 1.09] | 64.41 req/s [60.51, 66.97] |
| stub_direct | Stress 4-Client | 4 | 50.03 ms [43.04, 50.05] | 71.12 ms [70.71, 73.60] | 1.75 ms [1.43, 1.81] | 69.14 req/s [68.94, 76.91] |
| stub_direct | Stress 10-Client | 10 | 155.83 ms [152.84, 161.53] | 219.35 ms [207.16, 219.43] | 2.38 ms [2.36, 2.48] | 52.55 req/s [49.77, 53.04] |
| stub_direct | Lockstep (10 Clients) | 10 | 142.70 ms [131.47, 150.59] | 170.47 ms [156.50, 220.82] | 5.72 ms [2.23, 7.31] | 19.05 req/s [18.01, 21.07] |
| stub_proxy | Sequential (1 Client) | 1 | 15.49 ms [14.78, 15.64] | 19.30 ms [19.20, 19.89] | 1.07 ms [1.03, 1.10] | 58.93 req/s [58.39, 61.14] |
| stub_proxy | Stress 4-Client | 4 | 51.34 ms [46.93, 58.41] | 76.31 ms [72.35, 90.20] | 1.75 ms [1.70, 2.04] | 67.58 req/s [57.60, 70.43] |
| stub_proxy | Stress 10-Client | 10 | 154.02 ms [153.00, 167.48] | 224.00 ms [220.25, 226.36] | 2.32 ms [2.25, 2.52] | 51.58 req/s [48.78, 51.95] |
| stub_proxy | Lockstep (10 Clients) | 10 | 150.16 ms [143.19, 156.70] | 176.07 ms [168.59, 180.91] | 2.96 ms [2.71, 3.80] | 20.58 req/s [18.43, 20.94] |

---

## 5. Compliance Export Benchmark (NFR-5)

NFR-5 specifies export of up to 5,000 transactions within 5 s. Endpoint: `GET /api/v1/export`. Note: `/api/v1/export` exports all rows matching query filters up to 5,000 rows (not flagged-only). Observations recorded across 3 runs per format and route.

| Format | Route | Target Ceiling | Run 1 Duration | Run 2 Duration | Run 3 Duration | Rows / Pages | Content Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CSV | Direct (:8000) | 5.0 s | 5.00 s | 4.60 s | 4.52 s | 5,000 rows (5,001 lines) | ~789 KB |
| CSV | Proxy (:5173) | 5.0 s | 5.13 s | 6.57 s | 6.48 s | 5,000 rows (5,001 lines) | ~789 KB |
| PDF | Direct (:8000) | 5.0 s | 8.06 s | 10.41 s | 8.81 s | 129 pages | ~427 KB |
| PDF | Proxy (:5173) | 5.0 s | 9.69 s | 9.83 s | 9.89 s | 129 pages | ~427 KB |

Observation: CSV export duration for 5,000 records averaged ~4.7 s on the direct route and ~6.1 s through the reverse proxy (observed difference of 0.6–2.0 s between proxy and direct). PDF export duration averaged ~9.1 s on direct and ~9.8 s through the reverse proxy (observed difference of 0.0–1.6 s between proxy and direct), with exactly 129 formatted pages produced (32 records on the title page, 39 records across each of the subsequent 128 pages). Export timing exhibits session-to-session variance compared to the earlier clean pilot (3.3–3.6 s CSV, 6.1–6.6 s PDF). As an operational limitation, the 5,000 database records were seeded in stub mode.

---

## 6. Resource Profiling (Peak CPU & Memory)

Captured via 1.0 Hz `docker stats` telemetry under concurrent scoring load.

| Service Container | Peak CPU (%) | Sample Working Set RSS Memory |
| :--- | :--- | :--- |
| `fraudlens-gateway-backend-1` | 271.5% | 208MiB / 7.695GiB |
| `fraudlens-gateway-postgres-1` | 7.6% | 41.34MiB / 7.695GiB |

---

## 7. Provenance & Hardware Environment

| Attribute | Value | Description |
| :--- | :--- | :--- |
| Git Commit | `3bc8c22` | Target commit of gateway codebase |
| Backend Image Built Commit | `ae3dae2` | Commit hash from which backend container was built |
| Backend Image Created | `2026-11-01T20:03:56Z` | Container image creation timestamp from docker inspect |
| Model Manifest SHA-256 | `3E04C08C99389688...` | Integrity check of ML bundle (class_weight) |
| Raw Cache Archive SHA-256 | `0F55E985BE48E23A...` | SHA-256 checksum of raw latencies zip (stored outside repository) |
| OMP_NUM_THREADS | `unset` | OpenMP thread parallelism setting |
| Uvicorn Workers | `1` | Single ASGI worker process count |
| Host Environment | `Windows-11-10.0.26300-SP0` (8 logical CPUs) | Benchmark test runner host environment |
| Operator Notes & Power | AC connected (plugged in), Windows Power Scheme: Balanced | Detected AC power state, active Windows scheme, and operator note |
| HTTP Client Transport | HTTP keep-alive (`httpx.Client`) | Connection reuse enabled |
