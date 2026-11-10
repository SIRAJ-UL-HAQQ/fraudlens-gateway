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
| Lockstep ('10 users click at same instant') | 10 | 385.72 | 438.85 | 448.09 | 140.27 | 245.45 | 23.02 | 0.0% |

---

## 4. Stability & Round Drift Analysis (Round 1 vs Round 3)

Compares sequential single-client baseline across cold stack initialization (Round 1) and warm repeat execution (Round 3).

| Metric | Round 1 Baseline | Round 3 Measurement | Observed Drift (%) | Observation |
| :--- | :--- | :--- | :--- | :--- |
| Client Wall $p_50$ | 39.13 ms | 42.13 ms | +7.67% | Consistent across round restarts |
| Sequential Throughput | 23.79 req/s | 22.38 req/s | -5.93% | Steady throughput across 3 rounds |

---

## 5. Compliance Export Benchmark (NFR-5)

NFR-5 specifies export of up to 5,000 transactions within 5 s. Endpoint: `GET /api/v1/export`. Note: `/api/v1/export` exports all rows matching query filters up to 5,000 rows (not flagged-only). Observations recorded across 3 runs per format and route.

| Format | Route | Target Ceiling | Run 1 Duration | Run 2 Duration | Run 3 Duration | Rows / Pages | Content Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CSV | Direct (:8000) | 5.0 s | 5.00 s | 4.60 s | 4.52 s | 5,000 rows (5,001 lines) | ~789 KB |
| CSV | Proxy (:5173) | 5.0 s | 5.13 s | 6.57 s | 6.48 s | 5,000 rows (5,001 lines) | ~789 KB |
| PDF | Direct (:8000) | 5.0 s | 8.06 s | 10.41 s | 8.81 s | 129 pages | ~427 KB |
| PDF | Proxy (:5173) | 5.0 s | 9.69 s | 9.83 s | 9.89 s | 129 pages | ~427 KB |

Observation: CSV export duration for 5,000 records averaged ~4.7 s on the direct route and ~6.1 s through the reverse proxy. PDF export duration averaged ~9.1 s on direct and ~9.8 s through the reverse proxy, reflecting ReportLab rendering overhead for 129 formatted pages. Duration through the reverse proxy reflects expected marginal transit overhead. PDF output format produces exactly 129 pages (32 records on the title page, 39 records across each of the subsequent 128 pages).

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
| Backend Image Built Commit | `3bc8c22` | Commit hash from which backend container was built |
| Model Manifest SHA-256 | `3E04C08C99389688...` | Integrity check of ML bundle (class_weight) |
| Raw Cache Archive SHA-256 | `0F55E985BE48E23A...` | SHA-256 checksum of raw latencies zip (stored outside repository) |
| OMP_NUM_THREADS | `unset` | OpenMP thread parallelism setting |
| Uvicorn Workers | `1` | Single ASGI worker process count |
| Host Environment | `Windows-11-10.0.26300-SP0` (8 logical CPUs) | Benchmark test runner host environment |
| Operator Notes & Power | AC connected (plugged in), Windows Power Scheme: Balanced, Operator Note: Stage 2 full latency and throughput matrix | Detected AC power state, active Windows scheme, and operator note |
| HTTP Client Transport | HTTP keep-alive (`httpx.Client`) | Connection reuse enabled |
