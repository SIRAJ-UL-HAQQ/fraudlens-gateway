# FraudLens Gateway Frontend (`frontend/`)

The web dashboard and analyst triage console for the **FraudLens Gateway** — an explanation-reliable financial fraud detection gateway evaluating model stability under extreme class imbalance (PaySim dataset).

## Overview & Architecture

The frontend provides an accessible, dark-themed operational dashboard for fraud analysts and ML engineers. It implements:
- **Session-Based Authentication**: Encapsulated in `tokenStore` (backed by `sessionStorage`), with client-side expiry verification and de-duplicated 401 handling.
- **Contract-Conforming API Client**: Relative path requests (`/api/v1`, `/health`) with normalized error handling (`{ status, message, fieldErrors, retryable }`).
- **Health Monitoring & Persistent Banners**: Background polling using TanStack Query to detect degraded gateway states (`stub` mode, `unloaded` model 503, database disconnected, network unreachable).
- **Accessible Design Tokens**: Strictly following accessibility standards (`role="meter"` for risk scores, neutral icons for Low Risk and Stable states with **no check marks**, causality disclaimers for rule flags and SHAP attributions).
- **Live Transaction Feed & Triage**: 2-second real-time polling with visibility detection, two-way URL search params synchronization, and server ordering disclaimer (`created_at` descending).
- **Flagged Queue & Variant B Routing**: Segmented tabs for `Analyst Review` and `High Risk` with honest counts from API queries and educational banners detailing SRS Variant B routing.
- **Accessible Transaction Detail Drawer**: Slide-out dialog (`role="dialog"`, `aria-modal="true"`, focus trap, Escape key close) with PaySim raw financial parameters, latency indicators, and prominent unstable warnings.
- **SHAP Feature Attribution Bars**: Scaled horizontal bars relative to `max(|shap_value|)`, division-by-zero guards, signed formatting, explicit direction badges, and causality disclaimers.
- **Append-Only Review Workflow**: Immutable review audit trail and verdict submission form with zero optimistic updates and input preservation on 503/network failures.
- **Pipeline Metrics & Research Benchmark (`MetricsView`)**: Operational telemetry (total scored, explanation stability ratio, mean risk score, p50/p95 latency) paired with conditional offline ML benchmark evaluation (strategy comparison table, SVG charts, 2x2 confusion matrix grid).
- **Test Transaction Sandbox (`TestTransactionView`)**: Interactive single-transaction simulation with 4 contract-valid presets (Obvious Fraud, Obvious Legitimate, Borderline/Ambiguous, Zero Amount), boundary validation, and real-time inference result cards.
- **Audit Log Export (`ExportView`)**: Authenticated CSV/PDF export with parameter filtering, client-side step range validation, dynamic server safety cap handling, and browser blob download triggering.
- **Zero-Production Mock System**: Development-only mock interception panel driven by `fixtures.json` and configurable overrides (`stub`, `unloaded`, `disconnected`, `network_error`), completely tree-shaken from production builds.

## Technology Stack & Architecture Decisions

- **Framework**: React 18.3.1 + React DOM 18.3.1
- **Bundler & Dev Server**: Vite 5.4.21
- **Routing**: React Router 6.30.6
- **Data Fetching & Cache**: TanStack Query (React Query) 5.62.8
- **Styling**: Tailwind CSS 3.4.17 + PostCSS 8.5.32 + Autoprefixer 10.4.20
- **Icons**: Lucide React 0.468.0
- **Testing**: Vitest 2.1.9 + React Testing Library 16.1.0 + JSDOM 25.0.1
- **Linting & A11y**: ESLint 9.39.5 (Flat Config) + `eslint-plugin-react` (`react/no-danger: "error"`) + `eslint-plugin-jsx-a11y` 6.10.2 (WCAG AA)
- **Resilience**: Dual-level `ErrorBoundary` with "Reload View" recovery and fault resilience suite

### Pure Accessible SVG Charting (Zero 3rd-Party Dependencies)
To maintain zero runtime vulnerabilities, prevent bundle bloat, and guarantee complete accessibility:
- **Decision Distribution & Strategy Comparison**: Rendered using pure SVG elements (`<svg role="img" aria-label="...">`, `<title>`, `<desc>`).
- **Full Theme & High-DPI Support**: Vector-sharp on all screen resolutions, dark-mode matched, and zero external graphing runtime overhead.

## Getting Started

### Prerequisites
- Node.js 24 LTS (`v24.22.x` recommended, matching Docker Alpine base)
- npm 10+

### Installation
```bash
# Clean deterministic install
npm ci
```

### Development Scripts
```bash
# Start local dev server at http://localhost:5173 with API proxy to :8000
npm run dev

# Run Vitest unit & integration test suite
npm test

# Run ESLint validation
npm run lint

# Build optimized production bundle to dist/
npm run build

# Preview production build locally
npm run preview
```

## Mock Mode & Development Workflow

In development (`npm run dev`), a floating **Mock Mode Controls** panel is rendered at the bottom-left of the screen.
- **Toggle Mock Mode**: Intercepts `/api/v1/*` and `/health` requests using `fixtures.json` and `overrides.js`.
- **Simulate States**:
  - `Normal (Fixtures)`: Simulates healthy backend and standard PaySim transaction fixtures.
  - `Stub Heuristic Mode`: Simulates gateway running with heuristic stubs (renders amber `role="status"` banner).
  - `Model Unloaded (503)`: Simulates gateway with unloaded model weights (renders rose `role="alert"` banner).
  - `Database Disconnected (503)`: Simulates database failure (renders rose `role="alert"` banner).
  - `Network Offline (0)`: Simulates network disconnect / offline gateway.
- **Dev Credentials**: `analyst_1` / `password123` (supports 1-click auto-fill on `/login`).
- **Production Safety**: All mock logic is guarded with `import.meta.env.DEV` and lazy-loaded / tree-shaken during `npm run build`. No mock tokens, credentials, or fixture IDs exist in `dist/`.

## Component Guidelines

| Component | Role / Accessibility | Styling / Rule |
| :--- | :--- | :--- |
| `DecisionBadge` | Text + Icon | Emerald for `Low Risk` (neutral `Shield` icon, never check marks), Rose for `High Risk`, Amber for `Analyst Review`. |
| `ReliabilityBadge` | Accessible Tooltip + Button | Neutral `Shield` for `Stable` (no check marks), `AlertOctagon` for `Unstable`. Tooltip explains SHAP attribution limits; closes on `Escape`. |
| `RiskScoreBar` | `role="meter"` | Visible numeric label, `aria-valuetext="Risk score X on a 0 to 1 scale"`, **no threshold marks** (no contract fields exist). |
| `RuleFlagList` | Disclaimer Notice | Displays observed flags with note: *"Observed rule flags do not imply causality in gateway routing."* Rendered unconditionally for all transactions. |
| `ShapFeatureBars` | Semantic Horizontal Bars | Relative to `max(\|shap_value\|)`, division-by-zero guard, signed strings (`+0.1234`), text badges (`Pushes toward Fraud`), causality notice. |
| `StatusBanner` | `role="status"` / `role="alert"` | Displays model unloaded, database disconnected, stub mode, or unreachable gateway states. |
| `TransactionDetailDrawer` | Accessible Dialog | `role="dialog"`, `aria-modal="true"`, focus trap, Escape key close. Displays PaySim parameters, prominent unstable banner, and audit timeline. |
| `ReviewHistoryTimeline` | Append-Only Audit Log | Newest-first chronological timeline with reviewer attribution, timestamps, outcome badges, and investigation notes. |
| `AnalystReviewForm` | Verdict Submission | `outcome` selection, advisory character counter, zero optimistic updates, input preservation on error. |
| `OperationalMetricsCard` | Accessible Telemetry Grid | Total scored, explanation stability ratio, mean risk score, p50/p95 latency with tooltip *"Server-side scoring + explanation time"*. |
| `DecisionDistributionChart`| `role="img"` SVG | Stacked proportional bar showing Low Risk, Analyst Review, High Risk counts and shares. |
| `StrategyComparisonTable` | Semantic Benchmark Table | Class imbalance benchmark across None, Class Weight (Deployed), and SMOTE-ENN for Precision, Recall, F1, PR-AUC, and Stability. |
| `StrategyComparisonChart` | `role="img"` Grouped SVG Bar | Visual performance comparison across imbalance strategies. |
| `ConfusionMatrixGrid` | 2x2 Evaluation Grid | Accessible 2x2 grid for TP, FP, TN, FN at high-risk threshold with automated fallback when absent. |
| `TestTransactionView` | Interactive Simulation | 9 PaySim fields, client validation, 4 presets, real-time result card with latency tooltip and stub/unstable warning banners. |
| `ExportView` | Blob Downloader & Filter | CSV and PDF export with step/date filters, dynamic server safety cap error handling, and browser download triggering. |

## Contract & API Truth

- `backend/app/models/schemas.py` and `docs/api-contract.md` are the single source of truth for all API shapes and enums.
- Tests in `src/tests/badges.test.jsx` read `docs/openapi.json` directly at test time to verify exact enum parity (`GatewayDecision`, `ExplanationStatus`).
