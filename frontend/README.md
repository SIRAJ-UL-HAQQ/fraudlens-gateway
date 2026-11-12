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
- **Zero-Production Mock System**: Development-only mock interception panel driven by `fixtures.json` and configurable overrides (`stub`, `unloaded`, `disconnected`, `network_error`), completely tree-shaken from production builds.

## Technology Stack

- **Framework**: React 18.3.1 + React DOM 18.3.1
- **Bundler & Dev Server**: Vite 5.4.14
- **Routing**: React Router 6.28.2
- **Data Fetching & Cache**: TanStack Query (React Query) 5.62.8
- **Styling**: Tailwind CSS 3.4.17 + PostCSS + Autoprefixer
- **Icons**: Lucide React 0.468.0
- **Testing**: Vitest 2.1.8 + React Testing Library 16.1.0 + JSDOM 25.0.1
- **Linting**: ESLint 9.17.0 (Flat Config) + `eslint-plugin-react` (`react/no-danger: "error"`)

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

In development (`npm run dev`), a floating **Mock Mode Controls** panel is rendered at the bottom-right of the screen.
- **Toggle Mock Mode**: Intercepts `/api/v1/*` and `/health` requests using `fixtures.json`.
- **Simulate States**:
  - `Normal (Fixtures)`: Simulates healthy backend and standard PaySim transaction fixtures.
  - `Stub Heuristic Mode`: Simulates gateway running with heuristic stubs (renders amber `role="status"` banner).
  - `Model Unloaded (503)`: Simulates gateway with unloaded model weights (renders rose `role="alert"` banner).
  - `Database Disconnected (503)`: Simulates database failure (renders rose `role="alert"` banner).
  - `Network Offline (0)`: Simulates network disconnect / offline gateway.
- **Dev Credentials**: `analyst_1` / `password123` (supports 1-click auto-fill on `/login`).
- **Production Safety**: All mock logic is guarded with `import.meta.env.DEV` and tree-shaken during `npm run build`. No mock tokens or fixture IDs exist in `dist/`.

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

## Contract & API Truth

- `backend/app/models/schemas.py` and `docs/api-contract.md` are the single source of truth for all API shapes and enums.
- Tests in `src/tests/badges.test.jsx` read `docs/openapi.json` directly at test time to verify exact enum parity (`GatewayDecision`, `ExplanationStatus`).
