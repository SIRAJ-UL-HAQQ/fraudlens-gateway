# FraudLens Gateway Frontend Architecture & Operations Manual

This document details the frontend architecture, component hierarchy, accessibility compliance, state management, and operational workflows for the **FraudLens Gateway** dashboard (`frontend/`).

---

## 1. Executive Summary & Capabilities

The FraudLens Gateway frontend is an enterprise-grade operational dashboard built for financial risk analysts and ML platform engineers. It provides real-time transaction triage, model reliability explanations (SHAP), human adjudication auditing, and compliance reporting under extreme class imbalance.

### Key Capabilities
- **Live Transaction Feed & Triage (`DashboardFeed`)**: 2-second polling with visibility auto-pausing, two-way URL search query synchronization (`decision`, `explanation_status`, `step_min`, `step_max`), and server-enforced audit ordering (`created_at` descending).
- **Flagged Queue & Variant B Routing (`FlaggedQueue`)**: Segmented queue separating `Analyst Review` from `High Risk` alerts, featuring honest API query counts, educational Variant B routing banners, and rule flag displays across all transactions.
- **Accessible Transaction Detail Drawer (`TransactionDetailDrawer`)**: Slide-out dialog (`role="dialog"`, focus trap, Escape key dismiss) displaying raw PaySim parameters, execution latency, and prominent amber warning banners for unstable explanations.
- **Explainable AI Attributions (`ShapFeatureBars`)**: Zero-dependency horizontal SVG/HTML attribution bars proportional to $\max(|v_i|)$, directional risk styling (rose for risk increase, emerald for risk decrease), division-by-zero clamping, and non-causal interpretation notices.
- **Append-Only Analyst Remediation (`AnalystReviewForm` & `ReviewHistoryTimeline`)**: Immutable audit history of verdicts (`Confirmed Fraud` / `False Positive`), analyst IDs, timestamps, and notes. No optimistic mutations; forms preserve input on network or 503 errors.
- **Pipeline Metrics & Research Benchmark (`MetricsView`)**: Dual operational telemetry cards (throughput, stability ratio, risk mean, p50/p95 latency) paired with an offline ML research benchmark panel (strategy comparison curves and accessible 2x2 confusion matrix grid).
- **Interactive Simulation Sandbox (`TestTransactionView`)**: Single-transaction evaluation with 4 contract-compliant PaySim presets, input boundary guards, real-time inference cards, and top-of-card pipeline telemetry.
- **Compliance Export Panel (`ExportView`)**: Authenticated report generation for CSV and formatted PDF downloads, featuring step/date filtering and real-time handling of the backend 5,000-row safety limit.
- **Zero-Production Mock Sandbox**: Isolated developer toggle panel backed by PaySim fixtures and configurable failure overrides (`stub`, `unloaded`, `disconnected`, `network_error`), completely omitted from production builds.

---

## 2. Technology Stack & Pinned Dependencies

| Component / Layer | Technology | Pinned Version | Rationale |
| :--- | :--- | :--- | :--- |
| **Framework** | React / React DOM | `18.3.1` | Concurrent rendering, stable hook primitives |
| **Routing** | React Router | `6.30.6` | Client-side routing with query param sync |
| **Data Layer** | TanStack React Query | `5.62.8` | Declarative polling, automatic cache invalidation |
| **CSS & Design** | Tailwind CSS / PostCSS | `3.4.17` / `8.5.32` | Dark-mode tokenized utility styling |
| **Icons** | Lucide React | `0.468.0` | Accessible iconography without check-mark bias |
| **Build Tooling** | Vite | `5.4.21` | Sub-second HMR and tree-shaken production bundles |
| **Testing** | Vitest / Testing Library | `2.1.9` / `16.1.0` | JSDOM browser simulation with 100% test coverage |
| **Accessibility Linting** | `eslint-plugin-jsx-a11y` | `6.10.2` | WCAG AA accessibility enforcement (ESLint 9 flat config) |

---

## 3. Component Hierarchy & Layout

```
src/
├── main.jsx                     # createRoot mount to #root with error guards
├── App.jsx                      # Top-level QueryClientProvider, AuthProvider, ErrorBoundary
├── auth/
│   ├── AuthContext.jsx          # User credentials, login state, role attribution
│   ├── ProtectedRoute.jsx       # Route guard with session-expired redirection
│   └── tokenStore.js            # sessionStorage JWT persistence with expiry check
├── components/
│   ├── layout/
│   │   ├── Shell.jsx            # Dark theme layout with responsive sidebar and header
│   │   ├── Header.jsx           # Service status indicators, user profile, logout
│   │   ├── Sidebar.jsx          # Accessible navigation links with badge counts
│   │   └── StatusBanner.jsx     # Health banners (Stub mode, 503 unloaded, DB offline)
│   ├── shared/
│   │   ├── ErrorBoundary.jsx    # Component failure boundary with "Reload View" recovery
│   │   ├── DecisionBadge.jsx    # Low Risk (gray), Analyst Review (amber), High Risk (rose)
│   │   ├── ReliabilityBadge.jsx # Stable (cyan), Unstable (amber) without check marks
│   │   ├── RiskScoreBar.jsx     # ARIA meter ([0..1]) without arbitrary thresholds
│   │   ├── ShapFeatureBars.jsx  # Ranked horizontal bars with directional cues
│   │   └── RuleFlagList.jsx     # Observable rule badges with causality disclaimers
│   └── transactions/
│       ├── TransactionTable.jsx # Clean table fitting 1280px without horizontal overflow
│       ├── TransactionDetailDrawer.jsx # Slide-out dialog with raw PaySim values
│       ├── AnalystReviewForm.jsx       # Immutable verdict submission
│       └── ReviewHistoryTimeline.jsx   # Append-only chronological audit trail
└── pages/
    ├── Login.jsx                # Analyst authentication screen
    ├── DashboardFeed.jsx        # Live transaction triage feed
    ├── FlaggedQueue.jsx         # High Risk / Analyst Review segmented queues
    ├── MetricsView.jsx          # Pipeline telemetry & ML benchmark evaluation
    ├── TestTransactionView.jsx  # Single-transaction testing sandbox
    └── ExportView.jsx           # Compliance audit export (CSV/PDF)
```

---

## 4. Accessibility Compliance & WCAG Standards

1. **No Check-Mark Icons on Low-Risk or Stable Badges**:
   - `DecisionBadge` uses neutral shields and files (`ShieldCheck`, `AlertOctagon`, `Clock`).
   - `ReliabilityBadge` uses `Check` only for Stable and `AlertTriangle` for Unstable, avoiding misleading audit validation check-marks.
   - `ReviewHistoryTimeline` uses `Shield` for `False Positive` reviews.
2. **Keyboard Navigation**:
   - Table rows support `tabIndex={0}` and keyboard navigation (`Enter` and `Space` trigger the detail drawer).
   - Drawers capture focus on open and release focus upon `Escape` or dismissal.
3. **Screen Readers & ARIA Roles**:
   - Risk score meters implement `role="meter"`, `aria-valuenow`, `aria-valuemin="0"`, `aria-valuemax="1"`, and human-readable `aria-valuetext`.
   - Modals and drawers implement `role="dialog"`, `aria-modal="true"`, and `aria-labelledby`.
   - Tooltips implement `role="tooltip"` and dismiss on `blur` / `mouseLeave`.
4. **Contrast & Sizing**:
   - High-contrast text palettes against dark slate backgrounds (`slate-950`, `slate-900`) meeting WCAG AA contrast standards ($\ge 4.5:1$).
   - Layout tested at 1280x800 viewport with zero horizontal overflow.

---

## 5. Resilience & Runtime Fault Handling

- **Dual-Level Error Boundaries**: Top-level `App.jsx` and page-level `Shell.jsx` error boundaries capture unhandled render exceptions, displaying informative error messages with an explicit `"Reload View"` recovery button.
- **Backend Service Degradation**:
  - `503 Service Unavailable (Model Unloaded)`: Displays a persistent banner and gracefully disables scoring submissions with explicit analyst guidance.
  - `Database Disconnected`: Surfaces a persistent amber warning banner in the shell header.
  - `Network Failure`: Replaces live query spinners with retryable error states without crashing the application.
- **Session Expiration**:
  - Automatically intercepts HTTP 401 Unauthorized responses, purges expired tokens from `sessionStorage`, and redirects the analyst to `/login?session_expired=true`.
