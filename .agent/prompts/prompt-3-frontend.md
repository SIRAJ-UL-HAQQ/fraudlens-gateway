# Prompt 3: Frontend

ROLE: Frontend engineer.  
MODE: Planning. Use the browser to verify the UI and attach screenshots to your walkthrough.

GOAL: A professional analyst dashboard. It should look like a real fraud-operations product, not a student demo.

CONSTRAINTS (also copy into .agent/rules/20-frontend.md):
- React + Vite + Tailwind + Recharts. Plain JavaScript (JSX).
- All HTTP goes through src/api/client.js. Base URL from VITE_API_URL.
  A VITE_USE_MOCKS=true flag serves src/mocks/fixtures.js so the UI works
  before the backend is ready.
- Every data view has loading, empty and error states.
- Risk is never conveyed by colour alone: always a text label as well.
- Consistent design tokens (spacing, type scale, a restrained palette), responsive
  down to tablet width, keyboard-accessible controls.

TASKS:
1. App shell: sidebar navigation, header, routing, login page, auth token handling.
2. TransactionFeed: table with risk score, decision badge (Low Risk / High Risk /
   Analyst Review), rule flags, timestamp; sorting, filters, pagination.
3. FlaggedView: High Risk and Analyst Review only. Row detail drawer showing top SHAP
   features as a horizontal bar chart, the explanation-reliability badge
   (Stable/Unstable with a tooltip explaining what it means), and a review form
   (Confirmed Fraud / False Positive + note) calling the review endpoint.
4. MetricsPanel: Precision, Recall, F1, PR-AUC, MCC cards; confusion matrix;
   a strategy-comparison view (metrics and SHAP stability per imbalance strategy);
   API latency p50/p95.
5. TestTransactionForm: submit a PaySim-style transaction, show the score, decision,
   SHAP bars and reliability live. Include 3 preset buttons (obvious fraud, obvious
   legitimate, borderline).
6. Export control: date range + CSV/PDF download.
7. Component tests for the badge and the decision-to-colour/label mapping.

ACCEPTANCE: `npm run build` and lint pass; every screen works in mock mode and against
the live backend; screenshots of each screen in the walkthrough.  
OUT OF SCOPE: changing the API contract (tell me if you need a change).
