# Frontend Conventions and Constraints

ROLE: Frontend engineer.
MODE: Planning. Use the browser to verify the UI and attach screenshots to your walkthrough.
GOAL: A professional analyst dashboard. It should look like a real fraud-operations product, not a student demo.

## Constraints
- React + Vite + Tailwind + Recharts. Plain JavaScript (JSX).
- All HTTP goes through `src/api/client.js`. Base URL from `VITE_API_URL`. A `VITE_USE_MOCKS=true` flag serves `src/mocks/fixtures.js` so the UI works before the backend is ready.
- Every data view has loading, empty and error states.
- Risk is never conveyed by colour alone: always a text label as well.
- Consistent design tokens (spacing, type scale, a restrained palette), responsive down to tablet width, keyboard-accessible controls.
- OUT OF SCOPE: changing the API contract (tell me if you need a change).
