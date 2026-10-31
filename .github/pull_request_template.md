## Description

<!-- Provide a concise summary of the changes and the rationale behind them. -->

## Branch & Merge Rules Checklist
- [ ] Feature branch created from `main` (`feature/<name>`)
- [ ] Never committing directly to `main`
- [ ] Never force-pushing (`git push --force`)
- [ ] No secrets, datasets, or model binary artifacts committed
- [ ] If 3+ files were modified, an implementation plan was presented and approved first
- [ ] Project checks run locally and reported honestly (`make lint && make test`)

## Scope of Changes
- [ ] Backend
- [ ] ML
- [ ] Frontend (Note: Vite app loads `/src/App.jsx` without `createRoot()`; UI currently displays blank root shell until entrypoint mounting is resolved)
- [ ] DevOps / CI / Docker
- [ ] Documentation

## Verification & Testing
<!-- Detail exact commands run and paste raw terminal output. -->
- `make lint` / `ruff check` / `npm run lint`:
- `make test` / `pytest`:
- Docker build / smoke test (if applicable):

## Known Limitations / Notes
<!-- Mention any temporary stubs, pending upstream merges, or frontend blank UI status. -->
