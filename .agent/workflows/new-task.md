---
description: Start a new task on a dedicated feature branch, review context rules, produce an implementation plan, and pause for approval.
---

# Workflow: /new-task

Ask me for the task and branch name. Create the branch, read the context rule and relevant docs, produce an implementation plan, and stop for approval.

## Execution Steps

1. **Ask for Details**:
   - Inquire about the specific task requirements.
   - Confirm the target branch name following the format `feature/<name>`.

2. **Branch Creation**:
   - Check out and create the feature branch from the latest base:
     ```bash
     git checkout -b feature/<name>
     ```
   - Never commit directly to `main`.

3. **Context & Documentation Review**:
   - Read `.agent/rules/00-project-context.md` (Always On).
   - Read the area-specific rule file matching the task:
     - Backend: `.agent/rules/10-backend.md`
     - Frontend: `.agent/rules/20-frontend.md`
     - ML: `.agent/rules/30-ml.md`
     - DevOps: `.agent/rules/40-devops.md`
   - Read relevant documentation (e.g. `docs/api-contract.md`, `README.md`).

4. **Implementation Plan**:
   - For any task touching 3 or more files, draft a comprehensive implementation plan outlining:
     - Files to create or modify.
     - Architecture/contract impacts.
     - Verification plan and test cases.

5. **Approval Stop**:
   - Present the plan to the user and stop execution. Wait for explicit approval before proceeding with file edits.
