---
name: ship
description: Commit and push work in this repo through the gate — tests, secret scan, citation check, detailed commit. Use when work reaches a milestone or the user says ship, commit, or push.
---

# Ship

This repo is a public assessment submission. Reviewers read the code, the
notebook outputs and `CITATIONS.md`, and one committed key disqualifies
everything. The gate exists for those three things.

## 1. First run only: bootstrap

Skip this step when `git rev-parse --is-inside-work-tree` succeeds.

1. `git init -b main`
2. Confirm that `git status --ignored` lists `.env`, `tasks.md`, `tasks.pdf`
   and `.venv/` as ignored. The repo's `.gitignore` covers them. The brief is
   marked "Confidential - Not for Distribution" and must stay out of the
   public repo.
3. The remote is a **public** GitHub repo named `CDAZZDEV-MLE-<YourName>`.
   Ask the user for the name and for a yes before creating it, because the
   repo becomes public as soon as it exists:
   `gh repo create CDAZZDEV-MLE-<Name> --public --source . --remote origin`

Done when `git status` works, those files show as ignored, and `origin`
points at the repo the user approved.

## 2. Gate: every check green before committing

1. **Tests.** Run `pytest -q` whenever a test file exists. It has to pass.
   Fix the cause of a failure. Change a test only when the behaviour change
   is intended and the user agrees.
2. **Lint.** Run `ruff check .` if ruff is installed, and fix what it flags.
3. **Stage, then scan.** Stage the files explicitly by path, then run
   `scripts/check_secrets.sh`. Any finding stops the ship. Move the value into
   `.env` locally or into Colab Secrets (`google.colab.userdata`) in notebooks,
   read it from the environment, and run the scan again.
4. **Notebooks.** Commit notebooks exactly as they were executed, outputs
   included, because reviewers score from those outputs. Cell outputs are
   inside the scanned diff, so a key printed by a cell fails step 3.
5. **Citations.** Every file in this commit that Claude wrote or changed is
   cited following the `cite-ai-usage` skill, and `scripts/check_citations.sh`
   exits 0.
6. **Review the staged set.** Read `git diff --cached --stat`. Only files
   that belong to this milestone should be there, and none of `.env`,
   `tasks.*` or `.venv/`.

Done when tests pass, the scan prints `clean`, every Claude-written file is
cited, and the staged set matches the milestone.

## 3. Commit

- Subject: imperative and specific ("Add Wilder-smoothed RSI with reference-value tests").
- Body: why the change exists, which task and rubric criterion it serves
  (for example "Task 1A, indicator accuracy"), and what verified it.
- End with the current Co-Authored-By attribution line for Claude.

## 4. Push

`git push -u origin main`. This is a solo repo with no branch protection, so
push straight to `main`. If the push is rejected, report the error to the user
and leave the history as it is.

Done when `git status` reports the branch up to date with `origin/main`.
Report the commit hash and what shipped.
