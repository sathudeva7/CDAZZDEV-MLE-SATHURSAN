---
name: ship
description: Ship work in this repo through the gate — tests, secret scan, citation check — then a branch, a detailed commit and a PR merged into main on the user's yes. Use when work reaches a milestone or the user says ship, commit, or push.
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

## 3. Branch and commit

Every commit lands on its own branch, so main only ever changes through a PR.

1. Draft the commit message:
   - Subject: imperative and specific ("Add Wilder-smoothed RSI with reference-value tests").
   - Body: why the change exists, which task and rubric criterion it serves
     (for example "Task 1A, indicator accuracy"), and what verified it.
   - End with the current Co-Authored-By attribution line for Claude.
2. Show the user the gate results, the staged set and the message, and wait
   for their yes.
3. Branch from an up-to-date main: `git switch main && git pull --ff-only`,
   then `git switch -c <task>-<topic>` in kebab case, for example
   `task1a-data-pipeline`. Staged changes carry over to the new branch.
4. Commit.

## 4. Push, PR, merge

1. `git push -u origin <branch>`.
2. Open the PR with `gh pr create --base main`, the commit subject as the
   title, and a body written following the `pr` skill. Show the user the PR
   URL.
3. Merge only after the user's second yes, given for that PR:
   `gh pr merge <number> --merge --delete-branch`, then
   `git switch main && git pull --ff-only`.

If a push, PR or merge is rejected, report the error to the user and leave
the history as it is.

Done when the PR is merged, its branch is deleted, and local main is up to
date with `origin/main`. Report the commit hash, the PR URL and what shipped.
