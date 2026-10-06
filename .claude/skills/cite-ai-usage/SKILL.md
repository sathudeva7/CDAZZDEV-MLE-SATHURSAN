---
name: cite-ai-usage
description: Cite AI-assisted or adapted code in the assessment's required format, with an inline marker plus a CITATIONS.md row. Use whenever writing or changing code, notebook cells or docs in this repo, or adapting code from a repo or docs page.
---

# Cite AI usage

AI tools are allowed in this assessment, but uncited AI use is penalised for
academic integrity (brief, Section 2.2). The interviewer will also ask about
every block, so each citation has to record what was actually asked. Every
piece Claude writes gets two records: an inline **marker** where the code
lives, and a **row** in `CITATIONS.md`.

## 1. Inline marker

```
# AI-ASSISTED: Claude (<model id>), Prompt: '<request>', Date: <YYYY-MM-DD>
```

- **Model id**: the id of the model running this session, for example `claude-opus-5-5`.
- **Prompt**: the user's request, verbatim when it stands on its own. When the
  user only replied "ok go" or "yes", write a faithful one-line summary of
  the request they approved, such as `'Create the cite-ai-usage skill'`. Keep it on one line.
- **Date**: today's date.

Where the marker goes:

| What Claude wrote | Marker placement |
|---|---|
| A whole file | Once, at the top, after any shebang or module docstring opener |
| A function or class added to an existing file | Directly above it |
| A notebook code cell | First line of the cell |
| A Markdown or HTML doc | `<!-- AI-ASSISTED: ... -->` as the first line |
| A change to already-cited code, for a new request | A new marker line under the existing one, so the history stays visible |

Use the comment syntax of the file: `#` for Python, shell, YAML and TOML, and
`<!-- -->` for Markdown and HTML.

## 2. Adapted code

When the code follows an outside repo, docs page or tutorial, the marker
names the source instead:

```
# SOURCE: Adapted from <url>, file: <file>, Lines <a-b>
```

For a docs page with no file or line numbers, use `section: <heading>`
instead of `file:` and `Lines`. When Claude writes adapted code, the block
gets both lines: `SOURCE` above `AI-ASSISTED`.

## 3. CITATIONS.md row

Add one row per file per request to the matching table in the root
`CITATIONS.md`. Use the repo-relative path in backticks, so that
`scripts/check_citations.sh` can find it. "What it produced" names the
artifact in a few words, such as "RSI with Wilder smoothing".

When Task 2 generates data with a teacher model, put the full system prompt
in an appendix notebook cell or a README section, and point to it from the
"Teacher-model data generation" section.

## 4. Verify

Run `scripts/check_citations.sh`. Done when it exits 0: every marker is well
formed and every file carrying one has its row.
