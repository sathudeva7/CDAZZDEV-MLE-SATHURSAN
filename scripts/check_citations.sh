#!/usr/bin/env bash
# Check that every inline AI-assistance / adapted-source marker is well formed
# and that each file carrying one is listed in CITATIONS.md. Uncited AI use is
# penalised for academic integrity (assessment brief, Section 2.2).
#
#   scripts/check_citations.sh    exit 1 on any problem
#
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Create the cite-ai-usage skill', Date: 2026-10-06

set -euo pipefail

# Built from pieces so this script's own source never matches its own search.
AI="AI-""ASSISTED:"
SRC="SO""URCE:"
MARKER="(#|<!--)[[:space:]]*(${AI}|${SRC})"
VALID_AI="${AI} [^(]+ \([^)]+\), Prompt: '.+', Date: [0-9]{4}-[0-9]{2}-[0-9]{2}"
VALID_SRC="${SRC} Adapted from https?://[^ ,]+"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  files=$(git ls-files --cached --others --exclude-standard)
else
  files=$(find . -type f -not -path './.git/*' -not -path './.venv/*' | sed 's|^\./||')
fi

problems=0
cited_files=()
while IFS= read -r f; do
  case "$f" in
    ""|.claude/*|CITATIONS.md|tasks.md|tasks.pdf) continue ;;
  esac
  marker_lines=$(grep -nIE -- "$MARKER" "$f" 2>/dev/null || true)
  [[ -z "$marker_lines" ]] && continue
  cited_files+=("$f")
  while IFS= read -r hit; do
    lineno=${hit%%:*}
    text=${hit#*:}
    if [[ "$text" =~ $AI ]] && ! grep -qE -- "$VALID_AI" <<< "$text"; then
      echo "$f:$lineno: malformed $AI line, expected: $AI <Tool> (<model>), Prompt: '<request>', Date: YYYY-MM-DD"
      problems=$((problems + 1))
    elif [[ "$text" =~ $SRC ]] && ! grep -qE -- "$VALID_SRC" <<< "$text"; then
      echo "$f:$lineno: malformed $SRC line, expected: $SRC Adapted from <url>, file: <file>, Lines <a-b>"
      problems=$((problems + 1))
    fi
  done <<< "$marker_lines"
done <<< "$files"

if (( ${#cited_files[@]} > 0 )); then
  if [[ ! -f CITATIONS.md ]]; then
    echo "CITATIONS.md is missing but ${#cited_files[@]} file(s) carry citation markers"
    problems=$((problems + 1))
  else
    for f in "${cited_files[@]}"; do
      if ! grep -qF -- "$f" CITATIONS.md; then
        echo "$f: has inline citations but no row in CITATIONS.md"
        problems=$((problems + 1))
      fi
    done
  fi
fi

if (( problems > 0 )); then
  echo "check_citations: $problems problem(s)" >&2
  exit 1
fi
echo "check_citations: ${#cited_files[@]} cited file(s), all listed in CITATIONS.md"
