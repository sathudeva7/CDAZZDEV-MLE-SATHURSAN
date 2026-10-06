#!/usr/bin/env bash
# Fail if a credential would reach the public repo. A committed key disqualifies
# the whole submission (tasks.md, Section 2.3), so this runs before every commit.
#
#   scripts/check_secrets.sh          scan staged files (what the next commit holds)
#   scripts/check_secrets.sh --all    scan every file git would push, tracked or new
#
# Prints file:line and the kind of key, never the key itself. Exit 1 on any hit.
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'create the ship skill', Date: 2026-10-06

set -euo pipefail

# label|extended regex. Provider prefixes for every API this assessment uses,
# plus a generic catch for a long literal assigned to a key-like name.
PATTERNS=(
  "Groq key|gsk_[A-Za-z0-9]{20,}"
  "OpenRouter key|sk-or-v1-[A-Za-z0-9]{20,}"
  "Anthropic key|sk-ant-[A-Za-z0-9_-]{20,}"
  "OpenAI key|sk-(proj-)?[A-Za-z0-9]{32,}"
  "Hugging Face token|hf_[A-Za-z0-9]{30,}"
  "LangSmith key|lsv2_(pt|sk)_[A-Za-z0-9_]{20,}"
  "Google key|AIza[0-9A-Za-z_-]{35}"
  "Hardcoded credential|(api[_-]?key|token|secret|password)[\"']?[[:space:]]*[:=][[:space:]]*[\"'][A-Za-z0-9_./+-]{16,}[\"']"
)

if [[ "${1:-}" == "--all" ]]; then
  files=$(git ls-files --cached --others --exclude-standard)
  read_file() { cat -- "$1"; }
else
  files=$(git diff --cached --name-only --diff-filter=ACMR)
  read_file() { git show ":$1"; }
fi

hits=0
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  base=$(basename -- "$f")
  if [[ "$base" == .env || ( "$base" == .env.* && "$base" != .env.example ) ]]; then
    echo "$f: env file would be committed"
    hits=$((hits + 1))
    continue
  fi
  content=$(read_file "$f" 2>/dev/null) || continue
  for entry in "${PATTERNS[@]}"; do
    label=${entry%%|*}
    re=${entry#*|}
    flags=-nIE
    [[ "$label" == "Hardcoded credential" ]] && flags=-nIiE
    lines=$(printf '%s\n' "$content" | grep $flags -- "$re" | cut -d: -f1 | tr '\n' ' ' || true)
    if [[ -n "$lines" ]]; then
      echo "$f: $label on line(s) ${lines% }"
      hits=$((hits + 1))
    fi
  done
done <<< "$files"

if (( hits > 0 )); then
  echo "check_secrets: $hits finding(s). Move each value to .env / Colab Secrets and read it from the environment." >&2
  exit 1
fi
echo "check_secrets: clean"
