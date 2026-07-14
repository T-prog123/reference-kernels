#!/bin/bash
# PreToolUse guard (Bash): hard-block the ranked popcorn leaderboard submission.
# Agents may run `--mode test` / `--mode benchmark` / `--mode profile`, but the
# ranked `--mode leaderboard` run is the user's job — never automate it.
# Fail-CLOSED: if the command can't be parsed, any popcorn command is blocked.
# Blocks with exit code 2; stderr is shown to the model.
export PATH="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

input="$(cat)"

# Extract the shell command. Fall back to the raw payload if jq is unavailable.
if command -v jq >/dev/null 2>&1; then
  cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty')"
else
  cmd="$input"
  jq_missing=1
fi

# Not a popcorn command -> nothing to guard.
printf '%s' "$cmd" | grep -qiE '(^|[^[:alnum:]_])popcorn([^[:alnum:]_]|$)' || exit 0

# jq missing and it *is* a popcorn command -> can't parse reliably, block to be safe.
if [ -n "$jq_missing" ]; then
  echo "guard_popcorn: cannot parse command (jq missing); blocking popcorn to stay safe." >&2
  exit 2
fi

# Block only when the SUBMISSION MODE is leaderboard. Note: every command carries the
# '--leaderboard eigh' flag (the problem slug) — that must NOT trigger a block; only
# '--mode leaderboard' does.
if printf '%s' "$cmd" | grep -qiE -- '--mode[[:space:]=]+leaderboard'; then
  echo "guard_popcorn: blocked — '--mode leaderboard' is the ranked submission." >&2
  echo "Agents may only run --mode test / benchmark / profile. Leaderboard is the user's job." >&2
  exit 2
fi

exit 0
