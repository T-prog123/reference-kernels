#!/bin/bash
# PreToolUse guard: forbid CUDA-stream tokens in implementation files.
# Scope: only .py implementation files (submission.py, attempts/, winners/, kernels/).
# READMEs, docs, rules, and notes (which legitimately discuss the ban) are not scanned.
# Blocks with exit code 2; stderr is shown to Claude.
export PATH="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

ROOT="${CLAUDE_PROJECT_DIR:-/Users/titouan/Desktop/code/reference-kernels/custom/eigh}"

input="$(cat)"

if ! command -v jq >/dev/null 2>&1; then
  echo "guard_streams: jq not found; cannot verify content — blocking to stay safe." >&2
  exit 2
fi

path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')"
[ -z "$path" ] && exit 0

# Only enforce for .py implementation files: the working submission.py at the
# project root, archived attempts/, promoted winners/, and any kernels/ scratch.
# Non-.py files (READMEs, notes) in those dirs are exempt so they can mention the ban.
case "$path" in
  "$ROOT"/submission.py|"$ROOT"/submission_*.py|"$ROOT"/attempts/*.py|"$ROOT"/winners/*.py|"$ROOT"/kernels/*.py) : ;;
  *) exit 0 ;;
esac

# Collect the text being written across Write / Edit / NotebookEdit shapes.
content="$(printf '%s' "$input" | jq -r '
  [ .tool_input.content,
    .tool_input.new_string,
    .tool_input.new_source
  ] | map(select(. != null)) | join("\n")')"

[ -z "$content" ] && exit 0

if printf '%s' "$content" | grep -iq 'stream'; then
  echo "guard_streams: blocked — CUDA-stream token found in an implementation file." >&2
  echo "  target: $path" >&2
  echo "Forbidden: stream / streams / cudaStream / current_stream / default_stream." >&2
  echo "Triton kernels must not use or mention streams (competition rule)." >&2
  exit 2
fi

exit 0
