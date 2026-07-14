#!/bin/bash
# PreToolUse guard: block Write/Edit/NotebookEdit targeting paths outside this
# project directory. Reading anywhere is unaffected (this only runs on write tools).
# Blocks with exit code 2; the stderr message is shown to Claude.
export PATH="/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

ROOT="${CLAUDE_PROJECT_DIR:-/Users/titouan/Desktop/code/reference-kernels/custom/eigh}"

input="$(cat)"

if ! command -v jq >/dev/null 2>&1; then
  echo "guard_paths: jq not found; cannot verify write target — blocking to stay safe." >&2
  exit 2
fi

path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')"

# No path (or a tool without one) -> nothing to guard.
[ -z "$path" ] && exit 0

# Reject parent-directory traversal outright.
case "$path" in
  *"/../"*|*"/.."|"../"*)
    echo "guard_paths: refusing path with '..' traversal: $path" >&2
    exit 2
    ;;
esac

# Allow only paths inside ROOT (or ROOT itself).
if [ "$path" = "$ROOT" ] || [ "${path#"$ROOT"/}" != "$path" ]; then
  exit 0
fi

echo "guard_paths: blocked write outside project directory." >&2
echo "  target: $path" >&2
echo "  allowed root: $ROOT" >&2
echo "Writes must stay inside custom/eigh. Read-only access elsewhere is fine." >&2
exit 2
