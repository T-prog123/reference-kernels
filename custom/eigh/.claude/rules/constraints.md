# Hard constraints

These are non-negotiable competition/repo rules. Two are enforced mechanically by
hooks in `.claude/hooks/`; all apply regardless.

## No CUDA streams  (hook-enforced)

Kernel and submission files must not contain any stream-related API usage or the
substrings `stream`, `streams`, `cudaStream`, `current_stream`, `default_stream`.
Triton does not need them. `guard_streams.sh` rejects any Write/Edit to the working
`submission.py`, an `attempts/` or `winners/` copy, or `kernels/` scratch that
contains a `stream` token.

## Writes stay inside this directory  (hook-enforced)

All file creation/edits must stay under `custom/eigh/`. Reading files elsewhere
(the grader, sibling projects, docs) is allowed and expected. `guard_paths.sh`
blocks any Write/Edit whose target is outside this directory — which also makes it
impossible to accidentally edit the grader.

## Never modify the grader

`problems/linalg/eigh_py/{reference.py,eval.py,task.py,task.yml}` are the
authoritative checker/harness. Read them to understand the gates; never change
them. Changing the grader invalidates every result.

## Ranked submission is user-only  (hook-enforced)

Agents may run popcorn `--mode test` / `benchmark` / `profile` autonomously, but the
ranked **`--mode leaderboard`** submission is the user's job — attempts are limited.
`guard_popcorn.sh` hard-blocks any `--mode leaderboard` command (fail-closed).

## Secret hygiene

The popcorn/Discord auth token must **never** be written into a tracked file.
`custom-kernels` is force-pushed to a public fork via `update_git.sh`, so a leaked
token would be published. Credentials live only in the popcorn CLI's own
out-of-repo config.
