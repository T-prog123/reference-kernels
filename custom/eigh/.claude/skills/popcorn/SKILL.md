---
name: popcorn
description: Benchmark the eigh kernel remotely on B200 via the popcorn CLI (GPU MODE KernelBot) and log results to notes/experiments.md. Agents may run test and benchmark autonomously; the ranked leaderboard submission is user-only and is hook-blocked.
---

# popcorn — remote submit & benchmark

All kernel execution happens **remotely on B200** through popcorn — there is no local
GPU. This skill covers running the allowed modes and recording the outcome.

## Usage policy (strict)

- **Use popcorn sparingly. Hard cap: at most 5 invocations per prompt.** Decide what
  you need to learn, then run the minimum.
- **Agents may run:** `test` (correctness) and `benchmark` (timing), plus `profile`.
  These are allow-listed, so they run without a permission prompt.
- **NEVER run `--mode leaderboard`.** That ranked submission is the user's job, and
  `guard_popcorn.sh` will hard-block it anyway.
- Always pass **`--no-tui`** so output is plain stdout you can parse.

## Commands (verified)

```bash
# Correctness — allowed for agents
popcorn submit --leaderboard eigh --gpu B200 --mode test      --no-tui submission.py

# Timing benchmark — allowed for agents
popcorn submit --leaderboard eigh --gpu B200 --mode benchmark --no-tui submission.py

# Ranked submission — USER-ONLY, hook-blocked for agents
popcorn submit --leaderboard eigh --gpu B200 --mode leaderboard --no-tui submission.py
```

General syntax: `popcorn submit --leaderboard <problem> --gpu <GPU> --mode <mode> <file>`.
Run against the working `submission.py` (or a specified `attempts/submission_vN.py`).

Setup/auth (`popcorn setup`, `register`, `join`) is handled by the user, separately.

## Output: save JSON, don't scrape stdout

With `--no-tui`, **progress goes to stderr and the final result to stdout**; no file
is written automatically. Always use **`--output`** to save the structured result
(it creates parent dirs) — parse *that*, not stdout:

```bash
# save a per-attempt JSON artifact (traceable to its vN)
popcorn submit --leaderboard eigh --gpu B200 --mode benchmark --no-tui \
  --output notes/runs/v<N>_benchmark.json submission.py

# capture absolutely everything (progress + errors) when debugging a failure
popcorn submit --leaderboard eigh --gpu B200 --mode test --no-tui \
  submission.py > notes/runs/v<N>_test.log 2>&1
```

Run from the project dir so relative `--output` paths land in `notes/runs/`. The JSON
is the source for per-case means and the geomean you log to `experiments.md`.

## Profiling (B200, Nsight Compute artifacts saved locally)

```bash
popcorn submit --leaderboard eigh --gpu B200 --profile-brev \
  --benchmark-index <N> --no-tui submission.py
```

Requires `POPCORN_BREV_PROFILER_URL` (or `BREV_PROFILER_URL`) in the environment.

## Required after every run: update the ledger

Update `notes/experiments.md` (see the file for the exact three-part layout):

1. **Runs log** — a row for this attempt: correctness, per-case times, computed
   geomean, records set, whether it entered `winners/`, verdict.
2. **Per-case best-of** — bump any of the 13 cells this attempt beat, regardless of
   its overall geomean.
3. **Champion + `winners/`** — only if it passed correctness AND beat the champion
   geomean: update the Champion row and copy the file into `winners/`.

Never record a local number — only B200 popcorn output counts.
