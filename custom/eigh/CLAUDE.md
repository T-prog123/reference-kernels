# eigh — batched symmetric eigendecomposition (Triton kernel competition)

This directory is the **Claude working environment** for a GPU-kernel competition
entry. The goal is a legitimate, robust batched real-symmetric eigensolver written
in **Triton** — not a benchmark exploit.

## Execution reality — read this first

- **Claude runs locally on this Mac. There is no GPU here.** `torch`, `triton`,
  and CUDA are **not** available. Do **not** attempt to run `eval.py`, import
  `triton`/`torch`, or execute any kernel locally. It cannot work and wastes turns.
- **The kernels run remotely on B200** via the popcorn CLI (GPU MODE KernelBot).
  All correctness, benchmarking, and profiling happen there — never locally.
- Locally, Claude only **authors and reasons about** code and notes.

## The task

Implement `custom_kernel(data) -> (Q, L)` in `submission.py`:

- Input `data`: `[batch, n, n]` CUDA `float32`, symmetric up to FP32 roundoff.
- `Q`: `[batch, n, n]` `float32`, columns are orthonormal eigenvectors.
- `L`: `[batch, n]` `float32`, eigenvalues **sorted ascending**.
- Convention matches `torch.linalg.eigh(A)` (the reference).

Correctness is **invariant-based**, not elementwise vs a reference solver: the
checker gates on residuals of `A@Q − Q@diag(L)`, reconstruction `Q@diag(L)@Qᵀ − A`,
orthogonality `QᵀQ − I`, and ascending order. Gates are FP64-measured and
dimension-scaled. Sign flips and eigenspace rotation are allowed. Ranking among
passing entries is by **geomean runtime over the benchmark cases** on B200.

## Where things live

- `claude_docs/` — **theory canon** (read-only reference). The four algorithm
  families and the per-shape decision table live in
  `batched_cuda_symmetric_evd_methods.md`; **how to make code fast + read `nsys`/`ncu`
  profiles** lives in `cuda_performance_and_profiling_guide.md` (bottlenecks, arithmetic
  intensity, occupancy, tensor cores, a fast interpretation table); official rules in
  `official_rules.rtf`.
- `problems/linalg/eigh_py/` (outside this dir) — the **authoritative grader**
  (`reference.py`, `eval.py`, `task.py`, `task.yml`). Read for reference; **never edit**.
- `notes/design.md` — **living strategy**: which approach we're betting on and why.
  Holds the **Attempts log** (per-version index: idea + links to code and results) —
  the single place to look up "what was tried and why". Maintained by `kernel-designer`.
- `notes/experiments.md` — **results ledger**: every remote run and its numbers.
- `submission.py` — the **active working kernel**: one self-contained file (Triton
  inline) that you iterate on in place. This is what gets submitted; it cannot
  import local modules.
- `attempts/` — **every idea we try**, working or not: full self-contained
  `submission_vN.py` copies with a parent+hypothesis header. The complete archive.
- `winners/` — the subset of attempts that **improved the overall benchmark
  geomean** (same `vN`); the record progression of submittable champions.
- `kernels/` — optional scratch for drafts/snippets; anything used **must be
  inlined** into `submission.py` before a run.

## Iteration discipline (this is a long, iterative process)

1. Before proposing any change, **read `notes/design.md` and `notes/experiments.md`**
   so you don't repeat a dead end.
2. Iterate **in place on the single working `submission.py`**. Every distinct idea
   you evaluate is archived to `attempts/submission_vN.py` (parent+hypothesis
   header) — **keep all ideas, even ones that fail or regress.** Every attempt also
   gets an entry in the **Attempts log** in `notes/design.md` (idea + links), not
   just a code header — so the reasoning is discoverable, not stranded in the file.
3. After **every** remote run, update `notes/experiments.md`: log the attempt's
   per-case times + geomean, update the **per-case best-of** table for any cell it
   beat, and — only if the overall geomean improved — update the **champion** and
   copy the file to `winners/submission_vN.py`.
4. **Ranking is the overall geomean, but the champion is a shape-dispatcher**: a
   per-case win that regresses geomean is still valuable — it's grafted into the
   dispatcher, not discarded. Never discard an attempt without recording why.

## Hard constraints (see `.claude/rules/`)

- **No CUDA streams** — no `stream`/`cudaStream`/`current_stream`/`default_stream`
  tokens in kernel/submission files. Enforced by a hook.
- **Writes stay inside this directory.** Reading elsewhere is fine. Enforced by a hook.
- **Never modify the grader** under `problems/linalg/eigh_py/`.
- **No gaming**: no fingerprinting exact benchmark shapes/seeds, no hardcoded
  outputs, no reading checker internals at runtime. Build a real eigensolver.
- **Keep it feasible**: the submission is a single self-contained file, **≤ 3000
  lines**, **heavily commented** (up to ~⅓ of lines, explaining code + design +
  results) — see `implementation-style.md`.
- **≤ 5 dispatch paths (HARD MAX, fewer is better)**: at most 5 distinct kernel
  code paths, including fast paths, keyed to size regime only — never per-shape.
  See `contract-and-integrity.md`.
- **No secrets in the repo** — the popcorn/Discord token never goes in a tracked
  file (`custom-kernels` is force-pushed to a fork).

## Remote runs

Benchmarking goes through the **popcorn CLI** — see `.claude/skills/popcorn/SKILL.md`
for the verified commands. The CLI is installed; **account auth and joining the
`eigh` leaderboard (`popcorn setup` / `register` / `join`) are done by the user**.

**popcorn usage policy:** use it sparingly, **max 5 invocations per prompt**. Claude
may run `test` (correctness), `benchmark` (timing), and `profile`. Claude **never**
runs the ranked leaderboard submission — that is the user's job.

## Orchestrator loop (default working mode)

**You (the main session) are the orchestrator. Preserve your context: delegate the
token-heavy work to subagents and keep only the thread of the conversation and
pointers to files.** The durable state lives on disk (`notes/design.md`,
`notes/experiments.md`, `attempts/`, `winners/`), so you don't need to hold detail in
context — agents reload it from disk each time and return a short summary.

Default cycle for each iteration:

1. **Discuss** the big move with the user (this is what stays in your context).
2. **`kernel-designer`** → decide the next approach, **mining the per-case best-of
   table** for kernels to build or graft into the dispatcher; updates
   `notes/design.md`. Returns: "try X for regime Y, because…".
3. **`kernel-implementer`** → edits the working `submission.py` and archives the
   idea to `attempts/submission_vN.py`. Returns: what changed + line count.
4. **`benchmark-runner`** → runs popcorn `test`/`benchmark` (never leaderboard),
   logs per-case times + geomean, updates the per-case table, and promotes to
   `winners/` only on overall-geomean improvement. Returns: correctness + geomean +
   which records it set.
5. **`submission-reviewer`** → integrity/contract/size gate before you'd promote it.
   Returns: PASS/FAIL + top issue.
6. **Relay** the distilled outcome to the user and decide the next move.

Delegate implementation and benchmark-log reading — do **not** do them inline, or
your context fills with edit churn and raw logs. Read files yourself only when you
need a specific detail to talk to the user.

### The agents

- `kernel-designer` — read/plan the next approach from canon + notes (writes design.md).
- `kernel-implementer` — author the self-contained submission file for a decision.
- `benchmark-runner` — run popcorn (test/benchmark/profile), parse, log to ledger.
- `submission-reviewer` — final integrity/contract/size gate.
