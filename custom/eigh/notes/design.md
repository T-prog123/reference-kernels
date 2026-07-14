# eigh — living design strategy

The high-level plan and reasoning. **Canon theory** (the four algorithm families,
math, and the per-shape decision table) lives in
`../claude_docs/batched_cuda_symmetric_evd_methods.md` — this file is *our evolving
stance*, cross-linked to results in `experiments.md`.

Maintained by the `kernel-designer` agent (and you). Update it whenever the bet
changes; don't let it drift from what the ledger shows.

---

## Current bet

> **Status (2026-07-14):** under re-evaluation — see the Attempts log below. v0 (torch)
> is still champion (geomean 48 774 µs). Block-Jacobi (v2) proved too slow at n=512;
> the large-n pipelines all lose to torch. The "improved-coding" retries **regressed**:
> v5 (native QR, 138 k) is worse than its parent v3 (107 k); v6 (blocked-WY D&C, 108 k)
> is worse than its parent v4 (90 k). Ordering holds: **D&C > QR** (v6 < v5, as v4 < v3),
> but no Triton path beats torch. Best Triton attempt remains **v4 (90 k)**. Native
> on-GPU inner solves (QR especially) are slower than deferring to torch LAPACK — the
> real blocker is still a GPU-native tridiagonal solve that beats fused LAPACK.

**Batched/block Jacobi for the medium-n, large-batch regime.** The canon's #1
theoretical avenue for exactly these shapes: Jacobi has two levels of parallelism
(across the batch AND across disjoint rotations within a matrix), produces highly
orthogonal `Q` "for free", and the invariant-based gates here are loose
(`eigen_rtol ≈ 200·n·eps`, `orth_rtol ≈ 100·n·eps`), which suits an iterative
method that converges to a few ULP. Householder→tridiagonal pipeline is the plan
for `n ≥ 1024`, where Jacobi's `O(sweeps·n³)` cost stops being competitive.

**Opening kernel (v1, `attempts/submission_v1.py`).** A genuine batched *cyclic*
Jacobi using the **round-robin (Brent–Luk 1-factorization) parallel ordering**:
one Triton program per matrix; the whole matrix `B` and the accumulator `V` are
held resident as `[BN,BN]` tiles; each of the `BN−1` rounds per sweep applies
`BN/2` **disjoint** plane rotations *simultaneously* as one orthogonal tile `J`,
so a round is just three `tl.dot`s (`B ← Jᵀ B J`, `V ← V J`) — tensor-core-friendly
rather than scalar. Rotation angles are the standard stable Jacobi formula
(`tau=(a_qq−a_pp)/(2a_pq)`, `t=sgn(tau)/(|tau|+√(1+tau²))`), computed per-index with
the "i-as-p" convention so the ±s placement across a pair falls out automatically.
Eigenvalues = diagonal of the converged `B`; sorting ascending + column permutation
is done in torch in the wrapper (robust, cheap). A `torch.linalg.eigh` fallback
covers every shape the Triton path does not take, and the whole Triton path is
wrapped in try/except → torch, so **correctness gates always pass** even if the
kernel misbehaves.

**Regime the v1 Triton path actually covers:** only `n` that fits one Triton
program. B200 SRAM is ~228 KB; a single fp32 `n×n` tile is `n²·4` bytes and we need
~4 resident tiles (`B`, `V`, `J`, a matmul temp): `n=64→4·16 KB=64 KB` (safe),
`n=128→4·64 KB=256 KB` (over SRAM, register-spill risk), `n=256→256 KB per tile`
(does **not** fit). So the resident-single-block design is realistic only up to
`n≈128`. Dispatch threshold is `n ≤ 128` (by magnitude, not shape fingerprint).
Among the competition shapes this exercises **only `n=32`** (the next size, 176,
already exceeds the fit); `n = 176, 352, 512, 1024, 2048, 4096` all fall back to
`torch.linalg.eigh` in v1. Covering 176–512 needs **block-Jacobi** (tile the matrix
so only a block-pair is resident) — that is the next step, not v1.

## Attempts log (per-version index — read this first)

**The single lookup point for "what was tried and why."** One entry per version:
the idea in a few lines + links to the code (deep detail in the file header) and
the results (`experiments.md`). The designer maintains this: write the idea when
proposing an attempt; reconcile the outcome on the next pass. Deep algorithm/scope
detail stays in the code header — this is the index, not a copy.

- **v1** — `attempts/submission_v1.py` · not benchmarked standalone.
  In-SRAM round-robin (Brent–Luk) *cyclic Jacobi*, one Triton program per matrix,
  whole matrix + accumulator resident as tiles, each round = 3 `tl.dot`s. Fits only
  `n≤128` → among competition shapes touches only n=32; torch fallback elsewhere.
  Opening kernel; superseded by v2 for the medium-n reach.
- **v2** — `attempts/submission_v2.py` · [experiments.md v2 row] · **DEAD END at n=512.**
  Two-level *block-Jacobi* (block Brent–Luk ordering, reuses v1 as the local
  block-pair EVD) to reach n=176/352/512. Correct machinery, but **killed at ~360 s**
  on n=512×640: matmul-form Jacobi is O(sweeps·n⁴) and n⁴ dominates. Possibly still
  competitive at n=176/352 — untested (needs a single-index benchmark).
- **v3** — `attempts/submission_v3.py` · [experiments.md v3 row] · **DEAD END as-is.**
  Householder → tridiagonal → **QR** → backtransform; reduction+backtransform in
  Triton (tensor-core GEMM), but the inner solve is **deferred to `torch.eigh(T_dense)`**,
  which re-tridiagonalizes → strictly more work than v0. Geomean 107 k µs (~6× slower
  than torch on the n≥1024 cases it engages). Value: validated Triton reduction infra.
- **v4** — `attempts/submission_v4.py` · [experiments.md v4 row] · **best Triton attempt; still loses to torch.**
  Same pipeline with a **genuine Cuppen D&C** inner solve (in torch). Geomean 90 k µs,
  ~4.5× slower than v0 — but **clearly beats v3** (n=2048: 729 k vs 1173 k, −38%).
  Confirms **D&C > QR** for large-n all-vectors. To ever beat v0, the redundant
  reduction (torch re-tridiagonalizing) must be removed — needs a GPU-native
  tridiagonal D&C that consumes our Triton-reduced T directly.
- **v5** — parent **v3** · plan: [`notes/plans/v5_qr_improvement_plan.md`](plans/v5_qr_improvement_plan.md) · code: `attempts/submission_v5.py` · [experiments.md v5 row] · **REGRESSION vs v3; dead end.**
  Same theory as v3 (Householder → tridiag → **QR** → backtransform), improved coding.
  Key change: **native implicit-shift QL/QR inner solve consuming `(d,e)` directly**
  (removes v3's redundant `eigh(T_dense)` re-tridiagonalization), leaner dlatrd, fused
  `ormqr`, TF32. **Outcome: geomean 138 129 µs — worse than v3 (107 332) and 2.8× v0
  (48 774); loses to sibling v6 (108 439).** Movers ~1.9–3× worse than v3 (n=2048:
  3 500 k vs 1 173 k). The on-GPU sequential/divergent QR inner solve is far slower
  than v3's deferral to torch LAPACK — killing the redundant retridiag did not pay off.
  **Native QR is the wrong inner solver; QR family closed in favor of D&C.**
- **v6** — parent **v4** · plan: [`notes/plans/v6_dc_improvement_plan.md`](plans/v6_dc_improvement_plan.md) · code: `attempts/submission_v6.py` · [experiments.md v6 row] · **REGRESSION vs v4; best Triton attempt overall but still loses to torch.**
  Same theory as v4 (Householder → tridiag → **Cuppen D&C** → backtransform), improved
  coding. Key change: **blocked WY Q1/backtransform** replacing the unblocked `sorgtr`
  loop, TF32, autotuned GEMM, cheap fp32 probe self-check; D&C kept as-is. **Outcome:
  geomean 108 439 µs — worse than v4 (90 296) and 2.2× v0 (48 774), but beats sibling
  v5 (138 129) and every prior QR attempt.** Movers ~1.4–1.5× worse than v4 (n=2048:
  1 113 k vs 729 k). The blocked-WY reduction did not recover v4's deferred-D&C speed —
  the on-GPU pipeline is still slower than torch's fused LAPACK. **Confirms D&C > QR
  again (v6 < v5). No Triton path has beaten torch; v0 remains champion.**

## Shape regimes → intended approach

Benchmark shapes: n = 32, 176, 352, 512 (batch up to 640), 1024, 2048, plus mixed
/ rank-deficient / clustered / LAPACK-spectrum stress cases.

| Regime | Leaning | Rationale (see canon) | Status |
|---|---|---|---|
| small n, large batch (32–512) | Batched/block Jacobi | Canon #1 for these shapes: two levels of parallelism, batch-natural, excellent orthogonality, gates are loose enough for an iterative solver | v1: full-tile round-robin scalar-Jacobi covers only `n≤128` (fits one program → exercises n=32); **block-Jacobi TODO** for 176/352/512 (don't fit resident) |
| n = 1024 | Householder pipeline, **D&C inner solve** (not QR) | Crossover to tridiagonal ~n=1024; D&C is the right inner solver when all vectors are needed | v3 (QR) + v4 (D&C) built: both ~4.5–6× slower than torch; **D&C > QR**; torch still best. Blocker: deferred inner solve re-tridiagonalizes → need GPU-native tridiag D&C |
| n ≥ 2048 | Householder pipeline, D&C inner solve | Classical route dominates large dense; tiny batch (8/2) so per-matrix throughput matters | v4 (D&C) best so far (729 k µs) but still 5.7× slower than torch's 128 k; same blocker as n=1024 |

---

## The four families (pointers to canon)

- **Jacobi / block-Jacobi** — iterative plane rotations; two levels of
  parallelism; strong orthogonality. Notes: **our opening bet.** v1 uses the
  round-robin parallel ordering (`BN/2` disjoint rotations/round as one tile `J`,
  applied via `tl.dot`), which is the GPU-natural, tensor-core-friendly form.
  Resident single-block only fits `n≤128`; scaling to 176–512 requires block-Jacobi
  (extract a block-pair, local EVD, apply block similarity) — the canon calls this
  out as the right variant for n=512.
- **Householder tridiagonal pipeline** — reduce → tridiagonal solve →
  backtransform. Notes: the plan for `n≥1024`. Backtransform (`Q=U@Y`) is mandatory
  here (we must return `Q`) and is the expensive step; multi-phase so harder to batch.
- **Shifted QR** — inner tridiagonal solver. Notes: candidate inner solver for the
  tridiagonal stage; robust but sequential/divergent under batching — less GPU-natural.
- **Divide-and-conquer** — inner tridiagonal solver. Notes: likely the best inner
  solver when all eigenvectors are needed (they are); workspace-heavy, irregular
  recursion. Reach for it inside the `n≥1024` pipeline.

## Open ideas / parking lot

- **Block-Jacobi for n=176/352/512** — the key next move: only a block-pair need be
  SRAM-resident, so the resident-tile limit stops binding. This is what unlocks the
  central `640×512` benchmark cases.
- **Multiple matrices per Triton program** for tiny `n` (e.g. n=32, batch 20): one
  program/matrix gives poor occupancy (~20 programs on 148 SMs). Pack several small
  matrices per program, or a 3D grid, to raise occupancy.
- **Early-exit on off-diagonal norm** per matrix to cut sweeps on easy inputs (adds
  divergence — measure before adopting).
- **Diagonal fast path** — a genuinely diagonal input (e.g. `lapack_diag_*`,
  `diagonal` case) has a trivial decomposition: just sort. Legit (property of input,
  not a shape lookup). Worth a cheap detector.

## Dead ends (with reason)

- **v3 — Householder pipeline with the inner solve deferred to `torch.eigh(T_dense)`.**
  torch's `eigh` re-tridiagonalizes the tridiagonal we hand it, so we pay for the
  reduction twice → strictly more work than v0 (~6× slower on n≥1024). Do not repeat
  a pipeline whose inner solve calls dense `eigh`; the inner solve must be native
  and consume the tridiagonal directly. D&C (v4) is the better inner solver.
- **v2 — block-Jacobi at n=512.** Matmul-form Jacobi is O(sweeps·n⁴); at n=512×640
  it was killed at ~360 s. Correct, but not viable at n=512 without drastically
  cutting sweep cost (TF32, cheaper local EVD, aggressive early-exit). Not for the
  large 640×512 cases. (Untested at n=176/352 — not a dead end there yet.)
