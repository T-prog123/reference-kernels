# v5 — improved-coding reimplementation of v3's QR pipeline

**Parent:** v3 (`attempts/submission_v3.py`).
**Theory:** unchanged — Householder → tridiagonal → **shifted QR** (implicit-shift
QL/QR on the tridiagonal) → backtransform. Same math as v3's stated method; we fix
the *implementation*, and in particular we make the inner solve be the QR that v3
only *named* (v3 actually deferred it to `torch.linalg.eigh(T_dense)`).
**What v5 changes:** (1) replace the redundant dense `eigh(T)` with a **native**
batched implicit-shift QR that consumes `(d,e)` directly; (2) cut the serial dlatrd
reduction's launch overhead; (3) fuse Q assembly + backtransform; (4) TF32 tensor
cores under the loose gates; (5) faster dense GEMMs (cuBLAS or autotuned Triton);
(6) a cheap fp32 self-check instead of per-call fp64 matmuls.

**Goal:** close v3's ~6× gap to torch on n≥1024 and prove the loss was
implementation, not the pipeline. **Honest ceiling (state up front):** even a clean
QR inner solve accumulates eigenvectors by O(n³) *memory-bound* Givens sweeps, which
is exactly why v4's D&C beat v3's QR by 38%. So v5's realistic aim is **parity with,
or a modest gap behind, v4/v6 (D&C)** — not a torch win. Its purpose is to establish
the honest cost of a *well-implemented* QR pipeline. If it still trails D&C, that
confirms the design.md verdict (D&C > QR for all-vectors) and v6 (native D&C) is the
path forward.

**Constraints respected:** stays at **2 dispatch paths** (pipeline for
n ≥ `_N_PIPELINE_MIN`=1024, torch otherwise; TF32/autotune/cuBLAS add *no* path),
≤3000 lines, self-contained, no streams.

---

## Diagnosis recap (given — do not re-derive)

v3 lost ~6× to torch on n≥1024. Verdict: BOTH, dominated by IMPLEMENTATION; the
design is the sound cuSOLVER pipeline. Prime causes:

- **(a) inner solve deferred to `torch.linalg.eigh(T_dense)`** — cuSOLVER does not
  special-case a tridiagonal input, so it re-tridiagonalizes → ~1× of the 6× is pure
  redundant work (we tridiagonalize, then torch does it again inside eigh).
- **(b) serial per-column dlatrd reduction** launching thousands of tiny BLAS-2
  kernels — **the biggest lever**, latency-bound because the batch is small (60 at
  n=1024, 8 at n=2048) so every tiny launch underutilizes the GPU.
- **(c) fp64 self-check on every call** — `_passes_gates` does 3–4 full n×n **fp64**
  matmuls per call; fp64 is slow on B200 and this can rival the solve cost itself.
- **(d) hand-rolled, un-autotuned `_bmm`** — fixed 64×64×32 tiles, `input_precision=
  "ieee"` (no TF32; on B200 this is emulated fp32, i.e. slow tensor-core use).

**Do NOT "fix" non-problems.** Batching is already correct — every loop is over
columns/reflectors, never over the batch. Do not "add batching". The `_householder`
reflector generation is batched and fine. The trailing rank-2k update is already
BLAS-3 on Triton — keep it.

---

## Ordered list of coding changes

Recommended **execution order = low-risk quick wins first, the risky core last**, so
the pipeline is fast and self-checking *before* the un-testable QR kernel goes in:
Fix 6 → Fix 4 → Fix 5 → Fix 3 → Fix 2 → **Fix 1**. Each item below:
**what · rationale · expected effect · risk.**

---

### Fix 1 — Native implicit-shift QL/QR inner solve consuming `(d,e)` (replaces `eigh(T_dense)`). HIGHEST IMPACT **and** BIGGEST RISK.

**What.** Delete the `torch.diag_embed`→`torch.linalg.eigh(T)` block. Solve the
tridiagonal eigenproblem directly from the `(d,e)` vectors we already produced, in a
**single batched Triton kernel, one program per matrix**. Algorithm = LAPACK
`ssteqr`/`tql2`: implicit-shift QL iteration with **Wilkinson shift**, deflating on
negligible off-diagonals, accumulating eigenvectors into `Z`.

Structure the kernel to separate the two very different sub-problems:

1. **Serial scalar part (cheap, SRAM-resident).** For n ≤ 2048, `d` and `e` are
   ≤ 8 KB each — hold both resident per program. Run the QL sweeps: at each active
   sub-block `[l..m]`, deflate where `|e_i| ≤ eps·(|d_i|+|d_{i+1}|)` (LAPACK test),
   pick the Wilkinson shift from the trailing 2×2, and bulge-chase with Givens
   rotations updating `d,e` in place. Data-dependent iteration count and deflation
   are handled **per program**, so batch divergence is just load imbalance across
   SMs — not wasted lockstep work.
2. **Parallel eigenvector part (the expensive, memory-bound part).** Do **not**
   apply each Givens rotation to `Z` inline (that is per-rotation BLAS-1, the slowest
   possible pattern). Instead, during a sweep **store the sequence of rotations**
   `(c_k, s_k)` (there are ≤ n−1 per sweep) into a scratch buffer, then apply the
   whole sweep to `Z` with a **vectorized `dlasr`-style pass** (rows of `Z`
   parallelized across threads; each thread rotates its row through the stored
   `(c,s)` pairs). `Z` (n×n, up to 16 MB) lives in **global memory** — it cannot be
   resident — but the `dlasr` pass is coalesced and touches each `Z` entry O(1) times
   per sweep. This is the standard high-performance way to accumulate QR
   eigenvectors and is the difference between "tolerable" and "catastrophic".

Return `Z` (eigenvectors of T) and `L=d` (eigenvalues). Sort ascending + permute
columns of `Z` in torch (cheap, robust — exactly as v1 does).

**Rationale.** Removes cause (a) entirely: no second tridiagonalization. The inner
problem is solved in the representation we already have.

**Expected effect.** Eliminates the ~1× redundant-reduction cost. *But* the Z
accumulation is intrinsically O(n³) and memory-bound (the `dlasr` sweeps), so the
inner solve does **not** become cheap — it becomes "as cheap as QR-for-eigenvectors
can be", which the v3→v4 result says is still behind D&C. Net: a real improvement
over v3, not necessarily over v4.

**Risk — HIGH, and the biggest in this plan.** (i) The kernel is a sequential,
branch-heavy transcription (shift formula, deflation index, QL-vs-QR direction, the
`v[0]=1` / rotation-sign conventions) written **with no local GPU to test on** — the
exact hazard v3's header called out. A silent indexing/shift bug yields *finite but
wrong* numbers, which the try/except would not catch. (ii) A convergence bug could
make *every* case silently fall back to torch (no crash, just slow) — watch for a
geomean that looks like pure-torch. **Mitigations:** the fp32 probe self-check
(Fix 6) + torch fallback bound any wrong result to a slow-but-correct answer; add an
iteration cap per sub-block (e.g. 30·n like LAPACK) that raises → fallback rather
than hanging; validate first on the smallest engaged case (n=1024) before trusting
n=2048. (iii) Occupancy: the scalar QL part is one program per matrix (8–60
programs on 148 SMs — poor occupancy) but it is the *cheap* part; the `dlasr`
eigenvector pass is where the FLOPs/traffic are and it parallelizes across rows, so
overall SM utilization is driven by the parallel part. Keep the scalar and parallel
phases in separate kernels (or separate program dimensions) so the eigenvector work
is not throttled by the serial loop's occupancy.

---

### Fix 2 — Leaner dlatrd reduction: cut BLAS-2 launch overhead (keep the BLAS-3 trailing update).

**What.** Keep v3's right-looking blocked structure and its Triton BLAS-3 trailing
rank-2k update `A[trail] −= V W^T + W V^T` (that part is correct and already
tensor-core). Attack cause (b) — the ~O(n·nb) tiny torch launches inside the panel:

1. **Be honest about what stays BLAS-2.** The symmetric matvec `Av = A22 @ v` and the
   within-panel `V^T v` / `W^T v` corrections are intrinsically BLAS-2 — this is the
   defining half of one-stage tridiagonalization and cannot be blocked into BLAS-3
   without a *two-stage* (dense→band→tridiag) algorithm, which changes the theory and
   is out of scope. Do not fake a BLAS-2→BLAS-3 conversion here.
2. **Collapse launches.** Consolidate the several small `torch.matmul(...).squeeze()`
   correction calls per column into as few batched calls as possible (`torch.baddbmm`
   / fused einsum). Every removed per-column launch is pure latency saved, and at
   small batch the panel loop is latency-bound.
3. **Optional, higher-payoff/higher-risk: fuse the whole panel into one Triton
   kernel.** A single kernel per panel that keeps `V`,`W`, and the running column
   resident and loops the `nb` columns internally (streaming `A22` from global for
   the matvec) collapses ~O(n·nb) launches into ~O(n/nb) kernel launches. This is the
   real launch-overhead win, but it is an un-tested fused reduction kernel (same
   class of risk as Fix 1). **Recommend deferring this** until the torch-launch
   consolidation (step 2) is measured; if the reduction is still a big fraction after
   Fixes 1/3/5, then attempt the fused panel behind the self-check.
4. **Sweep `_PANEL`** (16/32/48/64) on the remote runner — wider panel = fewer
   trailing-update launches but more in-panel BLAS-2 correction; 32 is the usual knee.
5. Route the trailing update GEMMs through the tuned/TF32 path (Fixes 3–5).

**Rationale.** The trailing update is the only genuine BLAS-3 in the reduction and is
already Triton; the rest is a memory-bound floor. The honest win is launch-overhead
reduction + TF32 on that one BLAS-3 piece.

**Expected effect.** Step 2: meaningful trim of the latency-bound panel overhead
(part of cause (b)). Step 3 (if taken): large additional cut. The reduction remains a
memory-bound floor regardless (see Biggest Risk).

**Risk.** Step 2: low (pure refactor of existing torch calls; self-check gates it).
Step 3: high (un-tested fused kernel) — hence deferred and gated.

---

### Fix 3 — Fused, blocked backtransform (replace `_form_U` + `_bmm(U,Y)`).

**What.** v3 forms `U` explicitly via `torch.linalg.householder_product` (orgqr) —
an O(n³) step that also materializes a full n×n `U` — and *then* does a second O(n³)
GEMM `Q = U @ Y`. Fuse these into one applied product `Q = H_0 … H_{n-3} Y`:

- **Primary (low risk): `torch.ormqr`.** `Q = torch.ormqr(reflectors, tau, Y,
  left=True, transpose=False)` applies the stored Householder product directly to `Y`
  through cuSOLVER — a **tested** primitive, BLAS-3, TF32-capable, no `U`
  materialization, and it removes the separate `U@Y` GEMM (~one full n³ GEMM saved
  per matrix). Verify `torch.ormqr` accepts our reflector layout and batches (if it
  loops over the small batch internally, that is still fine). This is the same
  reflectors/`tau` we already store for orgqr, just applied instead of accumulated.
- **Fallback (higher risk): blocked compact-WY `sormtr`.** If `ormqr` is unavailable
  or won't batch: group reflectors into panels of width `nb`, build the `nb×nb`
  upper-triangular WY `T`-factor per panel (`slarft` recurrence, small — do in
  torch), and apply right-to-left `Y ← Y − V (T (V^T Y))` via 3 batched GEMMs per
  panel through the tuned Triton `_bmm`. This is what v6 plans for its backtransform;
  reuse that machinery if it lands. Hand-rolled WY is the single most likely place
  for an un-testable off-by-one (v3's header flagged this) — prefer `ormqr`.

**Rationale.** v3 pays O(n³) twice (form U, then U@Y) plus an n×n materialization.
Applying the reflectors directly halves the backtransform work and drops the
materialization.

**Expected effect.** ~2× on the backtransform stage; removes an n×n allocation.

**Risk.** Primary: low (tested cuSOLVER primitive). Fallback: medium (WY indexing) —
gated by the self-check.

---

### Fix 4 — TF32 tensor cores under the loose gates.

**What.** Replace the hard-wired `input_precision="ieee"` with a per-call precision
knob. Gates are loose — orth `100·n·eps`, eigen `200·n·eps`, recon `400·n·eps`,
eps_fp32 ≈ 1.19e-7 (confirmed in `problems/linalg/eigh_py/reference.py`:
`_EIGEN_RTOL_FACTOR=200`, `_RECON_RTOL_FACTOR=400`, `_ORTH_RTOL_FACTOR=100`). For
n=1024 the orth budget is ≈0.012.

- reduction trailing update → `tf32` (dampened; the update is re-symmetrized).
- backtransform GEMMs → start with **`tf32x3`** (3-pass error-corrected TF32,
  near-fp32 accuracy at ~3× tf32 cost, still tensor-core and far faster than `ieee`
  fp32 emulation); drop to plain `tf32` if the self-check confirms it still passes.
- If the dense GEMMs go through cuBLAS (Fix 5), set
  `torch.backends.cuda.matmul.allow_tf32 = True` (and the cuDNN flag) to get the same
  TF32 benefit from torch.

**Rationale.** `ieee` fp32 on B200 tensor cores is emulated and slow; the gates have
ample room for TF32 on all but the full-length backtransform contraction, where
`tf32x3` keeps accuracy.

**Expected effect.** Large per-GEMM speedup vs `ieee`. This is what makes the
tensor-core steps actually cheap.

**Risk.** Plain TF32 on the full backtransform may trip the orth gate at large n →
self-check falls back to torch (slow, not wrong). Mitigated by the `tf32x3` default
there + the precision knob.

---

### Fix 5 — Faster dense GEMMs: prefer cuBLAS, else autotune the Triton `_bmm`.

**What.** v3's `_bmm` uses one fixed 64×64×32 config for both the skinny panel GEMM
(K=nb) and the large square backtransform (M=N=K=n) — near-optimal for neither, no
software pipelining (`num_stages`), no warp tuning.

- **Primary (recommended): route the dense GEMMs through `torch.matmul`/`baddbmm`
  (cuBLAS) with TF32 enabled.** cuBLAS is the fastest tuned path and is tested; the
  competition permits torch for the dense steps. With Fix 3 using `ormqr` and the
  trailing update using `baddbmm`, the *only* thing that must be Triton is the native
  QR inner solve (Fix 1) — which is precisely the part torch cannot do element-wise
  efficiently. That is a legitimate, strong division of labor and the best shot at
  closing the gap.
- **Alternative (Triton showcase): keep `_bmm` but wrap it in `@triton.autotune`**
  over `BLOCK_M/N/K`, `num_warps`, `num_stages`, keyed on bucketed `(M,N,K)`. Provide
  configs for both the tall-skinny panel GEMM and the large square GEMM (e.g.
  128×128×32, 128×256×{32,64}, 64×64×32, varying warps/stages). Keep the tail masks
  (v3's already handle non-divisible M/N/K).

**Rationale.** A hand-tuned Triton GEMM rarely beats cuBLAS on square fp32/TF32
GEMMs; ranking is pure runtime, so use the fastest correct path. Autotune is the
fallback if we want to keep the GEMMs in Triton.

**Expected effect.** cuBLAS+TF32: near-peak dense GEMM throughput. Autotuned Triton:
1.5–3× over the fixed tile. Either compounds with Fix 4.

**Risk.** Low. cuBLAS is tested. Autotune's one-time per-bucket compile is absorbed
by benchmark warmup; keep the config list small (compile time + feasibility).

---

### Fix 6 — Cheap fp32 probe self-check (replace per-call fp64 matmuls) + torch fallback.

**What.** Keep the try/except → `torch.linalg.eigh` fallback, but make the self-check
cheap. v3's `_passes_gates` does 3–4 **full n×n fp64** matmuls every call (cause
(c)). Replace with a **matrix-free random-probe residual in fp32**:

- draw `k`≈2–4 random unit vectors `x` shaped (b,n,k);
- eigen residual: `r = A@(Q@x) − Q@(L[:,None,:] * (Q^T@x))` via matvec-shaped bmms →
  **O(b·n²·k)** instead of O(b·n³); compare `||r||` to `200·n·eps·||A||` with margin;
- orthogonality: probe `||Q^T(Q x) − x||` (O(b·n²·k)) rather than forming `Q^TQ−I`;
- keep the `isfinite` guard on `Q,L`;
- keep the ascending check (it is O(n), already cheap).

Use fp32 throughout the check (no fp64). Use ≥2 probes and a conservative margin.

**Rationale.** The safety net must cost ≪ the pipeline or it eats the win. A
Hutchinson-style probe estimates the residual norm at O(n²), never uses per-call
fp64, and still guarantees "never return a wrong answer" via the fallback.

**Expected effect.** Turns the self-check from an O(n³) fp64 tax (which could rival
the solve) into an O(n²) fp32 rounding cost — directly recovers cause (c).

**Risk.** A probe can under-estimate a residual concentrated in a direction the
probes miss — most dangerous on clustered/near-rank-deficient spectra (cases #9,#10)
where QR-accumulated eigenvectors can lose orthogonality. Mitigate with ≥2 probes + a
conservative margin; any miss is still bounded by the loose gate, and the fallback
covers detected failures. If profiling shows the full fp32 `Q^TQ−I` (O(n³) but fp32,
not fp64) is affordable, use it for the orth check specifically, since orthogonality
is where QR is most at risk.

---

## Highest-impact change

**Fix 1 — the native implicit-shift QR inner solve consuming `(d,e)`**, replacing
`torch.linalg.eigh(T_dense)`. It removes the redundant second tridiagonalization that
is the defining flaw of v3 (~1× of the 6×) and is what makes the pipeline "a real
QR pipeline" rather than "our reduction wrapped around torch's full eigensolver".
(Fix 6 — dropping the per-call fp64 self-check — is the highest *impact-per-effort*
quick win and should be done first, before the risky kernel work.)

## Biggest risk

**Fix 1's native QR kernel, written and shipped with no local GPU.** Two failure
modes: (i) a silent shift/indexing/convention bug that returns finite-but-wrong
eigenpairs (caught only by the probe self-check → slow fallback, not by try/except);
(ii) a convergence bug that silently routes every case to the torch fallback,
masquerading as "no speedup". Even if it is perfectly correct, its eigenvector
accumulation is O(n³) memory-bound `dlasr` sweeps — the very cost D&C avoids — so v5
may land **behind v4/v6 (D&C)** even done well. That is an acceptable, informative
outcome: it isolates and quantifies the honest cost of a well-implemented QR pipeline
and confirms whether D&C (v6) is the correct large-n direction.

---

## Implementation notes (what was actually done vs planned)

Written to `attempts/submission_v5.py` (676 lines, no stream tokens, 2 dispatch
paths). Authored blind — **no local GPU, nothing here was run**. Correctness rests
entirely on the two safety nets below.

### Fully implemented
- **Fix 1 — native implicit-shift QL inner solve (Triton), consuming (d,e).**
  Replaced `torch.diag_embed → torch.linalg.eigh(T)` with `_steqr_kernel`, a
  0-indexed transcription of EISPACK `tql2` / NR `tqli`: Wilkinson shift,
  deflation test `|e[m]| ≤ eps·(|d[m]|+|d[m+1]|)`, bulge chase updating (d,e) and
  rotating two Z columns per Givens. **One program per matrix**; (d,e) in a
  per-matrix global working buffer (scalar L1-hot loads — Triton can't dynamically
  index a register-resident vector, which forced this); **Z stored column-major**
  so each rotation is one coalesced n-wide vectorized read-modify-write of two
  columns. Iteration cap 30/eigenvalue sets a per-matrix `flag` → those matrices
  re-solved by torch `eigh(T)`.
- **Fix 3 — fused backtransform via `torch.ormqr`.** Removed `_form_U` (orgqr +
  n×n materialization) and the separate `U@Z` GEMM; applies reflectors straight to
  Z[1:]. cuBLAS `orgqr + matmul` fallback (no hand-rolled WY) if ormqr raises.
- **Fix 4 — TF32.** `torch.backends.cuda.matmul.allow_tf32=True`; Triton trailing
  update uses `input_precision="tf32x3"` (near-fp32, tensor-core).
- **Fix 5 — GEMM.** The one remaining Triton GEMM (trailing rank-2k update) is now
  `@triton.autotune`'d over tiles/warps/stages keyed on (M,N,K). Dense fallbacks
  use cuBLAS.
- **Fix 6 — cheap fp32 self-check (no fp64).** isfinite + ascending (exact) +
  random-probe eigen residual (O(n²k), 4× margin) + **full fp32** `QᵀQ−I` (orth is
  QR's weak spot → checked exactly, but fp32 not fp64). Directly removes cause (c).

### Simplified / deferred (and why)
- **Fix 2 step 3 (fused whole-panel reduction kernel): DEFERRED.** Only step 2
  landed — the two per-column BLAS-2 correction matmuls are collapsed into one via
  operand stacking (safe algebraic identity), trimming latency-bound launches. The
  fused panel kernel is the same un-testable-kernel risk as Fix 1; not worth
  stacking a second blind kernel. The trailing BLAS-3 update stays Triton.
- **Fix 1 occupancy variant (row-tiled grid): DEFERRED.** Plan wanted grid =
  batch × n/rowtile with each tile re-running the scalar QL and rotating only its
  rows, to spread the eigenvector work across SMs. Shipped the simpler
  **one-program-per-matrix** version instead (fewer moving parts to get wrong
  blind). Consequence: only `batch` CTAs (8 @ n=2048, 60 @ n=1024) → poor SM
  occupancy at small batch. The row-tiled version is the obvious next perf step.
- **Self-check: orth is full fp32, not a probe.** Deliberately stricter than the
  plan's "probe everything." A random probe can under-estimate a rank-1
  orthogonality loss by ~√(n/k) — precisely the clustered/near-defective spectra
  (cases #9/#10) where QR eigenvectors fail — so probing orth risks passing a
  wrong answer through the *reliable* grader. Full fp32 `QᵀQ−I` is one O(n³) fp32
  matmul (cheap vs v3's fp64) and reliable. Eigen stays a probe (its errors are
  usually spread out) with a conservative margin.

### Correctness / perf risk
- **HIGH correctness risk on Fix 1 (untested, branch-heavy).** A silent
  shift/index/convention bug yields finite-but-wrong eigenpairs. Bounded by: (a)
  per-matrix non-convergence flag → torch `eigh(T)` rescue; (b) final fp32
  self-check → whole-batch torch fallback; (c) a Triton compile failure raises
  inside the `try/except` → torch fallback. So the worst case is slow-but-correct,
  **but** if the kernel is silently always-wrong or always-non-converging, v5
  collapses to (roughly) torch timing — read a torch-like geomean as that signal.
- **Triton control-flow assumption.** The kernel uses nested `while` loops with
  data-dependent `break`/`continue`-style flags and scalar `tl.load`-driven `if`s.
  Requires a reasonably modern Triton (B200 runner); if unsupported it will fail
  to compile → fallback (correct, no speedup).
- **fp32 QL accuracy.** Runs the QL in fp32 (like LAPACK `ssteqr`); clustered
  n=2048 spectra may not reach 200·n·eps → self-check → fallback on those.
- **Perf.** Expect at best parity-to-modest-gap-behind D&C, per the plan: the
  bulge chase is a long serial per-matrix chain (intrinsic QR cost) and the
  one-CTA-per-matrix mapping under-fills the GPU at small batch. Validate n=1024
  before trusting n=2048.
