# v6 — improved-coding reimplementation of v4's D&C pipeline

**Parent:** v4 (`attempts/submission_v4.py`).
**Theory:** unchanged — Householder → tridiagonal → **Cuppen divide-and-conquer** →
backtransform. This is exactly what cuSOLVER `syevd` does; the algorithm is sound.
**What v6 changes:** the *implementation only*. We attack the two serial O(n)
Python column-loops that dominate v4, turn the wasted BLAS-1/2 memory traffic into
tensor-core BLAS-3, and let the loose gates buy us TF32 throughput.

**Goal:** close the ~4.5× gap to torch on n≥1024 and prove v4's loss was
implementation, not algorithm. Realistic aim = parity; best shot at a win is the
small-batch n=2048 case (#5, batch 8) and n=1024 (#4, batch 60).

**Constraints respected:** stays at **2 dispatch paths** (pipeline for n≥1024, torch
otherwise — TF32 and autotuning add *no* new path), ≤3000 lines, self-contained, no
streams.

---

## Diagnosis recap (given — do not re-derive)

- v4 lost ~4.5× to torch on n≥1024 (best Triton attempt). Verdict: IMPLEMENTATION.
- Prime causes:
  - **(a) `_build_Q1` unblocked `sorgtr` loop** (`for k in range(n-2,-1,-1)`): each of
    the n iterations does a rank-1 update that reads+writes the *entire* (b,n,n) Q1.
    For n=1024,b=60 that is ~0.25 GB/update × 1024 updates ≈ hundreds of GB of
    BLAS-1/2 traffic. **This is the single biggest waste.**
  - **(a′) the reduction `_householder_tridiagonalize`** per-column loop: n batched
    torch einsums (BLAS-2 symmetric matvecs). Half of these flops are intrinsically
    BLAS-2 (that is what `slatrd` is) — but the launch overhead and the *trailing*
    piece can be improved.
  - **(b) the lone Triton `_bmm`** is untuned fp32: `input_precision="ieee"`, fixed
    64/64/32 tiles, no TF32, no autotune.
- **Do NOT "fix" non-problems:** v4 has **no** redundant re-tridiagonalization (its
  D&C consumes (d,e) directly); the **D&C merge is already well-batched** (vectorized
  secular bisection, lockstep fixed tree, torch.bmm merges, O(log n) levels) and is
  **not** the bottleneck. Batching is correct — every loop is over columns / merge
  nodes, never over the batch.

---

## Ordered list of coding changes

Each item: **what to change · rationale · expected effect · risk.**

---

### Fix 1 — Blocked WY back-application of reflectors (replaces `_build_Q1` sorgtr loop). HIGHEST IMPACT.

**What.** Delete the unblocked `_build_Q1` rank-1 loop. Assemble the reduction
transform in **WY (compact-WY) block form** instead.

The stored reflectors `v_0 … v_{n-2}` live in the lower triangle of `A` (as in v4,
`v_k` in `A[:, k+1:, k]`, `v_k[0]=1`). Group them into panels of width `nb`
(reuse `_NB_REDUCE`, i.e. 32). For one panel of reflectors the compact-WY identity is

```
H_i H_{i+1} … H_{i+nb-1} = I − V T V^T
```

where `V` is `(b, n, nb)` (the panel's reflector columns, zeros above their rows) and
`T` is `(b, nb, nb)` **upper-triangular** — the "T factor" built from `tau` and
`V^T V` (LAPACK `slarft`). Build `T` per panel with the standard triangular
recurrence (small `nb×nb` — do it in torch; it is cheap and not on the critical path).

**Two options for producing the final `Q = Q1 @ Y` — pick per the risk note:**

- **Option A (as the task frames it): form Q1 explicitly, blocked.**
  Start `Q1 = I`, apply panels right-to-left:
  `Q1 ← (I − V T V^T) Q1 = Q1 − V (T (V^T Q1))`.
  Each panel application = **3 batched GEMMs** through the tuned Triton `_bmm`:
  `S = V^T @ Q1` (nb×n), `S ← T @ S` (nb×n), `Q1 ← Q1 − V @ S` (n×n).
  Then the existing `Q = _bmm(Q1, Y)` backtransform GEMM stays.

- **Option B (recommended stretch — LAPACK `sormtr`): never form Q1; apply the
  panels directly to `Y`.**
  `Q = H_0 … H_{n-2} Y`, applied panel-by-panel to the `(b,n,n)` `Y`:
  `Y ← Y − V (T (V^T Y))`. Same 3-GEMM panel primitive, but this **also removes the
  separate `Q1 @ Y` GEMM** — it fuses assembly + backtransform, saving ~one full
  n³ GEMM per matrix and never materializing Q1.

**Rationale.** v4 replaces n rank-1 updates (each an O(b·n²) memory sweep of Q1 →
O(b·n³) *traffic*, on BLAS-1/2 units) with n/nb=~32 panel updates whose work is
BLAS-3 on tensor cores. The arithmetic is the same ~(4/3)n³·b flops but the memory
traffic collapses by ~nb× and moves onto the fast path. The diagnosis names this the
single biggest lever.

**Expected effect.** Q1 assembly goes from the dominant term (hundreds of GB of
BLAS-2 traffic; tens of ms) to a compute-bound handful of tensor-core GEMMs
(sub-ms compute). This alone should remove the bulk of the 4.5× gap. Option B saves
an additional full backtransform GEMM.

**Risk.** Correctness of the T-factor recurrence and the right-to-left panel order
(off-by-one in reflector indexing / the `v_k[0]=1` implicit-one convention). The
fp32 self-check (Fix 6) catches a wrong Q. Start with Option A (closer to v4's
mental model, easier to validate), then switch to Option B once A passes. Medium
implementation risk, very high payoff.

---

### Fix 2 — Leaner `slatrd` reduction panel (keep BLAS-3 trailing update; cut BLAS-2 launch overhead).

**What.** Keep v4's right-looking blocked structure and its Triton BLAS-3 trailing
rank-2k update `A[trail] −= V W^T + W V^T` — that part is already correct and already
tensor-core. Improve the *panel* (`slatrd`) work:

1. **Do not pretend the per-column matvec becomes BLAS-3.** The symmetric matvec
   `w = tau · Atrail @ v` and the within-panel `V^T v` / `W^T v` corrections are
   intrinsically **BLAS-2** — this is the defining half of one-stage tridiagonalization
   and cannot be blocked away without switching to a *two-stage* (dense→band→tridiag)
   algorithm, which changes the theory and is explicitly out of scope. Keep it BLAS-2.
2. **Cut Python/launch overhead:** consolidate the several small `torch.einsum`
   calls per column into as few batched calls as possible; keep `_NB_REDUCE` a tunable
   and sweep it (wider panel = fewer trailing-update launches but more in-panel BLAS-2
   correction — 32 is the usual knee, try 16/32/48/64 on the remote runner).
3. **Route the in-panel rank-updates that are large enough through the tuned Triton
   `_bmm` with TF32** where the operand is a matrix (the trailing update already is;
   the `Vp (Wp^T v)` corrections are matvecs and stay torch — tensor cores do not
   help matvecs).

**Rationale.** The trailing update is the only genuinely BLAS-3 part of the reduction
and it is already Triton; the rest is a memory-bound floor. The honest win here is
launch-overhead reduction + TF32 on the one BLAS-3 piece, not a fake BLAS-2→BLAS-3
conversion.

**Expected effect.** Modest: trims launch overhead and speeds the trailing update via
TF32/autotune. The reduction remains a memory-bound floor (see Biggest Risk).

**Risk.** Low. TF32 on the trailing update injects error, but the update is
re-symmetrized (`U + U^T`) and the self-check gates the final result.

---

### Fix 3 — TF32 tensor cores under the loose gates.

**What.** Replace the hard-wired `input_precision="ieee"` in the GEMM with a
per-call **precision knob**. Default the throughput-heavy GEMMs to TF32 and the
accuracy-sensitive final backtransform to **`tf32x3`** (3-pass error-corrected TF32,
near-fp32 accuracy at ~3× tf32 cost, still on tensor cores — far faster than `ieee`
fp32 emulation):

- reduction trailing update → `tf32` (dampened, re-symmetrized).
- Q1/backtransform panel GEMMs → start `tf32x3`; drop to `tf32` if the self-check
  confirms it still passes.

**Rationale.** Gates are loose: orth `100·n·eps`, eigen `200·n·eps` (eps_fp32≈1.19e-7).
For n=1024 the orth budget is ≈0.012; plain TF32 (~10-bit mantissa) error over a
length-n contraction is close to that budget for the *full* backtransform, so use
`tf32x3` there and plain `tf32` only where it verifiably fits. The self-check is the
backstop.

**Expected effect.** Large per-GEMM speedup vs `ieee` fp32 (which on tensor cores is
emulated/slow). This is what makes the tensor-core steps actually cheap.

**Risk.** Plain TF32 on the full backtransform may trip the orth gate on large n →
self-check falls back to torch (no speedup, not wrong). Mitigated by `tf32x3`
default on the backtransform and the precision knob.

---

### Fix 4 — Autotuned GEMM tiling for `_bmm`.

**What.** Wrap `_bmm_kernel` in `@triton.autotune` over `BLOCK_M/BLOCK_N/BLOCK_K`,
`num_warps`, `num_stages`, keyed on bucketed `(M,N,K)` (and the batch count). Provide
a config set spanning the shapes the kernel actually issues: the tall-skinny panel
GEMMs (M≈n, N≈n, K=nb) from Fixes 1–2 and the large square backtransform
(M=N=K=n). Include 128×128×32, 128×256×32/64, 64×64×32, varying warps/stages.

**Rationale.** v4 uses one fixed 64/64/32 config for *both* a skinny panel GEMM and a
large square GEMM — near-optimal for neither, and no pipelining (`num_stages`) or warp
tuning. On B200, larger tiles + deeper software pipelining are a big win for the square
backtransform.

**Expected effect.** 1.5–3× on the GEMM-bound steps vs the untuned fixed tile;
compounds with TF32.

**Risk.** Low. Autotune caching cost is one-time per shape bucket; keep the config
list small (feasibility + compile time). Ensure masking still correct for M/N/K not
divisible by the tile (v4's masks already handle this).

---

### Fix 5 — Keep the D&C; wire it to the new outputs (verify interface, no rewrite).

**What.** Leave `_divide_and_conquer` / `_secular_solve` / `_sign_normalize`
unchanged. Only confirm they still receive `(d, e, tau)` from the reworked reduction
and that `Y` is fed to the new WY backtransform (Fix 1). Do **not** touch the
secular bisection, deflation, or the fixed lockstep merge tree.

**Rationale.** The diagnosis is explicit: the D&C is well-batched and is not the
bottleneck; there is no redundant re-tridiagonalization to remove. Rewriting it is
pure risk with no expected gain.

**Expected effect.** None (correctly). Avoids regressing a working component.

**Risk.** Minimal — an interface mismatch only. Its internal `torch.bmm` merges are
small (2L×2L, well-batched); not worth moving to Triton.

---

### Fix 6 — Cheaper correctness safety net (probe-based fp32 residual) + torch fallback.

**What.** Keep the try/except → `torch.linalg.eigh` fallback and the fp32 self-check
gate, but make the self-check genuinely cheap. v4's `_self_check` currently does two
**full O(b·n³) bmm**s (`Q^T Q` and `A @ Q`) every call — non-trivial relative to the
solve. Replace the eigen-equation check with a **matrix-free random-probe residual**:

- draw a few (`k`≈2–4) random unit vectors `x` (b,n,k);
- compute `r = A@(Q@x) − Q@(L[:,:,None] * (Q^T@x))` via matvec-shaped bmms →
  **O(b·n²·k)** instead of O(b·n³); compare `||r||` to `200·n·eps·||A||` with margin.
- keep the orthogonality check but likewise probe: `||Q^T(Q x) − x||` (O(b·n²·k)),
  or keep the full `Q^TQ−I` only if profiling shows it is cheap enough.
- keep the `isfinite` guard.

**Rationale.** The safety net must cost ≪ the pipeline or it eats the speedup. A
Hutchinson-style probe gives a reliable norm estimate at O(n²) not O(n³), and never
uses per-call fp64 matmuls (the task's explicit constraint).

**Expected effect.** Turns the self-check from an O(n³) tax into an O(n²) rounding
error, preserving the win while keeping the "never returns a wrong answer" guarantee.

**Risk.** A probe can under-estimate a residual concentrated in a direction the
probes miss. Use ≥2 probes and a conservative margin; on clustered/repeated spectra
(where v4's light-deflation D&C loses orthogonality) the probe will still flag most
failures, and any miss is bounded by the loose gate. Acceptable given the fallback.

---

## Highest-impact change

**Fix 1 — the blocked WY back-application replacing the unblocked `sorgtr`
`_build_Q1` loop.** It converts the pipeline's single largest cost (hundreds of GB of
BLAS-1/2 traffic re-sweeping the whole (b,n,n) Q1 n times) into a handful of
tensor-core BLAS-3 GEMMs; Option B (`sormtr`, apply reflectors straight to Y)
additionally deletes one full n³ backtransform GEMM.

## Biggest risk

After Q1 is fixed, the **intrinsically BLAS-2, serial `slatrd` reduction becomes the
new floor**: ~n memory-bound symmetric-matvec launches (half the reduction flops) that
cannot become BLAS-3 without a two-stage algorithm (out of scope, changes the theory).
cuSOLVER's `syevd` is heavily fused here, so v6 may land at *parity* rather than a
clear win. Secondary risk: plain TF32 tripping the orthogonality gate on large n,
which the `tf32x3` default + fp32 self-check turn into a safe fallback (no speedup)
rather than a wrong answer.

---

## Implementation notes (what was actually done vs planned)

Written to `attempts/submission_v6.py` (774 lines, ≤3000; 2 dispatch paths; no
stream tokens; `ast.parse` clean). Cannot run locally (no GPU) — all reasoning.

**Fully implemented as planned:**
- **Fix 1 (blocked WY backtransform) — Option B, the recommended stretch.** New
  `_backtransform_wy` + `_wy_tfactor`. The unblocked `_build_Q1` sorgtr loop is
  deleted entirely; reflectors are grouped into `_NB_REDUCE`-wide panels and the
  compact-WY form `I − V T V^T` is applied **directly to Y** right-to-left
  (`Y ← Y − V(T(V^T Y))`), so Q1 is never materialized and the separate `Q1@Y`
  GEMM is fused away. T factors built with the forward/columnwise `slarft`
  recurrence in torch (small w×w, off critical path). V is reconstructed from A's
  lower triangle with a support mask (v_{p+i} occupies rows ≥ p+i+1, unit at
  p+i+1) — the identity is stagger-agnostic, so this is valid.
- **Fix 3 (TF32):** precision knob on `_bmm` (`ieee`/`tf32`/`tf32x3` via constexpr
  branch). Reduction trailing update → `tf32` (re-symmetrized); backtransform panel
  GEMMs → `tf32x3` (near-fp32).
- **Fix 4 (autotune):** `@triton.autotune` over 9 BLOCK_M/N/K·warps·stages configs,
  keyed on **pow2-bucketed (mk,nk,kk)** (extra unused kernel args) so per-panel
  trailing sizes don't re-autotune every panel — a deliberate fix to a real churn
  risk the naive "key on M,N,K" would cause.
- **Fix 5 (D&C):** `_divide_and_conquer`/`_secular_solve`/`_sign_normalize` copied
  verbatim from v4; only rewired (d,e→solve, tau+Y→new backtransform).
- **Fix 6 (cheap self-check):** replaced v4's two full O(n³) bmms with a
  k=4 random-unit-probe residual, O(b·n²·k): eigen `A(Qx)−Q(L⊙x)` and orth
  `Q^T(Qx)−x`, probe norm upscaled by √n as a conservative operator-norm estimate,
  compared to the grader's 200·n·eps·‖A‖₁ / 100·n·eps gates. No fp64. Kept
  try/except→torch fallback.

**Simplified / deferred (with reason):**
- **Fix 1 Option A skipped** — went straight to Option B. Both share the identical
  panel primitive and correctness risk; B is less code and saves a GEMM. If B is
  wrong the self-check catches it and falls back (correctness safe), but a bug
  would mean *no speedup on the pipeline path* until fixed on the runner.
- **Fix 2 (leaner slatrd panel):** kept v4's panel loop **as-is** (only trailing
  update precision→tf32). The plan's "consolidate einsums / sweep NB" is a modest
  launch-overhead trim; deferred to avoid risking the working BLAS-2 math blind. NB
  stays 32 (tunable). The intrinsically-BLAS-2 reduction floor is untouched, as the
  plan says it must be.
- **`tf32x3` not dropped to `tf32`** on the backtransform — the plan says start at
  tf32x3 and drop only if the self-check confirms; without a runner I can't confirm,
  so I keep the safe tf32x3 default.

**Correctness/perf risks:**
- Cannot verify locally. Main correctness risk = a sign/index/order bug in the WY
  backtransform (T recurrence, panel right-to-left order, V support mask) →
  self-check should catch and fall back to torch (no wrong answer, but no speedup).
- `tf32x3` support depends on the runner's Triton/B200 backend; if unsupported the
  GEMM errors → try/except → torch fallback (safe, no speedup).
- Autotune benchmarks configs on first call per bucket; one-time warmup cost.
- Probe self-check could in principle under-estimate a residual concentrated off
  the probe directions; mitigated by k=4 probes + √n upscale + the very large
  separation (correct ≈ roundoff, broken ≈ O(1)) and the loose gate.
- **Biggest expected perf ceiling (per plan):** once Q1 assembly is fixed, the
  serial BLAS-2 `slatrd` reduction becomes the floor vs cuSOLVER's fused syevd →
  v6 may land at parity rather than a clear win.
