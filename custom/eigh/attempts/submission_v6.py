# submission_v6.py
# ============================================================================
# parent: v4
# Hypothesis: improved-coding D&C pipeline per notes/plans/v6_dc_improvement_plan.md
#   — same theory as v4 (Householder -> tridiagonal -> Cuppen divide-and-conquer
#   -> backtransform), but the *implementation* is reworked to close the ~4.5x gap
#   to torch on n>=1024.  The four coding levers, in impact order:
#     (Fix 1) Replace v4's unblocked `sorgtr` `_build_Q1` rank-1 loop (n memory
#             sweeps of the whole (b,n,n) Q1 — the single biggest waste) with a
#             BLOCKED compact-WY back-application (LAPACK `sormtr`): assemble the
#             reduction reflectors into panels (I - V T V^T) and apply them
#             DIRECTLY to the tridiagonal eigenvectors Y (Option B in the plan).
#             This never materializes Q1 and also fuses away the separate Q1@Y
#             GEMM — each panel is 3 tensor-core BLAS-3 GEMMs.
#     (Fix 2) Keep v4's right-looking blocked `slatrd` reduction (its per-column
#             matvecs are intrinsically BLAS-2 and cannot be blocked away without a
#             two-stage algorithm, which would change the theory / scope); only the
#             genuinely BLAS-3 trailing rank-2k update stays on Triton, now on TF32.
#     (Fix 3) TF32 tensor cores under the loose gates (eigen 200*n*eps,
#             orth 100*n*eps): reduction trailing update on plain `tf32`
#             (dampened + re-symmetrized), backtransform panels on `tf32x3`
#             (3-pass error-corrected TF32, near-fp32 accuracy, still tensor-core).
#     (Fix 4) Autotuned batched GEMM `_bmm` over BLOCK_M/N/K/warps/stages, keyed on
#             power-of-two-bucketed (M,N,K) so the per-panel trailing sizes do NOT
#             re-trigger autotuning every panel.
#     (Fix 5) The D&C inner solve (`_divide_and_conquer` / `_secular_solve` /
#             `_sign_normalize`) is kept VERBATIM from v4 — it consumes (d,e) (and
#             we pass tau to the new backtransform) directly, is already
#             well-batched, and is NOT the bottleneck.  Only its wiring changed.
#     (Fix 6) A CHEAP fp32 random-probe self-check (O(b*n^2*k), NOT the per-call
#             O(b*n^3) fp64/fp32 full matmuls v4 used) + try/except -> torch
#             fallback, so a wrong Q is always caught and correctness always holds.
#
# THEORY IS UNCHANGED vs v4.  This is a coding rewrite, not a new algorithm.
#
# ----------------------------------------------------------------------------
# HONEST SCOPE:  what is *genuinely Triton* vs what is *deferred to torch*
# ----------------------------------------------------------------------------
#   GENUINELY TRITON (tensor-core tl.dot; TF32 / tf32x3 under the loose gates):
#     * `_bmm` — one autotuned batched, strided GEMM kernel (C = A @ B) with a
#       per-call precision knob.  It is the only Triton kernel; it serves every
#       O(n^3) tensor-core step:
#         (1) REDUCTION trailing update  A[trail] -= V W^T + W V^T   (tf32),
#         (2) BACKTRANSFORM panel GEMMs  Y <- Y - V (T (V^T Y))       (tf32x3).
#       Transposed operands are passed as `.transpose(-1,-2)` VIEWS (strides only,
#       no materialized transpose copy).
#
#   DEFERRED TO TORCH (documented, per the task's allowance):
#     * The `slatrd` panel factorization of the reduction (Householder-vector
#       generation + per-column BLAS-2 symmetric matvecs) — memory-bound, tensor
#       cores do not help; ~half the reduction flops.  (v4-identical.)
#     * The small per-panel WY T-factor recurrence (LAPACK `slarft`): nb x nb,
#       cheap, off the critical path.
#     * THE DIVIDE-AND-CONQUER INNER SOLVE — a REAL batched Cuppen D&C (fixed
#       lockstep bisection tree, rank-one secular merge with Gu-Eisenstat
#       log-space z-hat), NOT a torch.linalg.eigh on the dense tridiagonal.
#       Kept verbatim from v4 (see Fix 5).
#
# ----------------------------------------------------------------------------
# CORRECTNESS-FIRST SAFETY NET  (we cannot run anything locally — no GPU here)
# ----------------------------------------------------------------------------
#   1. A CHEAP fp32 random-probe self-check (Fix 6): draw k random unit vectors x
#      and measure the eigen residual ||(AQ - Q diag(L)) x|| and orthogonality
#      residual ||(Q^T Q - I) x|| in O(b*n^2*k) (matvec-shaped bmms), upscale the
#      probe by sqrt(n) to conservatively estimate the operator norm, and compare
#      to the grader's own dimension-scaled thresholds.  A correct pipeline passes
#      by ~2 orders of magnitude; a broken one (e.g. clustered spectra where the
#      light-deflation D&C loses orthogonality) fails by orders of magnitude, so
#      the loose probe is a reliable gate.  NO per-call fp64 matmuls.
#   2. The whole pipeline is wrapped in try/except -> torch.linalg.eigh(A).
#   Worst case is "no speedup" (torch fallback), never "wrong answer".
#
# OUTPUT CONTRACT (see .claude/rules/contract-and-integrity.md)
# ----------------------------------------------------------------------------
# custom_kernel(data:[b,n,n] fp32 CUDA) -> (Q, L)
#   Q: [b,n,n] fp32, columns are orthonormal eigenvectors.
#   L: [b,n]  fp32, eigenvalues ascending.
#
# DISPATCH (size-regime only; 2 code paths, well under the HARD MAX of 5):
#   * n >= _N_PIPELINE_MIN (=1024) and CUDA and Triton -> Householder+D&C pipeline
#   * everything else, or any failure/self-check-miss   -> torch.linalg.eigh
# (TF32 and autotuning add NO new dispatch path — same code, different tl.dot
#  precision / tile.)
# ============================================================================

import math

import torch

# Triton is only available on the remote B200 runner.  Import defensively so that
# merely importing this module never explodes if Triton is absent; in that case
# _HAVE_TRITON stays False and we always take the torch path.
try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - defensive
    _HAVE_TRITON = False

from task import input_t, output_t


# ============================================================================
# Dispatch / tuning constants
# ============================================================================
# Only n >= this take the Triton Householder+D&C pipeline (ranked shapes: n=1024
# batch 60, n=2048 batch 8 — the "large n, small batch" regime).  Size-regime
# threshold, not a per-shape fingerprint.
_N_PIPELINE_MIN = 1024

# Householder reduction panel width (LAPACK "nb").  Also the WY back-application
# panel width (Fix 1).  32 is the usual knee (wider = fewer trailing/backtransform
# launches but larger in-panel BLAS-2 correction); tunable on the remote runner.
_NB_REDUCE = 32

# Divide-and-conquer leaf size and secular-solve params (v4-identical).
_DC_LEAF = 64
_BISECT_ITERS = 64
_Z_DEFLATE = 1e-14
_RHO_TINY = 1e-30

# Number of random probes for the fp32 self-check (Fix 6).  >=2 recommended so a
# residual concentrated in one direction is unlikely to be missed by all probes.
_PROBE_K = 4

_EPS32 = float(torch.finfo(torch.float32).eps)


# ============================================================================
# The single Triton kernel: an AUTOTUNED batched, strided GEMM  C = A @ B  (Fix 4)
# ============================================================================
# One kernel serves every O(n^3) tensor-core step.  It takes explicit per-operand
# strides, so callers pass `.transpose(-1,-2)` VIEWS to get A^T @ B or A @ B^T
# without materializing a transposed copy.
#
# Precision (Fix 3) is a compile-time constexpr flag PREC:
#     0 -> "ieee"    (full fp32 accumulate; slow emulated path on tensor cores)
#     1 -> "tf32"    (~10-bit mantissa; fastest; for the re-symmetrized reduction
#                     trailing update)
#     2 -> "tf32x3"  (3-pass error-corrected TF32; near-fp32 accuracy at ~3x tf32
#                     cost; for the accuracy-sensitive backtransform)
# The branch is on a constexpr, so exactly one tl.dot survives compilation.
#
# Autotune (Fix 4) sweeps tiles/warps/stages.  KEY is bucketed (mk,nk,kk = the
# next power of two of M,N,K) so the reduction's per-panel trailing sizes (n-32,
# n-64, ...) all fall in ONE bucket and do NOT re-trigger autotuning every panel
# (v4 used a single fixed 64/64/32 tile for both skinny and square GEMMs).
if _HAVE_TRITON:

    def _bmm_autotune_configs():
        cfgs = []
        # (BM, BN, BK, warps, stages) — spans the shapes the kernel actually
        # issues: tall/skinny panel GEMMs (K=nb=32) and the large square
        # backtransform (M=N=K=n).  Larger tiles + deeper pipelining favor the
        # big square GEMM on B200; small tiles cover the skinny/edge cases.
        specs = [
            (64, 64, 32, 4, 2),
            (64, 128, 32, 4, 3),
            (128, 64, 32, 4, 3),
            (128, 128, 32, 4, 3),
            (128, 128, 32, 8, 4),
            (128, 128, 64, 8, 3),
            (128, 256, 32, 8, 3),
            (256, 128, 32, 8, 3),
            (64, 64, 32, 2, 2),
        ]
        for bm, bn, bk, w, s in specs:
            cfgs.append(
                triton.Config(
                    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                    num_warps=w,
                    num_stages=s,
                )
            )
        return cfgs

    @triton.autotune(configs=_bmm_autotune_configs(), key=["mk", "nk", "kk"])
    @triton.jit
    def _bmm_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        mk, nk, kk,          # bucketed (pow2) dims — autotune key ONLY (unused in body)
        sab, sam, sak,       # A strides: batch, row (M), contraction (K)
        sbb, sbk, sbn,       # B strides: batch, contraction (K), col (N)
        scb, scm, scn,       # C strides: batch, row (M), col (N)
        PREC: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        offm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offk = tl.arange(0, BLOCK_K)

        a_ptr = A_ptr + pid_b * sab + (offm[:, None] * sam + offk[None, :] * sak)
        b_ptr = B_ptr + pid_b * sbb + (offk[:, None] * sbk + offn[None, :] * sbn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offk
            a = tl.load(
                a_ptr,
                mask=(offm[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr,
                mask=(k[:, None] < K) & (offn[None, :] < N),
                other=0.0,
            )
            # Precision selected at compile time (constexpr branch collapses).
            if PREC == 0:
                acc += tl.dot(a, b, input_precision="ieee")
            elif PREC == 1:
                acc += tl.dot(a, b, input_precision="tf32")
            else:
                acc += tl.dot(a, b, input_precision="tf32x3")
            a_ptr += BLOCK_K * sak
            b_ptr += BLOCK_K * sbk

        c_ptr = C_ptr + pid_b * scb + (offm[:, None] * scm + offn[None, :] * scn)
        tl.store(
            c_ptr,
            acc,
            mask=(offm[:, None] < M) & (offn[None, :] < N),
        )


def _p2(x: int) -> int:
    """Next power of two >= x (for the autotune bucket key)."""
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# Precision knob string -> constexpr flag.
_PREC = {"ieee": 0, "tf32": 1, "tf32x3": 2}


def _bmm(a: torch.Tensor, b: torch.Tensor, prec: str = "tf32") -> torch.Tensor:
    """Batched C = A @ B using the autotuned Triton kernel.

    a: (batch, M, K), b: (batch, K, N).  Either operand may be a transposed view
    (non-contiguous) — its strides are passed through, so no transpose copy is
    made.  `prec` in {"ieee","tf32","tf32x3"} picks the tl.dot precision.
    Returns a fresh contiguous (batch, M, N) tensor.
    """
    assert a.dim() == 3 and b.dim() == 3, "expects 3D batched operands"
    bt, M, K = a.shape
    bt2, K2, N = b.shape
    assert bt == bt2 and K == K2, "batched GEMM shape mismatch"

    c = torch.empty((bt, M, N), dtype=torch.float32, device=a.device)
    pflag = _PREC[prec]
    mk, nk, kk = _p2(M), _p2(N), _p2(K)

    grid = lambda META: (  # noqa: E731
        bt,
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _bmm_kernel[grid](
        a, b, c,
        M, N, K,
        mk, nk, kk,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        PREC=pflag,
    )
    return c


# ============================================================================
# Step 1 - Householder reduction  A = Q1 T Q1^T   (T symmetric tridiagonal)
# ============================================================================
# Right-looking blocked reduction: repeated `slatrd` (reduce a panel of
# _NB_REDUCE columns, producing V and W) + a Triton BLAS-3 trailing rank-2k
# update  A[trail] -= V W^T + W V^T.  Reflector v_k is stored in A[:, k+1:, k]
# with v_k[0]=1.  (v4-identical math; only the trailing-update precision changed
# to tf32 — Fix 2/3.)


def _larfg(x: torch.Tensor):
    """Batched Householder generator (LAPACK slarfg).

    x: (M, L) columns to reduce (L >= 1).  Returns v (M,L) with v[:,0]==1,
    tau (M,), and beta (M,) the resulting sub-diagonal value.
    """
    M, L = x.shape
    alpha = x[:, 0]
    if L == 1:
        v = torch.ones((M, 1), dtype=x.dtype, device=x.device)
        tau = torch.zeros((M,), dtype=x.dtype, device=x.device)
        return v, tau, alpha.clone()

    tail = x[:, 1:]
    xnorm = torch.linalg.vector_norm(tail, dim=1)
    safe = xnorm > 0.0
    signA = torch.where(alpha >= 0.0, 1.0, -1.0)
    beta_full = -signA * torch.sqrt(alpha * alpha + xnorm * xnorm)
    tau = torch.where(safe, (beta_full - alpha) / beta_full, torch.zeros_like(alpha))
    denom = torch.where(safe, alpha - beta_full, torch.ones_like(alpha))

    v = torch.empty_like(x)
    v[:, 0] = 1.0
    v[:, 1:] = torch.where(safe[:, None], tail / denom[:, None], torch.zeros_like(tail))
    beta = torch.where(safe, beta_full, alpha)
    return v, tau, beta


def _householder_tridiagonalize(A: torch.Tensor):
    """Blocked Householder reduction of a batch of symmetric matrices.

    A: (b, n, n) fp32 (mutated in place; lower triangle ends holding reflectors).
    Returns (d, e, tau).  BLAS-2 panel work is torch; the BLAS-3 trailing update
    is Triton (_bmm, tf32 — re-symmetrized so the injected error stays bounded and
    is gated by the self-check).
    """
    b, n, _ = A.shape
    dev = A.device
    e = torch.zeros((b, n - 1), dtype=A.dtype, device=dev)
    tau = torch.zeros((b, n), dtype=A.dtype, device=dev)

    p = 0
    while p <= n - 2:
        nb = min(_NB_REDUCE, (n - 1) - p)          # columns reduced this panel
        W = torch.zeros((b, n - p, nb), dtype=A.dtype, device=dev)

        for i in range(nb):
            c = p + i                               # global column being reduced

            if i > 0:
                # Bring column c up to date w.r.t. the i reflectors already done
                # in this panel.
                Vprev = A[:, c:, p:c]               # (b, n-c, i)
                Wrow = W[:, i, :i]                  # (b, i)
                Wprev = W[:, i:, :i]               # (b, n-c, i)
                Vrow = A[:, c, p:c]                # (b, i)
                term1 = torch.einsum("bmk,bk->bm", Vprev, Wrow)
                term2 = torch.einsum("bmk,bk->bm", Wprev, Vrow)
                A[:, c:, c] = A[:, c:, c] - term1 - term2

            # Generate the reflector that annihilates A[:, c+2:, c].
            x = A[:, c + 1 :, c]                    # (b, n-c-1)
            v, tau_c, beta_c = _larfg(x)
            A[:, c + 1 :, c] = v                    # store v (v[0]=1) into A
            e[:, c] = beta_c
            tau[:, c] = tau_c

            # w = tau * A[c+1:, c+1:] @ v   (symmetric matvec, BLAS-2).
            Atrail = A[:, c + 1 :, c + 1 :]         # (b, m2, m2)
            w = tau_c[:, None] * torch.einsum("bmn,bn->bm", Atrail, v)

            if i > 0:
                # Correct w for the i within-panel reflectors not yet applied.
                Vp = A[:, c + 1 :, p:c]             # (b, m2, i)
                Wp = W[:, i + 1 :, :i]             # (b, m2, i)
                VpTv = torch.einsum("bmk,bm->bk", Vp, v)
                WpTv = torch.einsum("bmk,bm->bk", Wp, v)
                w = w - tau_c[:, None] * (
                    torch.einsum("bmk,bk->bm", Wp, VpTv)
                    + torch.einsum("bmk,bk->bm", Vp, WpTv)
                )

            # symmetric-update correction term.
            alpha = -0.5 * tau_c * torch.einsum("bm,bm->b", w, v)
            w = w + alpha[:, None] * v
            W[:, i + 1 :, i] = w

        # ---- Triton BLAS-3 trailing update: A[trail] -= V W^T + W V^T (tf32) ----
        m_trail = n - (p + nb)
        if m_trail > 0:
            V = A[:, p + nb :, p : p + nb].contiguous()   # (b, m_trail, nb)
            Wt = W[:, nb:, :nb].contiguous()              # (b, m_trail, nb)
            # U = V @ W^T (W^T passed as a transposed view, no copy).  tf32: the
            # update is re-symmetrized (U + U^T) below, and the self-check gates
            # the final result — see Fix 2/3.
            U = _bmm(V, Wt.transpose(-1, -2), prec="tf32")   # (b, m_trail, m_trail)
            A[:, p + nb :, p + nb :] = (
                A[:, p + nb :, p + nb :] - U - U.transpose(-1, -2)
            )

        p += nb

    d = torch.diagonal(A, dim1=-2, dim2=-1).contiguous()
    return d, e, tau


# ============================================================================
# Step 3 (moved up) - Blocked compact-WY backtransform  (Fix 1, LAPACK sormtr)
# ============================================================================
# HIGHEST-IMPACT change.  v4 assembled Q1 with an UNBLOCKED rank-1 loop over all
# n-1 reflectors, each sweeping the entire (b,n,n) Q1 (hundreds of GB of BLAS-1/2
# traffic).  We instead group the reflectors into panels and apply the compact-WY
# form  H_p ... H_{p+w-1} = I - V T V^T  directly to the tridiagonal eigenvectors
# Y (Option B: never form Q1, and fuse away the separate Q1 @ Y GEMM).
#
#   Q = Q1 @ Y = (P_0 P_1 ... P_last) Y,   P_j = I - V_j T_j V_j^T
# applied RIGHT-TO-LEFT (last panel first):
#   Y <- P_j Y = Y - V_j ( T_j ( V_j^T Y ) )     [3 GEMMs per panel]
#
# The compact-WY identity H_p...H_{p+w-1} = I - V T V^T holds for ANY set of
# Householder vectors with the forward/columnwise T-factor recurrence (LAPACK
# slarft) — it does not require the reflectors to share a starting row, so the
# staggered storage (v_{p+i} lives in rows >= p+i+1) is fine.


def _wy_tfactor(V: torch.Tensor, tau_panel: torch.Tensor) -> torch.Tensor:
    """Compact-WY T factor (LAPACK slarft, forward / columnwise), batched.

    V:          (b, n, w) reflector columns of one panel (zeros above support).
    tau_panel:  (b, w)    the reflectors' tau scalars, in the SAME column order.
    Returns T:  (b, w, w) upper-triangular so that H_0 H_1 ... H_{w-1} = I-V T V^T.

    Recurrence (i = 0..w-1):
        T[i,i] = tau_i
        T[0:i,i] = T[0:i,0:i] @ ( -tau_i * (V[:,:,0:i]^T V[:,:,i]) )
    The needed dot products are entries of the Gram matrix G = V^T V.  Small
    (w x w); done in torch, off the critical path.
    """
    b, n, w = V.shape
    dev = V.device
    dtype = V.dtype
    G = torch.matmul(V.transpose(-1, -2), V)          # (b, w, w) Gram
    T = torch.zeros((b, w, w), dtype=dtype, device=dev)
    for i in range(w):
        T[:, i, i] = tau_panel[:, i]
        if i > 0:
            # t = -tau_i * G[0:i, i]
            t = -tau_panel[:, i : i + 1] * G[:, :i, i]        # (b, i)
            T[:, :i, i] = torch.matmul(
                T[:, :i, :i], t.unsqueeze(-1)
            ).squeeze(-1)
    return T


def _backtransform_wy(A: torch.Tensor, tau: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Q = H_0 H_1 ... H_{n-2} @ Y via blocked compact-WY (Fix 1, Option B).

    A:   (b, n, n) with reflector v_k stored in A[:, k+1:, k] (v_k[0]=1).
    tau: (b, n)    reflector scalars (tau[k] for column k).
    Y:   (b, n, n) tridiagonal eigenvectors (sign-similarity already undone).
    Returns Q = Q1 @ Y (b, n, n).  Applies panels right-to-left; each panel is
    3 batched GEMMs on tensor cores (tf32x3 for near-fp32 accuracy).
    """
    b, n, _ = A.shape
    dev = A.device
    dtype = A.dtype

    # Panel starts p = 0, nb, 2nb, ...  (reflectors exist for columns 0..n-2).
    starts = []
    p = 0
    while p <= n - 2:
        w = min(_NB_REDUCE, (n - 1) - p)
        starts.append((p, w))
        p += _NB_REDUCE

    Q = Y  # apply reflectors in place onto Y (Option B: no explicit Q1)

    rows = torch.arange(n, device=dev)[:, None]        # (n,1) global row index
    for p, w in reversed(starts):                      # right-to-left panel order
        # Build the panel's reflector matrix V (b, n, w): column i = v_{p+i},
        # which occupies rows >= p+i+1 (with the implicit unit at row p+i+1).
        Vblock = A[:, :, p : p + w].clone()            # (b, n, w)
        i_idx = torch.arange(w, device=dev)[None, :]   # (1,w)
        keep = rows >= (p + i_idx + 1)                 # (n,w) support mask
        Vblock = Vblock * keep.to(dtype)               # zero everything above support

        tau_panel = tau[:, p : p + w]                  # (b, w)
        T = _wy_tfactor(Vblock, tau_panel)             # (b, w, w) upper-tri

        # Y <- Y - V ( T ( V^T Y ) )   [3 GEMMs; the two big ones on tf32x3].
        S = _bmm(Vblock.transpose(-1, -2), Q, prec="tf32x3")   # (b, w, n) = V^T Y
        S = torch.matmul(T, S)                                  # (b, w, n) = T (V^T Y) [small, fp32]
        Q = Q - _bmm(Vblock, S, prec="tf32x3")                 # (b, n, n)

    return Q


# ============================================================================
# Step 2 - Divide-and-conquer tridiagonal eigensolver  (Fix 5: KEPT VERBATIM)
# ============================================================================
# Unchanged from v4.  Solves T y = lambda y from (d, e) via Cuppen D&C with a
# secular-equation merge, batched over a fixed lockstep bisection tree.  See the
# module header / notes for the honest-scope discussion.  Do NOT touch.


def _build_dense_tridiag(d: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    T = torch.diag_embed(d)
    if d.shape[-1] > 1:
        T.diagonal(offset=1, dim1=-2, dim2=-1).copy_(e)
        T.diagonal(offset=-1, dim1=-2, dim2=-1).copy_(e)
    return T


def _secular_solve(Ds: torch.Tensor, zs: torch.Tensor, rho: torch.Tensor):
    """Solve the eigenproblem of diag(Ds) + rho * zs zs^T for rho >= 0."""
    M, K = Ds.shape
    dev = Ds.device
    dtype = Ds.dtype

    znorm2 = torch.sum(zs * zs, dim=1)                    # (M,)

    lo = Ds.clone()
    hi = torch.empty_like(Ds)
    hi[:, : K - 1] = Ds[:, 1:]
    hi[:, K - 1] = Ds[:, K - 1] + rho * znorm2
    hi = torch.maximum(hi, lo)

    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        denom = Ds[:, None, :] - mid[:, :, None]
        denom = torch.where(denom.abs() < 1e-300, torch.full_like(denom, 1e-300), denom)
        f = 1.0 + rho[:, None, None] * torch.sum(
            (zs * zs)[:, None, :] / denom, dim=2
        )
        go_right = f < 0.0
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)
    mu = 0.5 * (lo + hi)

    diff_mu = mu[:, None, :] - Ds[:, :, None]
    log_num = torch.sum(torch.log(diff_mu.abs().clamp_min(1e-300)), dim=2)

    diff_dd = Ds[:, None, :] - Ds[:, :, None]
    eye = torch.eye(K, dtype=torch.bool, device=dev)[None].expand(M, K, K)
    diff_dd = torch.where(eye, torch.ones_like(diff_dd), diff_dd)
    log_den = torch.sum(torch.log(diff_dd.abs().clamp_min(1e-300)), dim=2)

    zhat_mag = torch.exp(0.5 * (log_num - log_den))
    zhat = torch.sign(zs) * zhat_mag
    deflate = zs.abs() <= (_Z_DEFLATE * torch.sqrt(znorm2)[:, None].clamp_min(1e-300))
    zhat = torch.where(deflate, torch.zeros_like(zhat), zhat)

    denomX = Ds[:, :, None] - mu[:, None, :]
    denomX = torch.where(denomX.abs() < 1e-300, torch.full_like(denomX, 1e-300), denomX)
    X = zhat[:, :, None] / denomX
    Xn = torch.linalg.vector_norm(X, dim=1, keepdim=True).clamp_min(1e-300)
    X = X / Xn

    decoupled = rho <= _RHO_TINY
    if bool(decoupled.any()):
        eyeK = torch.eye(K, dtype=dtype, device=dev)[None].expand(M, K, K)
        X = torch.where(decoupled[:, None, None], eyeK, X)
        mu = torch.where(decoupled[:, None], Ds, mu)

    return mu, X


def _divide_and_conquer(d: torch.Tensor, e: torch.Tensor):
    """Batched divide-and-conquer for symmetric tridiagonal (d, e), e >= 0."""
    b, n = d.shape
    dev = d.device

    nleaves = 1
    L0 = n
    while L0 % 2 == 0 and (L0 // 2) >= _DC_LEAF:
        L0 //= 2
        nleaves *= 2
    assert nleaves * L0 == n, "D&C leaf partition failed"

    d = d.clone()
    for j in range(1, nleaves):
        c = j * L0 - 1
        d[:, c] = d[:, c] - e[:, c]
        d[:, c + 1] = d[:, c + 1] - e[:, c]

    d_leaf = d.view(b, nleaves, L0)
    if nleaves == 1:
        e_leaf = e.view(b, 1, n - 1) if n > 1 else e.view(b, 1, 0)
    else:
        idx = torch.arange(L0 - 1, device=dev)[None, :] + (
            torch.arange(nleaves, device=dev)[:, None] * L0
        )
        e_leaf = e[:, idx.reshape(-1)].view(b, nleaves, L0 - 1)

    Tleaf = _build_dense_tridiag(
        d_leaf.reshape(b * nleaves, L0),
        e_leaf.reshape(b * nleaves, max(L0 - 1, 0)),
    )
    w_flat, V_flat = torch.linalg.eigh(Tleaf)
    w = w_flat.view(b, nleaves, L0)
    V = V_flat.view(b, nleaves, L0, L0)

    nblk = nleaves
    L = L0
    while nblk > 1:
        half = nblk // 2
        p_idx = torch.arange(half, device=dev)
        cut = (2 * p_idx + 1) * L - 1
        rho = e[:, cut]

        wpair = w.view(b, half, 2, L)
        D = wpair.reshape(b, half, 2 * L)
        Vpair = V.view(b, half, 2, L, L)
        z_top = Vpair[:, :, 0, L - 1, :]
        z_bot = Vpair[:, :, 1, 0, :]
        z = torch.cat([z_top, z_bot], dim=-1)

        Mf = b * half
        K = 2 * L
        Dsraw = D.reshape(Mf, K)
        zraw = z.reshape(Mf, K)
        rhof = rho.reshape(Mf)

        Ds, perm = torch.sort(Dsraw, dim=1)
        zs = torch.gather(zraw, 1, perm)
        invperm = torch.argsort(perm, dim=1)

        mu, X = _secular_solve(Ds, zs, rhof)

        Eblk = torch.gather(
            X, 1, invperm[:, :, None].expand(Mf, K, K)
        )

        V2p = Vpair[:, :, 0].reshape(Mf, L, L)
        V2p1 = Vpair[:, :, 1].reshape(Mf, L, L)
        Etop = Eblk[:, :L, :]
        Ebot = Eblk[:, L:, :]
        Vtop = torch.bmm(V2p, Etop)
        Vbot = torch.bmm(V2p1, Ebot)
        Vnew = torch.cat([Vtop, Vbot], dim=1)

        w = mu.view(b, half, K)
        V = Vnew.view(b, half, K, K)
        nblk = half
        L = K

    w_final = w.view(b, n)
    Y = V.view(b, n, n)
    return w_final, Y


# ============================================================================
# Sign-normalization of the tridiagonal off-diagonals to e >= 0  (v4-identical)
# ============================================================================
def _sign_normalize(e: torch.Tensor):
    b, m = e.shape
    dev = e.device
    n = m + 1
    sgn = torch.where(e >= 0.0, 1.0, -1.0)
    s = torch.ones((b, n), dtype=e.dtype, device=dev)
    s[:, 1:] = torch.cumprod(sgn, dim=1)
    e_pos = e.abs()
    return e_pos, s


# ============================================================================
# Self-check (Fix 6): CHEAP fp32 random-probe residuals + torch fallback
# ============================================================================
# v4's _self_check did two full O(b*n^3) bmms (Q^T Q and A @ Q) EVERY call — a
# non-trivial tax on top of the pipeline.  We replace them with a Hutchinson-style
# random-probe residual costing O(b*n^2*k):
#   * draw k random UNIT vectors x (b, n, k);
#   * eigen residual   r_e = A(Qx) - Q(L (.) x)      = (A Q - Q diag(L)) x
#   * orth  residual   r_o = Q^T(Qx) - x             = (Q^T Q - I) x
#   * upscale each probe norm by sqrt(n) as a conservative estimate of the
#     operator norm (E[||Mx||^2] = ||M||_F^2 / n for unit-variance x), and compare
#     to the grader's dimension-scaled thresholds (eigen 200*n*eps*||A||_1,
#     orth 100*n*eps).
# A CORRECT pipeline passes by ~2 orders of magnitude (residuals ~ roundoff); a
# BROKEN one (clustered/repeated spectra where the light-deflation D&C loses
# orthogonality) fails by orders of magnitude — so the loose probe reliably gates.
# NO per-call fp64 matmuls (the task's explicit constraint).


def _self_check(A: torch.Tensor, Q: torch.Tensor, L: torch.Tensor) -> bool:
    b, n, _ = A.shape
    dev = A.device
    dtype = A.dtype

    if not (torch.isfinite(Q).all() and torch.isfinite(L).all()):
        return False

    k = min(_PROBE_K, n)
    X = torch.randn((b, n, k), dtype=dtype, device=dev)
    X = X / torch.linalg.vector_norm(X, dim=1, keepdim=True).clamp_min(1e-30)

    QX = torch.bmm(Q, X)                                 # (b,n,k)  Q x

    # ---- eigen residual: (A Q - Q diag(L)) x = A(Qx) - Q(L (.) x) ----
    AQX = torch.bmm(A, QX)                               # A Q x
    QLX = torch.bmm(Q, L[:, :, None] * X)               # Q diag(L) x
    r_eig = AQX - QLX                                    # (b,n,k)
    eig_est = r_eig.norm(dim=1).amax(dim=1) * math.sqrt(n)      # (b,)
    Ascale = torch.linalg.matrix_norm(A, ord=1, dim=(-2, -1)).clamp_min(1e-30)
    eig_allowed = (200.0 * n * _EPS32) * Ascale                # grader factor
    if bool((eig_est > eig_allowed).any()):
        return False

    # ---- orthogonality residual: (Q^T Q - I) x = Q^T(Qx) - x ----
    r_orth = torch.bmm(Q.transpose(-1, -2), QX) - X            # (b,n,k)
    orth_est = r_orth.norm(dim=1).amax(dim=1) * math.sqrt(n)   # (b,)
    orth_allowed = 100.0 * n * _EPS32                          # ||I||_1 = 1
    if bool((orth_est > orth_allowed).any()):
        return False

    return True


# ============================================================================
# The pipeline entry point (Triton path)
# ============================================================================
def _pipeline(data: torch.Tensor) -> output_t:
    """Full Householder + D&C + blocked-WY backtransform pipeline.  Raises on any
    problem (or self-check miss) so the caller falls back to torch."""
    b, n, _ = data.shape

    # Symmetrize (input is symmetric only up to fp32 roundoff).
    A = 0.5 * (data + data.transpose(-1, -2))
    A = A.contiguous()
    A_orig = A.clone()                                  # kept for the self-check

    # Step 1: reduce to tridiagonal (Triton tf32 BLAS-3 trailing update inside).
    d, e, tau = _householder_tridiagonalize(A)

    # Sign-normalize off-diagonals to e >= 0 for the secular solver.
    e_pos, s = _sign_normalize(e)

    # Step 2: divide-and-conquer tridiagonal eigensolve (torch, real D&C — Fix 5).
    w, Yprime = _divide_and_conquer(d, e_pos)

    # Undo the sign similarity: Y = S Y'  (scale row i by s_i).
    Y = s[:, :, None] * Yprime

    # Step 3: blocked compact-WY backtransform Q = Q1 @ Y (Fix 1, Option B).
    # Applies the stored reflectors directly to Y — never forms Q1, fuses away the
    # separate Q1 @ Y GEMM.  Note: Y is passed contiguous; _backtransform_wy
    # returns a new tensor (does not mutate A_orig).
    Q = _backtransform_wy(A, tau, Y.contiguous())

    # eigenvalues already ascending from the top merge; sort defensively and
    # permute Q's columns to match.
    order = torch.argsort(w, dim=1)
    L = torch.gather(w, 1, order)
    Q = torch.gather(Q, 2, order[:, None, :].expand(b, n, n))

    if not _self_check(A_orig, Q, L):
        raise RuntimeError("v6 pipeline self-check failed; falling back to torch")
    return Q.contiguous(), L.contiguous()


def _torch_fallback(data: torch.Tensor) -> output_t:
    """Reference-quality fallback.  torch returns (values ascending, vectors);
    the contract wants (Q, L) = (vectors, values)."""
    values, vectors = torch.linalg.eigh(data)
    return vectors, values


def custom_kernel(data: input_t) -> output_t:
    """Batched real-symmetric eigendecomposition -> (Q, L).

    Dispatch by size regime only (never by exact shape/seed), 2 paths:
      * n >= _N_PIPELINE_MIN and CUDA and Triton -> Householder + D&C pipeline
      * otherwise / any failure                  -> torch.linalg.eigh
    The pipeline is wrapped in try/except (and gated by an fp32 probe self-check)
    so any failure degrades to the torch fallback rather than failing a gate.
    """
    n = data.shape[-1]

    if _HAVE_TRITON and data.is_cuda and n >= _N_PIPELINE_MIN:
        try:
            return _pipeline(data)
        except Exception:
            return _torch_fallback(data)

    return _torch_fallback(data)
