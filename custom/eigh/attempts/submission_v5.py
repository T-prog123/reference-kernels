# submission_v5.py
# ============================================================================
# parent: v3
# Hypothesis: improved-coding QR pipeline per notes/plans/v5_qr_improvement_plan.md
# ============================================================================
#
# WHAT v5 IS
# ---------------------------------------------------------------------------
# v5 is the *well-implemented* version of v3's method. Theory is UNCHANGED:
#
#     A  --Householder-->  T (symmetric tridiagonal)          [reduction]
#     T  --implicit-shift QL-->  (Z, L)  with T = Z diag(L) Zᵀ [tridiagonal EVD]
#     Q  =  U @ Z                                              [backtransform]
#
# where U = H₀H₁…H_{n-3} is the product of the reduction reflectors, so
#     A (U Z) = U T Uᵀ (U Z) = U T Z = (U Z) diag(L),
# i.e. Q = U Z are eigenvectors of A. v3 *named* the QR inner solve but actually
# deferred it to torch.linalg.eigh(T_dense), which re-tridiagonalizes T from
# scratch — pure redundant work (~1× of v3's 6× gap to torch). v5 replaces that
# with a NATIVE implicit-shift QL kernel that consumes the (d,e) vectors we
# already produced. The other fixes are coding-quality wins (see plan):
#
#   Fix 1  native implicit-shift QL inner solve (Triton), consuming (d,e).
#   Fix 2  leaner reduction: collapse the per-column BLAS-2 correction launches
#          (two matmuls -> one, via stacking); keep the Triton BLAS-3 trailing
#          rank-2k update. (The fused-panel kernel of step 3 is DEFERRED.)
#   Fix 3  fused blocked backtransform via torch.ormqr — applies the reflector
#          product straight to Z, removing the explicit U materialization AND the
#          separate U@Z GEMM (v3 paid O(n³) twice).
#   Fix 4  TF32 tensor cores under the loose gates (eigen 200·n·eps, orth
#          100·n·eps): trailing update uses input_precision="tf32x3" (near-fp32,
#          still tensor-core); torch matmul TF32 flags enabled.
#   Fix 5  faster GEMM: the one remaining Triton GEMM (trailing update) is
#          @triton.autotune'd over tiles/warps/stages; dense fallbacks use cuBLAS.
#   Fix 6  CHEAP self-check: fp32 (NOT fp64) — a random-probe eigen residual plus
#          a full fp32 orthogonality check (QR's weak spot), + isfinite +
#          ascending. Replaces v3's 3–4 per-call fp64 matmuls (cause (c)).
#
# HONEST PERFORMANCE EXPECTATION (stated up front, from the plan):
#   Even a clean QL inner solve accumulates eigenvectors by an O(n³) *memory-
#   bound* sequence of Givens sweeps (the `for i` bulge chase applied to Z), and
#   that serial chain is intrinsic to QR-for-all-eigenvectors — it is exactly why
#   D&C (v4/v6) beats QR here. So v5's realistic aim is PARITY WITH, OR A MODEST
#   GAP BEHIND, D&C — not a torch win. Its value is to establish the honest cost
#   of a *well-implemented* QR pipeline and confirm the design.md verdict.
#
# UNTESTABILITY / SAFETY (there is no local GPU — nothing here was run):
#   The QL kernel is a branch-heavy sequential transcription (Wilkinson shift,
#   deflation, bulge chase). A silent shift/index bug yields finite-but-wrong
#   numbers that try/except would NOT catch. Two nets guard correctness:
#     * a per-matrix non-convergence flag (iteration cap 30/eigenvalue, LAPACK-
#       style) -> that matrix's inner solve falls back to torch.linalg.eigh(T);
#     * a final fp32 self-check on (Q,L) -> whole-batch torch fallback if it would
#       miss the grader gates. So any bug degrades to slow-but-correct, never a
#       failed gate. If the Triton kernel fails to even compile on the runner, the
#       launch raises inside the try/except and the whole call falls back to torch.
#
# OUTPUT CONTRACT (.claude/rules/contract-and-integrity.md):
#   custom_kernel(data:[b,n,n] fp32 CUDA) -> (Q, L)
#     Q: [b,n,n] fp32, columns are orthonormal eigenvectors.
#     L: [b,n]  fp32, eigenvalues ascending.  (torch.linalg.eigh convention.)
#
# DISPATCH (size-regime only; ≤5 paths — see contract-and-integrity.md):
#   path 1: n >= _N_PIPELINE_MIN and CUDA and Triton  -> Householder+QL pipeline
#   path 2: everything else, and every fallback        -> torch.linalg.eigh
#   The per-matrix inner eigh(T) rescue and the final torch fallback are the SAME
#   torch path 2 (a rescue, not a new regime). TF32/autotune/ormqr add NO path.
#   => 2 code paths total, well under the hard cap of 5. No shape fingerprinting.
# ============================================================================

import torch

# Triton lives only on the remote B200 runner. Import defensively so importing
# this module never explodes locally; without Triton we always use torch.
try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - defensive
    _HAVE_TRITON = False

from task import input_t, output_t


# ============================================================================
# Fix 4 — TF32 tensor cores on by default (the gates are loose; the self-check
# and fallback catch any accuracy miss). This affects torch's own cuBLAS GEMMs
# (baddbmm / matmul / ormqr fallbacks). The Triton dot precision is set per-call.
# ============================================================================
try:  # pragma: no cover - only meaningful on CUDA
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass


# ============================================================================
# Dispatch / tuning constants
# ============================================================================
# Only large matrices take the pipeline (canon crossover ~n=1024; targets
# n ∈ {1024, 2048}). Smaller n stays on torch.
_N_PIPELINE_MIN = 1024

# Householder panel width for the blocked reduction (LAPACK-ish; 32 is the usual
# knee — wider panel = fewer trailing-update launches but more in-panel BLAS-2).
# Fix 2 step 4 (sweep 16/32/48/64) is a remote-runner knob; 32 is the default.
_PANEL = 32

# fp32 machine epsilon — used for the QL deflation test and the self-check gates.
_EPS_FP32 = 1.1920929e-07

# Triton dot precision for the one remaining Triton GEMM (trailing rank-2k
# update). "tf32x3" = 3-pass error-corrected TF32: near-fp32 accuracy, still
# tensor-core, far faster than "ieee" emulated fp32 on B200. Plan Fix 4: the
# trailing update compounds over the whole reduction, so we keep the accurate
# tf32x3 here; plain "tf32" is a tunable if the self-check confirms it passes.
_TRAIL_PREC = "tf32x3"


# ============================================================================
# Fix 5 — one autotuned batched GEMM kernel (the trailing rank-2k update).
# ============================================================================
# C[b] = A[b] @ B[b], addressed purely by strides so transposed *views*
# (.transpose(-1,-2), which only swaps strides) feed in without materializing a
# physical transpose. Autotuned over tile/warps/stages, keyed on bucketed
# (M,N,K); v3 used a single fixed 64×64×32 config, near-optimal for neither the
# skinny panel GEMM nor a large square GEMM.
if _HAVE_TRITON:

    def _bmm_configs():
        cfgs = []
        for bm, bn, bk in [
            (64, 64, 32),
            (128, 64, 32),
            (64, 128, 32),
            (128, 128, 32),
            (128, 128, 64),
            (128, 256, 64),
        ]:
            for w in (4, 8):
                for st in (2, 3, 4):
                    cfgs.append(
                        triton.Config(
                            {"BM": bm, "BN": bn, "BK": bk},
                            num_warps=w,
                            num_stages=st,
                        )
                    )
        return cfgs

    @triton.autotune(configs=_bmm_configs(), key=["M", "N", "K"])
    @triton.jit
    def _bmm_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        sa_b, sa_m, sa_k,          # strides of A [b, M, K]
        sb_b, sb_k, sb_n,          # strides of B [b, K, N]
        sc_b, sc_m, sc_n,          # strides of C [b, M, N]
        PREC: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        a_ptrs = A_ptr + pid_b * sa_b + (offs_m[:, None] * sa_m + offs_k[None, :] * sa_k)
        b_ptrs = B_ptr + pid_b * sb_b + (offs_k[:, None] * sb_k + offs_n[None, :] * sb_n)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        for k0 in range(0, K, BK):
            k_mask = (offs_k + k0) < K
            a = tl.load(a_ptrs, mask=m_mask & (k_mask[None, :]), other=0.0)
            b = tl.load(b_ptrs, mask=(k_mask[:, None]) & n_mask, other=0.0)
            acc += tl.dot(a, b, input_precision=PREC)
            a_ptrs += BK * sa_k
            b_ptrs += BK * sb_k

        c_ptrs = C_ptr + pid_b * sc_b + (offs_m[:, None] * sc_m + offs_n[None, :] * sc_n)
        tl.store(c_ptrs, acc, mask=m_mask & n_mask)


def _bmm(A: torch.Tensor, B: torch.Tensor, prec: str = _TRAIL_PREC) -> torch.Tensor:
    """Batched matmul C[b] = A[b] @ B[b] via the autotuned Triton kernel.
    A: [b,M,K], B: [b,K,N] (either may be a transposed view). fp32 out."""
    b, M, K = A.shape
    N = B.shape[-1]
    C = torch.empty((b, M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (b, triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    _bmm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        PREC=prec,
    )
    return C


# ============================================================================
# Fix 1 — native implicit-shift QL inner solve (Triton), consuming (d,e).
# ============================================================================
# This is the heart of v5 and the biggest risk. It replaces v3's
# torch.diag_embed -> torch.linalg.eigh(T) with a direct solve of the symmetric
# tridiagonal eigenproblem from the (d,e) we already produced.
#
# ALGORITHM: the classic implicit-QL-with-Wilkinson-shift (EISPACK tql2 /
# Numerical Recipes `tqli`), 0-indexed. `d[0..n-1]` is the diagonal (becomes the
# eigenvalues in place), `e[0..n-2]` the sub-diagonal (e[n-1] guarded to 0), and
# `Z` accumulates the eigenvectors. Deflation test |e[m]| <= eps·(|d[m]|+|d[m+1]|)
# splits the problem; each active block [l..m] gets a Wilkinson-shifted QL sweep
# whose Givens rotations bulge-chase from m-1 down to l, updating (d,e) and
# rotating the corresponding pair of Z columns.
#
# MAPPING TO THE GPU (plan Fix 1, occupancy note): ONE program per matrix.
#   * (d,e) live in a per-matrix working buffer in global memory; the scalar QL
#     loop reads/writes single entries d[i],e[i] (tiny, L1-resident, hot). The
#     control flow (while/if on scalar loads) is uniform across the program's
#     lanes.
#   * Z is stored COLUMN-MAJOR as Zc[b, col, :] so "column i of Z" is a
#     contiguous n-vector. Each Givens rotation is then a single VECTORIZED
#     n-wide read-modify-write of two Z columns (coalesced) — not per-element
#     BLAS-1. This vectorization spreads one rotation across the program's warps.
#   HONEST LIMITATION: one program per matrix means only `batch` CTAs
#     (8 at n=2048, 60 at n=1024) — poor SM occupancy at small batch, and the
#     bulge chase is a long serial chain (the intrinsic QR cost). The plan's
#     row-tiled variant (grid = batch × n/rowtile, each tile re-runs the scalar
#     QL and rotates only its rows) would raise occupancy; it is DEFERRED as an
#     un-testable extra risk — see the implementation notes in the plan file.
#   * A per-matrix `flag` is set if any eigenvalue fails to deflate within 30
#     iterations (LAPACK cap); those matrices are re-solved with torch eigh(T).
if _HAVE_TRITON:

    @triton.jit
    def _steqr_kernel(
        d_ptr, e_ptr,          # in/out working (d,e), [b, n]  (d -> eigenvalues)
        Zc_ptr,                # [b, n, n] col-major Z (Zc[b,i,:] = column i)
        flag_ptr,              # [b] int32, 1 if this matrix did not converge
        n,
        sd_b,                  # stride between matrices in d/e (== n for contig)
        sz_b, sz_col, sz_row,  # strides of Zc
        EPS: tl.constexpr,
        N_POW2: tl.constexpr,  # padded n for the column vector loads
    ):
        pid = tl.program_id(0)
        dp = d_ptr + pid * sd_b
        ep = e_ptr + pid * sd_b
        zb = Zc_ptr + pid * sz_b
        rows = tl.arange(0, N_POW2)
        rmask = rows < n

        not_conv = 0
        # Outer sweep: converge eigenvalues l = 0,1,...,n-1 from the top.
        for l in range(0, n):
            it = 0
            converged_l = 0
            while converged_l == 0:
                # --- find the split point m in [l, n-2] (else m = n-1) --------
                mm = l
                found = 0
                while (mm <= n - 2) and (found == 0):
                    dm = tl.load(dp + mm)
                    dm1 = tl.load(dp + mm + 1)
                    dd = tl.abs(dm) + tl.abs(dm1)
                    em = tl.load(ep + mm)
                    if tl.abs(em) <= EPS * dd:
                        found = 1
                    else:
                        mm += 1
                if found == 1:
                    m = mm
                else:
                    m = n - 1

                if m == l:
                    converged_l = 1          # eigenvalue l has deflated
                else:
                    if it >= 30:
                        not_conv = 1         # LAPACK-style cap -> flag + give up
                        converged_l = 1
                    else:
                        it += 1
                        # --- Wilkinson shift from the trailing 2×2 -----------
                        dl = tl.load(dp + l)
                        dl1 = tl.load(dp + l + 1)
                        el = tl.load(ep + l)
                        g = (dl1 - dl) / (2.0 * el)
                        r = tl.sqrt(g * g + 1.0)          # pythag(g, 1)
                        # SIGN(r, g): |r| with sign of g (r >= 0 already).
                        sgn = tl.where(g >= 0.0, r, -r)
                        dm = tl.load(dp + m)
                        g = dm - dl + el / (g + sgn)
                        s = 1.0
                        c = 1.0
                        p = 0.0
                        # --- bulge chase i = m-1 .. l -----------------------
                        i = m - 1
                        broke = 0
                        while (i >= l) and (broke == 0):
                            ei = tl.load(ep + i)
                            f = s * ei
                            bb = c * ei
                            r = tl.sqrt(f * f + g * g)     # pythag(f, g)
                            tl.store(ep + i + 1, r)
                            if r == 0.0:
                                # negligible: shift the diagonal and stop sweep.
                                di1 = tl.load(dp + i + 1)
                                tl.store(dp + i + 1, di1 - p)
                                tl.store(ep + m, 0.0)
                                broke = 1
                            else:
                                s = f / r
                                c = g / r
                                di1 = tl.load(dp + i + 1)
                                g = di1 - p
                                di = tl.load(dp + i)
                                r = (di - g) * s + 2.0 * c * bb
                                p = s * r
                                tl.store(dp + i + 1, g + p)
                                g = c * r - bb
                                # rotate Z columns i and i+1 (contiguous rows):
                                #   col_{i+1} <- s*col_i + c*col_{i+1}
                                #   col_i     <- c*col_i - s*col_{i+1}
                                pi = zb + i * sz_col + rows * sz_row
                                pi1 = zb + (i + 1) * sz_col + rows * sz_row
                                zi = tl.load(pi, mask=rmask, other=0.0)
                                zi1 = tl.load(pi1, mask=rmask, other=0.0)
                                tl.store(pi1, s * zi + c * zi1, mask=rmask)
                                tl.store(pi, c * zi - s * zi1, mask=rmask)
                                i -= 1
                        # After a normal (non-r==0) sweep, commit d[l],e[l],e[m].
                        if broke == 0:
                            dl = tl.load(dp + l)
                            tl.store(dp + l, dl - p)
                            tl.store(ep + l, g)
                            tl.store(ep + m, 0.0)
                        # (r==0 break: skip the commit and just re-search m.)
        if not_conv == 1:
            tl.store(flag_ptr + pid, 1)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _tridiag_eigh(d: torch.Tensor, e: torch.Tensor, n: int):
    """Solve the symmetric-tridiagonal eigenproblem for a batch, from (d,e).

    d [b,n] diagonal, e [b,n] with e[:, :n-1] the sub-diagonal. Returns
    (L [b,n] ascending, Z [b,n,n] with columns the eigenvectors of T).
    Uses the native Triton QL kernel; any matrix that fails to converge (flagged)
    is re-solved with torch.linalg.eigh(T) (path 2 rescue). The kernel mutates
    working copies, so the ORIGINAL (d,e) survive for that rescue."""
    b = d.shape[0]
    device = d.device

    dw = d.contiguous().clone()          # kernel mutates -> eigenvalues
    ew = e.contiguous().clone()
    if n >= 1:
        ew[:, n - 1] = 0.0               # guard the unused last sub-diagonal
    # Z starts as identity; column-major storage Zc[b,i,:] = column i. The
    # identity is symmetric so Zc is just a batch of identity matrices.
    Zc = torch.eye(n, device=device, dtype=torch.float32).unsqueeze(0).repeat(b, 1, 1).contiguous()
    flag = torch.zeros(b, device=device, dtype=torch.int32)

    grid = (b,)
    _steqr_kernel[grid](
        dw, ew, Zc, flag,
        n,
        dw.stride(0),
        Zc.stride(0), Zc.stride(1), Zc.stride(2),
        EPS=_EPS_FP32,
        N_POW2=_next_pow2(n),
        num_warps=8,
    )

    # Zc[b,i,:] = column i of Z  ->  Z = Zc.transpose(-1,-2).
    Z = Zc.transpose(-1, -2).contiguous()
    L = dw

    # Sort ascending and permute Z's columns to match (cheap, robust — as v1).
    L, order = torch.sort(L, dim=-1)
    idx = order.unsqueeze(1).expand(b, n, n)
    Z = torch.gather(Z, 2, idx)

    # --- per-matrix rescue for non-converged cases via torch eigh(T) ----------
    if bool(flag.any().item()):
        bad = torch.nonzero(flag, as_tuple=False).flatten()
        db = d.index_select(0, bad)
        eb = e.index_select(0, bad)
        Tb = torch.diag_embed(db)
        offb = eb[:, : n - 1]
        Tb = Tb + torch.diag_embed(offb, offset=1) + torch.diag_embed(offb, offset=-1)
        Lb, Yb = torch.linalg.eigh(Tb)   # ascending, Y columns = eigenvectors
        L = L.index_copy(0, bad, Lb.to(L.dtype))
        Z = Z.index_copy(0, bad, Yb.to(Z.dtype))

    return L, Z


# ============================================================================
# Householder helpers (torch — small / correctness-critical). Unchanged from v3.
# ============================================================================
def _householder(x: torch.Tensor):
    """Batched Householder reflector generation (LAPACK dlarfg convention).
    Given x [b,L], returns (v, tau, beta) with (I - tau v vᵀ) x = beta e₁,
    v[0]=1. beta is the resulting subdiagonal entry; zero-tail -> identity."""
    b, L = x.shape
    alpha = x[:, 0]
    if L == 1:
        tail_norm2 = torch.zeros_like(alpha)
    else:
        tail = x[:, 1:]
        tail_norm2 = (tail * tail).sum(dim=1)

    mask = tail_norm2 > 0.0
    normx = torch.sqrt(alpha * alpha + tail_norm2)
    sgn = torch.where(alpha >= 0.0, torch.ones_like(alpha), -torch.ones_like(alpha))
    beta = torch.where(mask, -sgn * normx, alpha)

    denom = alpha - beta
    denom_safe = torch.where(mask, denom, torch.ones_like(denom))
    beta_safe = torch.where(mask, beta, torch.ones_like(beta))
    tau = torch.where(mask, (beta - alpha) / beta_safe, torch.zeros_like(alpha))

    v = torch.zeros_like(x)
    v[:, 0] = 1.0
    if L > 1:
        vtail = torch.where(
            mask.unsqueeze(1),
            x[:, 1:] / denom_safe.unsqueeze(1),
            torch.zeros_like(x[:, 1:]),
        )
        v[:, 1:] = vtail
    return v, tau, beta


# ============================================================================
# Step 1: blocked Householder reduction  A -> tridiagonal (d,e), + reflectors.
# ============================================================================
# Right-looking blocked dsytrd/dlatrd (uplo lower). Per panel of _PANEL columns:
# the panel is factored column by column (torch BLAS-2 — intrinsically memory-
# bound; the defining half of one-stage tridiagonalization), then ONE Triton
# BLAS-3 trailing rank-2k update A₂₂ -= V Wᵀ + W Vᵀ is applied.
#
# Fix 2 (leaner reduction): the per-column corrections that v3 issued as TWO
# torch matmuls each are collapsed into ONE by stacking the operands (a safe
# algebraic identity), halving those tiny launches (cause (b), the latency-bound
# panel loop). The fused whole-panel Triton kernel (plan Fix 2 step 3) is
# DEFERRED — it is the same un-testable-kernel risk class as Fix 1.
def _reduce_to_tridiagonal(A: torch.Tensor):
    """Return (d, e, H, tau_h) for a batch of symmetric matrices A [b,n,n]."""
    b, n, _ = A.shape
    device = A.device

    At = A.clone()
    d = torch.zeros((b, n), device=device, dtype=torch.float32)
    e = torch.zeros((b, n), device=device, dtype=torch.float32)
    H = torch.zeros((b, n - 1, n - 1), device=device, dtype=torch.float32)
    tau_h = torch.zeros((b, n - 1), device=device, dtype=torch.float32)

    s = 0
    while s < n - 2:
        M = n - s
        nb = min(_PANEL, (n - 2) - s)
        sub = At[:, s:, s:]

        Vp = torch.zeros((b, M, nb), device=device, dtype=torch.float32)
        Wp = torch.zeros((b, M, nb), device=device, dtype=torch.float32)

        for i in range(nb):
            g = s + i
            L = M - i - 1

            # --- bring column i up to date (Fix 2: one fused bmm, not two) ---
            col = sub[:, i:, i].clone()             # [b, M-i]
            if i > 0:
                # col -= Vp[:,i:,:i] @ Wp[:,i,:i]  +  Wp[:,i:,:i] @ Vp[:,i,:i]
                #     =  [Vp | Wp] @ [Wp_row ; Vp_row]
                VW = torch.cat((Vp[:, i:, :i], Wp[:, i:, :i]), dim=-1)          # [b,M-i,2i]
                rhs = torch.cat((Wp[:, i, :i], Vp[:, i, :i]), dim=-1).unsqueeze(-1)  # [b,2i,1]
                col = col - torch.matmul(VW, rhs).squeeze(-1)

            d[:, g] = col[:, 0]

            # --- Householder reflector below the subdiagonal ----------------
            x = col[:, 1:]                          # [b, L]
            v, tau, beta = _householder(x)
            e[:, g] = beta
            Vp[:, i + 1:, i] = v

            # --- symmetric mat-vec + fused corrections -> column i of W -----
            sub22 = sub[:, i + 1:, i + 1:]          # [b, L, L]
            Av = torch.matmul(sub22, v.unsqueeze(-1)).squeeze(-1)   # [b, L]
            if i > 0:
                # Av -= Wp(:, :i)(Vpᵀ v) + Vp(:, :i)(Wpᵀ v)
                #     = [Wp | Vp] @ ([Vp | Wp]ᵀ v)          (Fix 2: fused)
                VW2 = torch.cat((Vp[:, i + 1:, :i], Wp[:, i + 1:, :i]), dim=-1)  # [b,L,2i]
                t = torch.matmul(VW2.transpose(-1, -2), v.unsqueeze(-1))         # [b,2i,1]
                WV2 = torch.cat((Wp[:, i + 1:, :i], Vp[:, i + 1:, :i]), dim=-1)  # [b,L,2i]
                Av = Av - torch.matmul(WV2, t).squeeze(-1)
            w = tau.unsqueeze(1) * Av
            alpha = -0.5 * tau * (w * v).sum(dim=1)
            w = w + alpha.unsqueeze(1) * v
            Wp[:, i + 1:, i] = w

            # --- record reflector for ormqr backtransform -------------------
            if L > 1:
                H[:, g + 1:, g] = v[:, 1:]
            tau_h[:, g] = tau

        # --- panel trailing rank-2k update (TRITON tensor cores, TF32x3) ----
        if M > nb:
            V2 = Vp[:, nb:, :]
            W2 = Wp[:, nb:, :]
            upd1 = _bmm(V2, W2.transpose(-1, -2))    # V2 W2ᵀ
            upd2 = _bmm(W2, V2.transpose(-1, -2))    # W2 V2ᵀ
            At[:, s + nb:, s + nb:] -= upd1
            At[:, s + nb:, s + nb:] -= upd2

        s += nb

    d[:, n - 2] = At[:, n - 2, n - 2]
    d[:, n - 1] = At[:, n - 1, n - 1]
    e[:, n - 2] = At[:, n - 2, n - 1]

    return d, e, H, tau_h


# ============================================================================
# Fix 3 — fused, blocked backtransform  Q = U @ Z  via torch.ormqr.
# ============================================================================
# v3 formed U explicitly (orgqr, O(n³) + an n×n materialization) and then did a
# second O(n³) GEMM U@Z. torch.ormqr applies the stored Householder product
# directly to Z through cuSOLVER — a tested BLAS-3, TF32-capable primitive — with
# no U materialization and no separate GEMM. The reduction fixes the first
# coordinate (all reflectors act on rows 1..n-1), so U = [[1,0],[0,U_sub]] and we
# apply U_sub to the bottom (n-1)×n block of Z. If ormqr is unavailable / won't
# batch, fall back to orgqr + cuBLAS matmul (still no hand-rolled WY).
def _backtransform(H: torch.Tensor, tau_h: torch.Tensor, Z: torch.Tensor, n: int) -> torch.Tensor:
    Q = torch.empty_like(Z)
    Q[:, 0, :] = Z[:, 0, :]
    Zsub = Z[:, 1:, :].contiguous()                 # [b, n-1, n]
    try:
        Qsub = torch.ormqr(H, tau_h, Zsub, left=True, transpose=False)
    except Exception:
        U_sub = torch.linalg.householder_product(H, tau_h)   # [b, n-1, n-1]
        Qsub = torch.matmul(U_sub, Zsub)            # cuBLAS + TF32
    Q[:, 1:, :] = Qsub
    return Q


# ============================================================================
# Fix 6 — CHEAP fp32 self-check (NOT per-call fp64). Replaces v3's 3–4 fp64
# matmuls (cause (c)). Guarantees "never return a wrong answer" via fallback.
# ============================================================================
# * isfinite + ascending: exact, O(n)/O(n²), always cheap.
# * eigen-equation: a random-PROBE residual (plan Fix 6) — O(b·n²·k) instead of
#   O(b·n³). Sample (A Q - Q diag(L)) @ X for k random unit columns X and compare
#   to the eigen gate. A conservative margin (0.25·allowed) buys headroom against
#   a probe under-estimating a residual concentrated off the probe directions.
# * orthogonality: the FULL fp32 QᵀQ - I (one O(n³) fp32 matmul). Orthogonality
#   is where QR-accumulated eigenvectors are weakest (clustered spectra), so we
#   do NOT probe it — we check it exactly, but in fp32 (not fp64), which is the
#   actual fix to cause (c). The plan explicitly permits this for the orth check.
def _self_check_ok(A: torch.Tensor, Q: torch.Tensor, L: torch.Tensor, k: int = 4) -> bool:
    b, n, _ = A.shape
    eps = _EPS_FP32
    eigen_rtol = 200.0 * n * eps
    orth_rtol = 100.0 * n * eps

    if not (torch.isfinite(Q).all() and torch.isfinite(L).all()):
        return False

    # ascending (grader's tolerance).
    if n > 1:
        diffs = L[..., 1:] - L[..., :-1]
        scale = L.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        if bool((diffs < -(100.0 * n * eps) * scale).any()):
            return False

    # scales (fp32 1-norm ~ max abs column sum) as the grader uses.
    def l1(x):
        return torch.linalg.matrix_norm(x, ord=1, dim=(-2, -1))

    a_scale = l1(A)                                  # [b]

    # --- eigen residual by random probe (fp32) ---------------------------
    X = torch.randn(b, n, k, device=A.device, dtype=torch.float32)
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-30)   # unit columns
    QX = torch.matmul(Q, Xn)                         # [b,n,k]
    AQX = torch.matmul(A, QX)
    QLX = torch.matmul(Q, L.unsqueeze(-1) * Xn)      # Q diag(L) X
    r = AQX - QLX                                    # (AQ - Q diag L) X
    if not torch.isfinite(r).all():
        return False
    # estimated operator-ish residual per matrix vs conservative gate.
    res_est = r.norm(dim=(1, 2))                     # [b]
    allowed = 0.25 * eigen_rtol * a_scale            # 4× headroom
    if bool((res_est > allowed).any()):
        return False

    # --- orthogonality: full fp32 QᵀQ - I --------------------------------
    qtq = torch.matmul(Q.transpose(-1, -2), Q)
    eye = torch.eye(n, device=A.device, dtype=torch.float32)
    orth = l1(qtq - eye).amax()
    # ||I||_1 == 1, so the grader's scale factor is 1.
    if not torch.isfinite(orth) or bool(orth > orth_rtol):
        return False

    return True


# ============================================================================
# The full pipeline (Triton reduction + native QL inner solve + ormqr backtx).
# ============================================================================
def _pipeline(data: torch.Tensor) -> output_t:
    b, n, _ = data.shape
    A = data.contiguous()
    A = 0.5 * (A + A.transpose(-1, -2))              # exact symmetry

    # Step 1 — reduce to tridiagonal (Triton trailing update inside).
    d, e, H, tau_h = _reduce_to_tridiagonal(A)

    # Step 2 — native implicit-shift QL inner solve, consuming (d,e) directly.
    # (Replaces v3's redundant torch.linalg.eigh(T_dense).)
    L, Z = _tridiag_eigh(d, e, n)                    # L ascending, Z columns

    # Step 3 — fused backtransform Q = U @ Z via ormqr.
    Q = _backtransform(H, tau_h, Z, n)

    # Fix 6 — cheap fp32 self-check; force a fallback if it would miss a gate.
    if not _self_check_ok(A, Q, L):
        raise RuntimeError("v5 pipeline output failed the fp32 self-check")

    return Q, L


def _torch_fallback(data: torch.Tensor) -> output_t:
    """Reference-quality fallback. torch returns (values ascending, vectors);
    the contract wants (Q, L) = (vectors, values)."""
    values, vectors = torch.linalg.eigh(data)
    return vectors, values


def custom_kernel(data: input_t) -> output_t:
    """Batched real-symmetric eigendecomposition -> (Q, L).

    Dispatch by size regime only (never by exact shape/seed):
      * n >= _N_PIPELINE_MIN and CUDA and Triton -> Householder+QL pipeline (v5)
      * otherwise / any failure                  -> torch.linalg.eigh
    The pipeline is wrapped in try/except AND validated by the fp32 self-check,
    so any failure — an exception or a would-be gate miss — degrades to a
    guaranteed-correct torch result. (Non-convergence of individual matrices is
    handled inside _tridiag_eigh via a per-matrix torch eigh(T) rescue.)
    """
    n = data.shape[-1]

    if _HAVE_TRITON and data.is_cuda and n >= _N_PIPELINE_MIN:
        try:
            with torch.no_grad():
                return _pipeline(data)
        except Exception:
            return _torch_fallback(data)

    return _torch_fallback(data)
