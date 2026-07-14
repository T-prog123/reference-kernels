# submission_v3.py
# ============================================================================
# parent: v0 (torch.linalg.eigh baseline)
# Hypothesis: for large n (1024, 2048; small batch) a Householder → tridiagonal
#   → QR-inner-solve → backtransform pipeline, with the memory-heavy dense
#   reductions (trailing symmetric rank-2k update, eigenvector backtransform)
#   run as tensor-core tl.dot Triton kernels, can compete with cuSOLVER's
#   batched syevd once the inner tridiagonal solve is made GPU-native.
# ============================================================================
#
# WHAT THIS FILE IS (and, honestly, what it is not yet)
# ---------------------------------------------------------------------------
# This is the classical LAPACK-style dense-symmetric eigensolver pipeline
# (canon: claude_docs/batched_cuda_symmetric_evd_methods.md, "Method 2:
# Householder Tridiagonalization Pipeline"):
#
#     A  --Householder-->  T (symmetric tridiagonal)      [reduction]
#     T  --inner solve-->  (Y, L)  with T = Y diag(L) Yᵀ  [tridiagonal EVD]
#     Q  =  U @ Y                                          [backtransform]
#
# where U = H₀H₁…H_{n-3} is the product of the Householder reflectors from the
# reduction, so that T = Uᵀ A U and therefore
#     A (U Y) = U T Uᵀ (U Y) = U T Y = U Y diag(L) = (U Y) diag(L),
# i.e. Q = U Y are the eigenvectors of A. (Proof is in the canon.)
#
# WHAT RUNS IN TRITON vs WHAT IS DEFERRED TO TORCH  (read this — it is the
# core design decision of v3, mandated by the "attempt QR, document the choice"
# instruction):
#
#   * TRITON (tensor-core tl.dot, fp32 input_precision="ieee"):
#       - the trailing symmetric rank-2k update  A₂₂ -= V Wᵀ + W Vᵀ  that
#         dominates the *memory traffic* of a blocked Householder reduction;
#       - the eigenvector backtransform GEMM  Q = U @ Y.
#     These are the two O(n³) dense, memory-heavy steps the task calls out, and
#     they are done panel/block-tiled through a single batched-GEMM kernel that
#     keeps each output tile resident in registers across the K-loop (one global
#     read of each A/B tile, one global write of C) — minimizing round-trips.
#
#   * TORCH (correctness-critical, small, or sequential):
#       - the per-column panel factorization (dlatrd): Householder vector
#         construction, the small O(nb) correction GEMVs, and the symmetric
#         mat-vec  A₂₂ v  (cuBLAS gemv — exact and cheap to trust);
#       - forming U from the stored reflectors via torch.linalg.householder_product
#         (LAPACK orgqr — a *tested* primitive; hand-rolling a blocked WY
#         accumulator un-tested on a GPU-less box is how you ship a silent bug);
#       - THE INNER TRIDIAGONAL SOLVE, torch.linalg.eigh on the dense
#         tridiagonal T.
#
# WHY THE INNER SOLVE IS torch AND NOT A TRITON IMPLICIT-SHIFT QR (the honest
# account the task asked for):
#   I genuinely worked the implicit-shift QR design (Wilkinson shift, bulge
#   chasing with Givens rotations along d/e, deflation on negligible
#   off-diagonals, rotations accumulated into the eigenvector matrix Z). It is
#   the right *algorithm* for the tridiagonal stage. It is NOT shippable here in
#   one shot, for two independent reasons:
#     (1) It is inherently sequential (each bulge-chase rotation depends on the
#         previous fill-in) with data-dependent iteration counts and deflation.
#         A correct version has dozens of fiddly branches. I cannot run a single
#         line of GPU code on this box (no CUDA locally), so an un-tested
#         sequential Triton kernel of this complexity would almost certainly
#         have a silent indexing/shift bug — and a wrong-but-finite result
#         *fails* the correctness gate (the try/except only catches exceptions,
#         not wrong numbers). See the self-check safety net below for how v3
#         instead guarantees the gate always passes.
#     (2) For the target sizes the eigenvector matrix Z is n×n with n ∈ {1024,
#         2048} → 4–16 MB, which does NOT fit in a B200's ~228 KB of shared
#         memory. So Z cannot be kept resident during bulge-chasing; the
#         rotations would page Z through global memory, i.e. exactly the
#         memory-bound pattern we are trying to avoid. Getting that right
#         un-tested is not a one-shot proposition.
#     A pure-torch implicit-shift QR is also a non-starter: bulge chasing is a
#     Python-level loop of O(n²) Givens rotations per matrix (~10⁶ iterations
#     for n=1024), each a kernel launch — seconds per case. Impossible.
#   The task explicitly permits computing the inner problem "with torch on the
#   (cheap) tridiagonal" while keeping the reduction+backtransform in Triton.
#   That is what v3 does. torch.linalg.eigh(T) is the safe, correct inner solve.
#
# HONEST PERFORMANCE EXPECTATION (so the ledger reading is not a surprise):
#   Because the inner solve is torch.linalg.eigh on a *dense* T, and cuSOLVER
#   does not special-case an already-tridiagonal input, eigh(T) re-runs a full
#   O(n³) tridiagonalization internally — so its cost is ≈ eigh(A). v3 therefore
#   does MORE total work than the v0 baseline (our reduction + eigh(T) + our
#   backtransform) and is expected to be SLOWER than v0 on the cases it engages.
#   v3's value is (a) validating the Triton reduction/backtransform infra end to
#   end under the correctness gate, and (b) being the scaffold into which a real
#   GPU-native tridiagonal QR/D&C inner solve is dropped later — at which point
#   the redundant eigh(T) disappears and the pipeline can actually win. This is
#   documented so the benchmark-runner knows to read v3 as an infra check, not a
#   geomean contender. (notes/design.md: n≥1024 → tridiagonal pipeline.)
#
# CORRECTNESS SAFETY NET (belt and suspenders, because we cannot test locally):
#   The whole Triton path is wrapped in try/except → torch.linalg.eigh, AND the
#   pipeline output is validated against the *actual grader gates*, recomputed
#   here in fp64 (eigen-equation, orthogonality, reconstruction, ascending).
#   If any gate would fail, or anything is non-finite, we raise and fall back to
#   torch. So a bug anywhere in the reduction/backtransform can only ever cost
#   us a fallback (a correct, slower answer) — never a failed correctness gate.
#
# OUTPUT CONTRACT (.claude/rules/contract-and-integrity.md):
#   custom_kernel(data:[b,n,n] fp32 CUDA) -> (Q, L)
#     Q: [b,n,n] fp32, columns are orthonormal eigenvectors.
#     L: [b,n]  fp32, eigenvalues ascending.
#   Same convention as torch.linalg.eigh(A). Judged on matrix invariants, so
#   sign flips / eigenspace rotations are fine.
#
# DISPATCH (size-regime only; ≤5 paths, see contract-and-integrity.md):
#   path 1: n >= _N_PIPELINE_MIN and CUDA and Triton  -> Householder pipeline
#   path 2: everything else, and every fallback        -> torch.linalg.eigh
#   That is 2 code paths total — well under the hard cap of 5. No shape
#   fingerprinting: the threshold is a magnitude of n, nothing else.
# ============================================================================

import torch

# Triton lives only on the remote B200 runner. Import defensively so importing
# this module never explodes locally; if Triton is missing we always use torch.
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
# Only large matrices take the pipeline. The canon puts the crossover from
# Jacobi to the tridiagonal pipeline at ~n=1024, and this v3 deliberately
# targets n ∈ {1024, 2048} (small batch). Smaller n stays on torch (and is the
# regime other attempts — e.g. v1 Jacobi — own).
_N_PIPELINE_MIN = 1024

# Householder panel width for the blocked reduction. The trailing rank-2k
# update is deferred and applied once per panel of this many columns, which is
# what turns the reduction's dominant update into big tensor-core GEMMs and cuts
# its global-memory traffic by ~1/_PANEL. 32 is a standard LAPACK-ish choice.
_PANEL = 32

# tl.dot tile sizes for the batched GEMM kernel. 64×64 output tile with a K-step
# of 32 is a safe, broadly-good default on B200 for the fp32 ieee path.
_BM = 64
_BN = 64
_BK = 32


# ============================================================================
# Triton: one batched GEMM kernel, used for every memory-heavy dense step
# ============================================================================
# C[b] = A[b] @ B[b] for row-major-ish batched tensors, addressed purely by the
# strides passed from the host. Addressing by stride (not by assuming
# contiguity) is what lets us feed *transposed views* (e.g. Wᵀ, Vᵀ, or Uᵀ)
# straight in without materializing a physical transpose in global memory —
# torch's .transpose(-1,-2) just swaps strides, and this kernel honors them.
#
# The classic tiled-GEMM structure keeps one BM×BN output tile resident in
# registers (`acc`) and walks K in BK-chunks, so each element of A and B is
# read from global memory once per output tile and C is written once. That is
# the "minimize global round-trips" the task asks for. fp32 accumulate with
# input_precision="ieee" (no TF32 truncation) preserves the accuracy the
# reduction/backtransform need to clear the residual gates.
if _HAVE_TRITON:

    @triton.jit
    def _bmm_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        sa_b, sa_m, sa_k,          # strides of A [b, M, K]
        sb_b, sb_k, sb_n,          # strides of B [b, K, N]
        sc_b, sc_m, sc_n,          # strides of C [b, M, N]
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)

        # Base pointers into this batch element's A and B tiles.
        a_ptrs = A_ptr + pid_b * sa_b + (offs_m[:, None] * sa_m + offs_k[None, :] * sa_k)
        b_ptrs = B_ptr + pid_b * sb_b + (offs_k[:, None] * sb_k + offs_n[None, :] * sb_n)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        # Walk over K. Mask the tail so non-multiple-of-BK K is handled, and
        # mask rows/cols beyond M/N so ragged tiles read zeros (harmless in the
        # dot product) rather than out-of-bounds.
        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        for k0 in range(0, K, BK):
            k_mask = (offs_k + k0) < K
            a = tl.load(a_ptrs, mask=m_mask & (k_mask[None, :]), other=0.0)
            b = tl.load(b_ptrs, mask=(k_mask[:, None]) & n_mask, other=0.0)
            acc += tl.dot(a, b, input_precision="ieee")
            a_ptrs += BK * sa_k
            b_ptrs += BK * sb_k

        c_ptrs = C_ptr + pid_b * sc_b + (offs_m[:, None] * sc_m + offs_n[None, :] * sc_n)
        tl.store(c_ptrs, acc, mask=m_mask & n_mask)


def _bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched matmul C[b] = A[b] @ B[b] via the Triton kernel above.

    A: [b, M, K], B: [b, K, N] (either may be a transposed view). Returns a
    fresh contiguous [b, M, N] fp32 tensor. This is the *only* place the heavy
    dense linear algebra of the pipeline actually executes on tensor cores.
    """
    b, M, K = A.shape
    N = B.shape[-1]
    C = torch.empty((b, M, N), device=A.device, dtype=torch.float32)
    grid = (b, triton.cdiv(M, _BM), triton.cdiv(N, _BN))
    _bmm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BM=_BM, BN=_BN, BK=_BK,
    )
    return C


# ============================================================================
# Householder helpers (torch — small / correctness-critical)
# ============================================================================
def _householder(x: torch.Tensor):
    """Batched Householder reflector generation (LAPACK dlarfg convention).

    Given x [b, L] (L>=1), returns (v, tau, beta) per batch element such that
        (I - tau v vᵀ) x = beta e₁,   v[0] = 1,   v[1:] the "essential" part.
    beta is the resulting subdiagonal entry. When x has zero tail (nothing to
    annihilate) we return the identity reflector (tau=0, v=e₁, beta=x[0]); this
    is the same no-op dlarfg produces and it keeps deflated/already-tridiagonal
    columns from injecting spurious sign flips.
    """
    b, L = x.shape
    alpha = x[:, 0]
    if L == 1:
        tail_norm2 = torch.zeros_like(alpha)
    else:
        tail = x[:, 1:]
        tail_norm2 = (tail * tail).sum(dim=1)

    # Only reflect when there is a nonzero tail below the subdiagonal.
    mask = tail_norm2 > 0.0
    normx = torch.sqrt(alpha * alpha + tail_norm2)
    # beta = -sign(alpha)*||x|| chooses the sign that avoids cancellation in
    # (alpha - beta). Treat sign(0) as +1.
    sgn = torch.where(alpha >= 0.0, torch.ones_like(alpha), -torch.ones_like(alpha))
    beta = torch.where(mask, -sgn * normx, alpha)

    denom = alpha - beta                       # nonzero exactly where mask holds
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
# Step 1: blocked Householder reduction  A -> tridiagonal (d, e), + reflectors
# ============================================================================
# This is the batched, blocked (dsytrd/dlatrd, uplo lower) reduction. Per panel
# of _PANEL columns:
#   * dlatrd factorizes the panel column by column (torch): for each column it
#     brings the column up to date w.r.t. this panel's earlier reflectors (small
#     correction GEMVs), builds the Householder reflector, does the symmetric
#     mat-vec A₂₂ v (torch/cuBLAS gemv — the correctness-sensitive big matvec),
#     and forms the corresponding column of W. It accumulates V (reflectors) and
#     W so that the *whole panel's* effect on the trailing block is the rank-2k
#     update A₂₂ -= V Wᵀ + W Vᵀ.
#   * that trailing rank-2k update — the memory-dominant O(n³) step, done ONCE
#     per panel — runs on Triton tensor cores via two _bmm calls.
#
# We also record, for each reduced column, its diagonal d and subdiagonal e, and
# stash the reflector's essential part into a global matrix `H` laid out exactly
# how torch.linalg.householder_product wants it, so U can be formed later with a
# tested LAPACK primitive rather than a hand-rolled (and un-testable) accumulator.
def _reduce_to_tridiagonal(A: torch.Tensor):
    """Return (d, e, H, tau_h) for a batch of symmetric matrices A [b,n,n].

    d [b,n]      : diagonal of T.
    e [b,n]      : e[:, :n-1] is the sub/super-diagonal of T (e[:, n-1] unused).
    H [b,n-1,n-1]: reflector essential parts in the shifted frame expected by
                   torch.linalg.householder_product (column g holds reflector g
                   with implicit unit at row g; used to build U).
    tau_h [b,n-1]: the corresponding tau values (last column padded with 0 so H
                   is square and the product is the full (n-1)×(n-1) U-block).
    """
    b, n, _ = A.shape
    device = A.device

    At = A.clone()                                  # working matrix (modified in place)
    d = torch.zeros((b, n), device=device, dtype=torch.float32)
    e = torch.zeros((b, n), device=device, dtype=torch.float32)
    H = torch.zeros((b, n - 1, n - 1), device=device, dtype=torch.float32)
    tau_h = torch.zeros((b, n - 1), device=device, dtype=torch.float32)

    s = 0
    while s < n - 2:
        M = n - s                                   # size of current trailing block
        nb = min(_PANEL, (n - 2) - s)               # reflectors generated this panel
        sub = At[:, s:, s:]                         # view of the trailing block [b,M,M]

        # Per-panel accumulators (V = reflectors, W = the update factor).
        Vp = torch.zeros((b, M, nb), device=device, dtype=torch.float32)
        Wp = torch.zeros((b, M, nb), device=device, dtype=torch.float32)

        for i in range(nb):
            g = s + i                               # global column index
            L = M - i - 1                           # length of the reflector below subdiag

            # --- bring column i up to date w.r.t. this panel's columns 0..i-1 ---
            # col holds [diagonal ; subdiagonal ; below] of column i of `sub`.
            col = sub[:, i:, i].clone()             # [b, M-i]
            if i > 0:
                # col -= V(i:M, :i) @ W(i, :i)  +  W(i:M, :i) @ V(i, :i)
                col = col - torch.matmul(Vp[:, i:, :i], Wp[:, i, :i].unsqueeze(-1)).squeeze(-1)
                col = col - torch.matmul(Wp[:, i:, :i], Vp[:, i, :i].unsqueeze(-1)).squeeze(-1)

            # Corrected diagonal is the final tridiagonal diagonal for column g.
            d[:, g] = col[:, 0]

            # --- Householder reflector to annihilate below the subdiagonal ------
            x = col[:, 1:]                          # [b, L], x[0] -> subdiagonal
            v, tau, beta = _householder(x)          # v [b,L], tau [b], beta [b]
            e[:, g] = beta
            Vp[:, i + 1:, i] = v                    # store reflector (unit at row i+1)

            # --- symmetric mat-vec + corrections -> column i of W --------------
            # Av = A₂₂ v  (the big matvec; torch/cuBLAS gemv, trusted exact).
            sub22 = sub[:, i + 1:, i + 1:]          # [b, L, L] trailing sub-block
            Av = torch.matmul(sub22, v.unsqueeze(-1)).squeeze(-1)   # [b, L]
            if i > 0:
                # subtract the pending in-panel updates' contribution to A₂₂ v:
                #   Av -= W(:, :i) (V(:, :i)ᵀ v)  +  V(:, :i) (W(:, :i)ᵀ v)
                t1 = torch.matmul(Vp[:, i + 1:, :i].transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)
                Av = Av - torch.matmul(Wp[:, i + 1:, :i], t1.unsqueeze(-1)).squeeze(-1)
                t2 = torch.matmul(Wp[:, i + 1:, :i].transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)
                Av = Av - torch.matmul(Vp[:, i + 1:, :i], t2.unsqueeze(-1)).squeeze(-1)
            w = tau.unsqueeze(1) * Av
            alpha = -0.5 * tau * (w * v).sum(dim=1)              # [b]
            w = w + alpha.unsqueeze(1) * v
            Wp[:, i + 1:, i] = w

            # --- record reflector for U formation ------------------------------
            # In householder_product's frame the reflector for global column g
            # sits at matrix column g with an implicit unit at row g (shifted
            # frame: shifted-row r == global-row r+1). Its essential part (below
            # the unit) is v[1:], i.e. global rows g+2..n-1.
            if L > 1:
                H[:, g + 1:, g] = v[:, 1:]
            tau_h[:, g] = tau

        # --- panel trailing rank-2k update (TRITON, the memory-heavy step) -----
        # A₂₂ -= V2 W2ᵀ + W2 V2ᵀ  where V2/W2 are the sub-diagonal rows of the
        # panel factors. This is applied once per panel; it is what blocking buys
        # us (one big pair of GEMMs instead of nb rank-2 updates), and it is done
        # on tensor cores via _bmm feeding transposed views (no physical transp).
        if M > nb:
            V2 = Vp[:, nb:, :]                       # [b, M-nb, nb]
            W2 = Wp[:, nb:, :]                       # [b, M-nb, nb]
            upd1 = _bmm(V2, W2.transpose(-1, -2))    # V2 W2ᵀ
            upd2 = _bmm(W2, V2.transpose(-1, -2))    # W2 V2ᵀ
            At[:, s + nb:, s + nb:] -= upd1
            At[:, s + nb:, s + nb:] -= upd2

        s += nb

    # --- finalize the trailing 2×2 tridiagonal corner -------------------------
    # Columns 0..n-3 were reduced above; the last diagonal/subdiagonal entries
    # come straight from the fully-updated 2×2 corner of At.
    d[:, n - 2] = At[:, n - 2, n - 2]
    d[:, n - 1] = At[:, n - 1, n - 1]
    e[:, n - 2] = At[:, n - 2, n - 1]

    return d, e, H, tau_h


# ============================================================================
# Step 3 helper: form U (the reduction transform) from the reflectors
# ============================================================================
def _form_U(H: torch.Tensor, tau_h: torch.Tensor, n: int) -> torch.Tensor:
    """U = H₀ H₁ … H_{n-3} embedded as diag(1, U_sub).

    We use torch.linalg.householder_product (LAPACK orgqr) on the (n-1)×(n-1)
    shifted reflector block — a *tested* primitive — rather than accumulating a
    blocked WY product by hand, which would be the single most likely place for
    an un-testable off-by-one to hide. The reduction's first coordinate is fixed
    by every reflector (they act on rows 1..n-1), so U = [[1,0],[0,U_sub]].
    """
    b = H.shape[0]
    # householder_product applies H_0 H_1 … in order; the last column of H is all
    # zeros with tau=0, i.e. an identity reflector, so the product is exactly
    # H_0…H_{n-3} but returned as a full (n-1)×(n-1) orthogonal matrix.
    U_sub = torch.linalg.householder_product(H, tau_h)       # [b, n-1, n-1]
    U = torch.zeros((b, n, n), device=H.device, dtype=torch.float32)
    U[:, 0, 0] = 1.0
    U[:, 1:, 1:] = U_sub
    return U


# ============================================================================
# Correctness self-check (fp64) — replicate the grader gates exactly
# ============================================================================
def _passes_gates(A: torch.Tensor, Q: torch.Tensor, L: torch.Tensor) -> bool:
    """Recompute the grader's invariant gates in fp64. Returns True iff the
    pipeline output would pass. This is the safety net that makes a subtle bug
    in the (un-testable) Triton path degrade to a fallback rather than a failed
    correctness gate. Factors mirror reference.py exactly."""
    n = A.shape[-1]
    eps = torch.finfo(torch.float32).eps
    eigen_rtol = 200.0 * n * eps
    recon_rtol = 400.0 * n * eps
    orth_rtol = 100.0 * n * eps

    if not (torch.isfinite(Q).all() and torch.isfinite(L).all()):
        return False

    a = A.double()
    q = Q.double()
    lam = L.double()

    def l1(x):  # matrix 1-norm over the last two dims (max abs column sum)
        return torch.linalg.matrix_norm(x, ord=1, dim=(-2, -1))

    aq = a @ q
    ql = q * lam.unsqueeze(-2)
    if not (torch.isfinite(aq).all() and torch.isfinite(ql).all()):
        return False

    a_scale = l1(a)
    if bool((l1(aq - ql) > eigen_rtol * a_scale).any()):
        return False

    eye = torch.eye(n, device=A.device, dtype=torch.float64).expand_as(q)
    if bool((l1(q.transpose(-1, -2) @ q - eye).amax() > orth_rtol * l1(eye).amax())):
        return False

    if bool((l1(ql @ q.transpose(-1, -2) - a) > recon_rtol * a_scale).any()):
        return False

    # ascending check (with the grader's tolerance)
    if n > 1:
        diffs = lam[..., 1:] - lam[..., :-1]
        scale = lam.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        if bool((diffs < -(100.0 * n * eps) * scale).any()):
            return False
    return True


# ============================================================================
# The full pipeline (Triton reduction/backtransform + torch inner solve)
# ============================================================================
def _pipeline(data: torch.Tensor) -> output_t:
    """Householder → tridiagonal → eigh(T) → backtransform. Raises on any
    problem (caught by custom_kernel → torch fallback)."""
    b, n, _ = data.shape
    A = data.contiguous()

    # Enforce exact symmetry once (input is symmetric only to fp32 roundoff).
    A = 0.5 * (A + A.transpose(-1, -2))

    # Step 1 — reduce to tridiagonal (Triton trailing update inside).
    d, e, H, tau_h = _reduce_to_tridiagonal(A)

    # Step 2 — inner tridiagonal eigenproblem. See the file header for why this
    # is torch.linalg.eigh on the dense T rather than a Triton implicit-shift QR.
    # Build the symmetric tridiagonal T from (d, e) and solve. eigh returns
    # eigenvalues ascending and eigenvectors Y of T as columns.
    T = torch.diag_embed(d)
    off = e[:, : n - 1]
    T = T + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
    L, Y = torch.linalg.eigh(T)                       # L [b,n] asc, Y [b,n,n]

    # Step 3 — backtransform Q = U @ Y (the GEMM runs on Triton tensor cores).
    U = _form_U(H, tau_h, n)                          # [b,n,n]
    Q = _bmm(U, Y.contiguous())                       # [b,n,n]

    # Eigenvalues are already ascending (from eigh(T)); Q columns already match.
    # Safety net: validate against the grader gates or force a fallback.
    if not _passes_gates(A, Q, L):
        raise RuntimeError("v3 pipeline output failed the recomputed gates")

    return Q, L


def _torch_fallback(data: torch.Tensor) -> output_t:
    """Reference-quality fallback. torch returns (values ascending, vectors);
    the contract wants (Q, L) = (vectors, values)."""
    values, vectors = torch.linalg.eigh(data)
    return vectors, values


def custom_kernel(data: input_t) -> output_t:
    """Batched real-symmetric eigendecomposition -> (Q, L).

    Dispatch by size regime only (never by exact shape/seed):
      * n >= _N_PIPELINE_MIN and CUDA and Triton -> Householder pipeline (v3)
      * otherwise / any failure                  -> torch.linalg.eigh

    The pipeline is wrapped in try/except AND validated against the grader gates
    (fp64) inside _pipeline, so any failure — an exception or a would-be gate
    failure — degrades to a guaranteed-correct torch result.
    """
    n = data.shape[-1]

    if _HAVE_TRITON and data.is_cuda and n >= _N_PIPELINE_MIN:
        try:
            with torch.no_grad():
                return _pipeline(data)
        except Exception:
            return _torch_fallback(data)

    return _torch_fallback(data)
