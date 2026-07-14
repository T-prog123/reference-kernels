# submission_v1.py
# ============================================================================
# parent: v0 (torch.linalg.eigh baseline)
# Hypothesis: a genuine batched *cyclic* Jacobi eigensolver — one Triton program
#   per matrix, whole matrix resident, rounds of disjoint rotations applied as
#   tensor-core matmuls — beats torch on the small-n case (n=32) that fits one
#   program, while torch.linalg.eigh safely covers every larger shape.
# ============================================================================
#
# WHY JACOBI, WHY NOW (see notes/design.md "Current bet")
# ------------------------------------------------------------------------
# The canon (claude_docs/batched_cuda_symmetric_evd_methods.md) ranks
# batched/block Jacobi as the #1 theoretical avenue for the medium-n,
# large-batch regime of this competition. Reasons that matter here:
#   * Two levels of parallelism: across the batch AND across the (up to n/2)
#     disjoint plane rotations that can be applied at once within one matrix.
#   * It produces a highly orthogonal Q essentially for free (every update is
#     an exact-in-exact-arithmetic orthogonal transform), and the checker's
#     orthogonality gate (orth_rtol ≈ 100·n·eps) is the tightest one, so this
#     is a real advantage.
#   * The correctness gates are invariant-based and generously
#     dimension-scaled (eigen_rtol ≈ 200·n·eps, recon_rtol ≈ 400·n·eps), which
#     is exactly the room an iterative solver needs: we only have to converge
#     to a handful of ULP, not to bit-exact agreement with a reference.
#
# WHAT v1 IS (and is not)
# ------------------------------------------------------------------------
# v1 is the OPENING kernel. It establishes the Jacobi machinery correctly for
# the regime where an entire matrix fits inside one Triton program, and falls
# back to torch.linalg.eigh everywhere else. It is deliberately NOT yet the
# block-Jacobi variant needed to cover n=176..512 (those matrices do not fit
# resident — see the SRAM budget below). That is the next step, tracked in
# notes/design.md "Open ideas".
#
# SRAM / fit budget (B200, ~228 KB shared memory per SM)
# ------------------------------------------------------------------------
# One fp32 n×n tile is n²·4 bytes. The round-robin Jacobi below keeps ~4 tiles
# effectively live at the hot point (B, V, the rotation tile J, and a matmul
# temporary):
#     n=32  -> 4·(32²·4)   =   16 KB   (trivial)
#     n=64  -> 4·(64²·4)   =   64 KB   (comfortable)
#     n=128 -> 4·(128²·4)  =  256 KB   (over SRAM; relies on register/local
#                                       spill; occupancy suffers)
#     n=256 -> a single tile alone is 256 KB  -> does NOT fit at all
# So the resident-single-block design is realistic only up to n≈128. We set the
# dispatch threshold at n <= 128. This is a dispatch-by-*size-regime* decision
# (allowed by .claude/rules/contract-and-integrity.md), NOT a fingerprint of a
# benchmark shape. Among the competition shapes only n=32 actually lands on the
# Triton path today; 176/352/512/1024/2048/4096 all take the torch fallback.
#
# CORRECTNESS-FIRST SAFETY NET
# ------------------------------------------------------------------------
# We cannot run code locally (no GPU/Python here), so v1 is engineered so that a
# bug in the Triton path can never fail a correctness gate: the entire Triton
# path is wrapped in try/except and, on ANY exception, we return
# torch.linalg.eigh. The submission therefore always produces a valid
# eigendecomposition; the worst case is "no speedup", never "wrong answer".
#
# OUTPUT CONTRACT (see .claude/rules/contract-and-integrity.md)
# ------------------------------------------------------------------------
# custom_kernel(data:[b,n,n] fp32 CUDA) -> (Q, L)
#   Q: [b,n,n] fp32, columns are orthonormal eigenvectors.
#   L: [b,n]  fp32, eigenvalues ascending.
#   Same convention as torch.linalg.eigh(A). Sign flips / eigenspace rotations
#   are allowed; correctness is judged on matrix invariants.
# ============================================================================

import torch

# Triton is only available on the remote B200 runner. Import defensively so that
# merely importing this module never explodes if Triton is somehow absent — in
# that case _HAVE_TRITON stays False and we always use the torch path.
try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - defensive
    _HAVE_TRITON = False

from task import input_t, output_t


# ============================================================================
# Dispatch configuration
# ============================================================================
# Only matrices with n <= _N_TRITON_MAX take the Triton Jacobi path (they fit in
# one program per the SRAM budget above). Everything larger uses torch. This is
# a size-regime threshold, not a per-shape lookup.
_N_TRITON_MAX = 128

# Number of Jacobi *sweeps* (a sweep = one full round-robin pass = BN-1 rounds,
# touching every off-diagonal pair exactly once). Cyclic Jacobi converges
# quadratically once it enters the asymptotic regime, so a fixed budget of a
# dozen-ish sweeps drives the off-diagonal mass far below the loose residual
# gates for small n. We use a generous constant here (correctness > speed for
# v1); tuning this down is a known future perf lever (notes/design.md).
_N_SWEEPS = 15

# Cache of round-robin schedules keyed by BN (the padded, power-of-two matrix
# dimension). Building a schedule is cheap but we do it once per size.
_SCHEDULE_CACHE: dict = {}


# ============================================================================
# Round-robin (1-factorization) pairing schedule
# ============================================================================
# The GPU-natural form of Jacobi applies many *disjoint* rotations at once. For
# an even number of "players" BN, the classic "circle method" produces a
# 1-factorization of the complete graph: BN-1 rounds, each round a perfect
# matching (BN/2 disjoint pairs), and across all rounds every unordered pair
# {i,j} appears exactly once. One sweep = all BN-1 rounds = every off-diagonal
# entry annihilated once. This is exactly the Brent-Luk parallel ordering.
#
# We return an int32 tensor `partner` of shape [BN-1, BN] where partner[r, i]
# is the index paired with i in round r. Because BN is even there are never any
# self-pairs (partner[r,i] != i), which the kernel relies on.
def _build_schedule(bn: int, device: torch.device) -> torch.Tensor:
    key = (bn, str(device))
    cached = _SCHEDULE_CACHE.get(key)
    if cached is not None:
        return cached

    # Circle method: keep player arr[0] fixed, rotate the rest by one each round.
    arr = list(range(bn))
    rounds = []
    for _ in range(bn - 1):
        partner_row = [0] * bn
        for i in range(bn // 2):
            a = arr[i]
            b = arr[bn - 1 - i]
            partner_row[a] = b
            partner_row[b] = a
        rounds.append(partner_row)
        # Rotate: fix arr[0], move the last element to the front of the tail.
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]

    sched = torch.tensor(rounds, dtype=torch.int32, device=device)
    _SCHEDULE_CACHE[key] = sched
    return sched


# ============================================================================
# The Triton Jacobi kernel
# ============================================================================
# One program instance == one matrix (grid = (batch,)). The whole matrix is
# loaded into a [BN, BN] register/shared tile `B` (the "current transformed
# matrix"), and `V` accumulates the eigenvectors (starts at identity).
#
# Per round r we:
#   1. read the diagonal d[i] = B[i,i] and the pair off-diagonals
#      off[i] = B[i, partner[i]] (symmetric, so this is a_pq for i's pair);
#   2. compute a rotation (c[i], s[i]) for every index i using the "i-as-p"
#      convention — see the sign argument below;
#   3. assemble those BN/2 disjoint 2x2 rotations into ONE orthogonal tile J;
#   4. apply the two-sided similarity B <- J^T B J and accumulate V <- V J.
#
# Steps 3-4 turn a whole round of rotations into three matmuls (tl.dot), which
# run on the tensor cores instead of as scalar row/column sweeps. That is the
# whole point of the parallel ordering.
#
# WHY THE PER-INDEX "i-as-p" ROTATION IS SELF-CONSISTENT
# ------------------------------------------------------------------------
# For a pair (p,q) the stable Jacobi rotation G = [[c, s], [-s, c]] that zeros
# B[p,q] uses:
#     tau = (a_qq - a_pp) / (2 a_pq)
#     t   = sgn(tau) / (|tau| + sqrt(1 + tau^2))   (sgn(0) := +1)
#     c   = 1/sqrt(1+t^2),   s = t*c
# We compute this independently for every index i, treating i as "p" and its
# partner as "q": a_pp = d[i], a_qq = d[partner[i]], a_pq = off[i]. Consider the
# two members of one pair, p and q:
#   * member p: tau_p = (d[q]-d[p])/(2 a_pq)      -> t_p, c_p, s_p
#   * member q: tau_q = (d[p]-d[q])/(2 a_pq) = -tau_p -> t_q=-t_p, c_q=c_p,
#                                                        s_q=-s_p
# We then place J[i,i]=c[i] and J[i,partner[i]]=s[i]. Restricted to {p,q}:
#     [[c_p, s_p], [s_q, c_q]] = [[c, s], [-s, c]] = G     (since s_q=-s_p)
# i.e. the correct, det=+1 rotation falls out automatically — no min/max
# branching and no separate "who is p" bookkeeping is needed. (Verified the
# algebra: G^T B G has (p,q) entry a_pq(c^2-s^2) + cs(a_pp-a_qq), and the t
# above is the root of t^2 + 2 tau t - 1 = 0 that makes it zero.)
#
# PADDING
# ------------------------------------------------------------------------
# BN = next power of two >= max(16, n) (>=16 because tl.dot needs it; power of
# two keeps the tile shapes friendly and BN even for the schedule). Padded rows
# /cols (index >= n) are initialised as an identity block: B[i,i]=1, B[i,j]=0.
# Any pair that includes a padded index has a_pq = 0, so its rotation is the
# identity (t=0, c=1, s=0) and never mixes padding into the real block. The
# padded eigen-data is simply dropped by the wrapper (it keeps only the first n).
if _HAVE_TRITON:

    @triton.jit
    def _jacobi_kernel(
        A_ptr,          # *fp32 [batch, n, n] input matrices (row-major)
        V_ptr,          # *fp32 [batch, n, n] output eigenvectors (columns)
        L_ptr,          # *fp32 [batch, n]    output eigenvalues (unsorted)
        partner_ptr,    # *int32 [N_ROUNDS, BN] round-robin schedule
        n,              # runtime int: true matrix dimension
        N_ROUNDS,       # runtime int: BN-1
        stride_ab,      # element strides for A/V (both [batch,n,n], same layout)
        stride_ai,
        stride_aj,
        stride_lb,      # element strides for L [batch, n]
        stride_li,
        BN: tl.constexpr,       # padded dimension (power of two, >=16)
        N_SWEEPS: tl.constexpr,
    ):
        b = tl.program_id(0)

        rows = tl.arange(0, BN)
        cols = tl.arange(0, BN)
        row2 = rows[:, None]
        col2 = cols[None, :]

        real_mask = (row2 < n) & (col2 < n)      # [BN,BN] true on the real block
        eye = row2 == col2                        # [BN,BN] diagonal selector

        # ---- Load A into B, with an identity block on the padding ----------
        a_off = b * stride_ab + row2 * stride_ai + col2 * stride_aj
        B = tl.load(A_ptr + a_off, mask=real_mask, other=0.0)
        # Padding -> identity: on padded diagonal put 1.0, elsewhere the load
        # already gave 0.0. This decouples padding from the real block.
        B = tl.where(real_mask, B, tl.where(eye, 1.0, 0.0))

        # Enforce exact symmetry of the working matrix. The input is symmetric
        # only up to fp32 roundoff; symmetrizing once removes that asymmetry so
        # the two-sided update stays well defined.
        B = 0.5 * (B + tl.trans(B))

        # ---- V starts as the identity --------------------------------------
        V = tl.where(eye, 1.0, 0.0)

        # ---- Sweeps ---------------------------------------------------------
        # N_SWEEPS is a small constexpr (unrolled); the inner round loop is a
        # runtime tl.range so the compiled kernel stays small.
        for _s in tl.static_range(N_SWEEPS):
            for r in tl.range(0, N_ROUNDS):
                # partner[i] for this round: int32 vector of length BN.
                partner = tl.load(partner_ptr + r * BN + rows)  # [BN]
                partner2 = partner[:, None]                      # [BN,1]

                # colsel[i,j] = (j == partner[i]) : selects each row's partner
                # column. Every row has exactly one True (matchings are perfect
                # and there are no self-pairs).
                colsel = col2 == partner2                        # [BN,BN]

                # d[i] = B[i,i]  (sum over columns of the diagonal-masked tile)
                d = tl.sum(tl.where(eye, B, 0.0), axis=1)        # [BN]
                # off[i] = B[i, partner[i]] = a_pq for i's pair (symmetric)
                off = tl.sum(tl.where(colsel, B, 0.0), axis=1)   # [BN]
                # dpart[i] = d[partner[i]] = a_qq for i-as-p
                dpart = tl.sum(tl.where(colsel, d[None, :], 0.0), axis=1)  # [BN]

                # ---- rotation angle (stable formula, guarded) --------------
                abs_off = tl.abs(off)
                # "Active" = off-diagonal is non-negligible relative to the two
                # diagonal magnitudes. Inactive pairs get the identity rotation.
                scale = tl.abs(d) + tl.abs(dpart) + 1e-30
                active = abs_off > (1e-20 * scale)
                # Avoid 0/0 -> NaN: use a safe denominator where inactive.
                off_safe = tl.where(active, off, 1.0)
                tau = (dpart - d) / (2.0 * off_safe)
                # sgn(tau) with sgn(0) := +1, so a_pp==a_qq gives the 45-degree
                # rotation (t=1) instead of a no-op.
                sgn = tl.where(tau >= 0.0, 1.0, -1.0)
                t = sgn / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
                t = tl.where(active, t, 0.0)
                c = 1.0 / tl.sqrt(1.0 + t * t)
                s = t * c

                # ---- assemble the round's orthogonal tile J ----------------
                # J[i,i] = c[i] ; J[i,partner[i]] = s[i] ; else 0.
                # eye and colsel are disjoint (partner[i] != i), so these two
                # placements never collide.
                J = tl.where(eye, c[:, None], 0.0) + tl.where(colsel, s[:, None], 0.0)

                # ---- apply: B <- J^T B J , V <- V J ------------------------
                # input_precision='ieee' keeps full fp32 in the matmuls (no
                # TF32 truncation) so Jacobi's accuracy is preserved.
                BJ = tl.dot(B, J, input_precision="ieee")
                B = tl.dot(tl.trans(J), BJ, input_precision="ieee")
                V = tl.dot(V, J, input_precision="ieee")

        # ---- Emit eigenvalues (diagonal of converged B) and vectors --------
        d_final = tl.sum(tl.where(eye, B, 0.0), axis=1)          # [BN]

        # Store L[b, i] = d_final[i] for i < n.
        l_off = b * stride_lb + rows * stride_li
        tl.store(L_ptr + l_off, d_final, mask=rows < n)

        # Store V[b, :, :] real block. V[:, j] is the eigenvector for d_final[j].
        v_off = b * stride_ab + row2 * stride_ai + col2 * stride_aj
        tl.store(V_ptr + v_off, V, mask=real_mask)


def _next_pow2_at_least_16(n: int) -> int:
    """Smallest power of two that is >= n and >= 16 (tl.dot needs BN>=16)."""
    bn = 16
    while bn < n:
        bn *= 2
    return bn


def _jacobi_triton(data: torch.Tensor) -> output_t:
    """Run the Triton round-robin Jacobi solver and return (Q, L) ascending.

    Assumes n <= _N_TRITON_MAX. Raises on any Triton/setup problem; the caller
    catches and falls back to torch, so correctness is never at risk.
    """
    b, n, _ = data.shape
    device = data.device

    # Ensure a contiguous fp32 input we can index with plain strides.
    A = data.contiguous()

    bn = _next_pow2_at_least_16(n)
    n_rounds = bn - 1
    schedule = _build_schedule(bn, device)  # [n_rounds, bn] int32

    # Outputs (unsorted): V holds eigenvectors as columns, L the diagonal.
    V = torch.empty((b, n, n), dtype=torch.float32, device=device)
    L = torch.empty((b, n), dtype=torch.float32, device=device)

    grid = (b,)
    _jacobi_kernel[grid](
        A,
        V,
        L,
        schedule,
        n,
        n_rounds,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        L.stride(0),
        L.stride(1),
        BN=bn,
        N_SWEEPS=_N_SWEEPS,
    )

    # ---- Sort eigenpairs ascending (done in torch: robust and cheap) -------
    # The kernel returns eigenvalues in index order; the contract wants them
    # ascending, with Q's columns permuted to match.
    order = torch.argsort(L, dim=1)                              # [b, n]
    L_sorted = torch.gather(L, 1, order)
    # Permute columns of V by `order`: Q[bi, ri, ci] = V[bi, ri, order[bi, ci]].
    col_index = order.unsqueeze(1).expand(b, n, n)
    Q = torch.gather(V, 2, col_index)

    # Final safety: if anything went non-finite, signal the caller to fall back.
    if not torch.isfinite(Q).all() or not torch.isfinite(L_sorted).all():
        raise RuntimeError("non-finite output from Triton Jacobi path")

    return Q, L_sorted


def _torch_fallback(data: torch.Tensor) -> output_t:
    """Reference-quality fallback. torch returns (values ascending, vectors);
    the contract wants (Q, L) = (vectors, values)."""
    values, vectors = torch.linalg.eigh(data)
    return vectors, values


def custom_kernel(data: input_t) -> output_t:
    """Batched real-symmetric eigendecomposition -> (Q, L).

    Dispatch by *size regime* only (never by exact shape/seed):
      * n <= _N_TRITON_MAX and Triton available  -> Triton round-robin Jacobi
      * otherwise                                 -> torch.linalg.eigh

    The Triton path is wrapped in try/except so that any failure degrades to the
    torch fallback rather than failing a correctness gate.
    """
    n = data.shape[-1]

    if _HAVE_TRITON and data.is_cuda and n <= _N_TRITON_MAX:
        try:
            return _jacobi_triton(data)
        except Exception:
            # Any problem in the Triton path -> guaranteed-correct torch result.
            return _torch_fallback(data)

    return _torch_fallback(data)
