# Output contract & integrity

## Output contract

`custom_kernel(data) -> (Q, L)` where `data` is `[batch, n, n]` CUDA `float32`:

- `Q`: `[batch, n, n]` `float32`, **columns** are orthonormal eigenvectors.
- `L`: `[batch, n]` `float32`, eigenvalues **sorted ascending**.
- Same convention as `torch.linalg.eigh(A)`. All outputs must be finite.
- Internal low-bit compute (FP16/FP8/etc.) is allowed, but the **returned factors
  must be FP32 and numerically meaningful**.

Correctness is judged on **matrix invariants**, not elementwise agreement with a
reference solver. Individual eigenvector signs may flip and vectors in a
repeated/clustered eigenspace may rotate — that is fine as long as the residual
gates pass:

- eigen-equation `A@Q − Q@diag(L)`
- reconstruction `Q@diag(L)@Qᵀ − A`
- orthogonality `QᵀQ − I`
- eigenvalues ascending

## Integrity — the line between legitimate and gaming

**Allowed (this is real engineering):**
- Genuine eigensolver algorithms (see `claude_docs/`).
- True mathematical fast paths — e.g. a genuinely diagonal input has a trivial
  decomposition; this is a property of the input, not a shape lookup.
- Dispatching by **algorithmic size regime** (`n`, `batch` magnitude), because
  different methods suit different sizes — but see the hard path cap below.

**Forbidden (do not do any of this):**
- Fingerprinting exact benchmark tuples `(batch, n, seed, case)` to shortcut them.
- Hardcoding or precomputing outputs for known cases.
- Reading the checker/reference internals or its state at runtime.
- Any trick that inflates the leaderboard without producing a correct, general
  eigendecomposition.

The deliverable is a **real, robust eigensolver** that would work on unseen inputs
of the same kind.

## Dispatch path cap — HARD MAX 5

The kernel may route inputs to **at most 5 distinct code paths** (algorithmic
branches), and this **includes any mathematical fast paths** (e.g. a diagonal
fast path counts as one). **5 is a hard maximum — never more. Fewer is better.**

This exists for two reasons you must respect:
- **Feasibility** — a handful of well-chosen regime paths stays a usable,
  maintainable kernel; a dozen does not.
- **Anti-overfitting** — an uncapped dispatcher quietly becomes per-shape
  fingerprinting (one specialized branch per benchmark case), which is forbidden.
  Paths must be keyed to **size regime** (`n`/`batch` magnitude), not to specific
  benchmark shapes, and there must be ≤ 5 of them.

If you think you need a 6th path, you're overfitting — consolidate regimes instead.
