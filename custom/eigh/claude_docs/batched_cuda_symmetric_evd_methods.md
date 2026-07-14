# Dense Symmetric EVD on Batched CUDA: Four Algorithm Families

## Target Problem

Given a batch of real symmetric matrices:

```text
A[b] ∈ R^{n×n}, A[b] = A[b].T
```

Compute:

```text
A[b] @ Q[b] ≈ Q[b] @ diag(L[b])
```

where:

- `Q[b]` is `n × n` and has orthonormal columns.
- `L[b]` is length `n`.
- `L[b]` contains eigenvalues sorted ascending.
- `Q[b]` contains eigenvectors as columns.

Also required:

```text
Q[b].T @ Q[b] ≈ I
A[b] ≈ Q[b] @ diag(L[b]) @ Q[b].T
```

Batching does **not** change the mathematical eigendecomposition problem. Each matrix has an independent eigendecomposition:

```text
A_b Q_b = Q_b Λ_b
```

The batch dimension mainly changes GPU scheduling, memory layout, parallelism, and convergence management.

---

# Method 1: Jacobi Rotations

## Summary

Jacobi eigendecomposition diagonalizes a symmetric matrix by repeatedly applying orthogonal plane rotations.

It is one of the most GPU-relevant approaches for batched small/medium dense symmetric matrices.

For batched CUDA EVD, Jacobi is important because:

- each matrix is independent;
- many rotations can be applied across the batch;
- the method produces highly orthogonal eigenvectors;
- it maps naturally to SIMT-style parallelism.

For the competition shapes, Jacobi is especially relevant for:

- `n = 32`
- `n = 176`
- `n = 352`
- `n = 512`

It may still be used for `n = 1024`, but the number of rotations/sweeps becomes expensive.

---

## Jacobi: Mathematical Goal

For one symmetric matrix `A`:

```text
A = A.T
```

We want:

```text
A Q = Q Λ
```

Equivalently:

```text
Q.T A Q = Λ
```

Jacobi builds `Q` as a product of rotations:

```text
Q = G_1 G_2 ... G_k
```

and updates:

```text
B_k = Q_k.T A Q_k
```

The goal is to make `B_k` diagonal.

At convergence:

```text
B_k ≈ Λ
A Q_k ≈ Q_k Λ
```

---

## Jacobi: One Rotation

Maintain:

```text
B = current transformed matrix
Q = accumulated eigenvectors
```

Initially:

```text
B = A
Q = I
```

Pick two indices `p` and `q`.

Look at the `2 × 2` submatrix:

```text
[ b_pp   b_pq ]
[ b_pq   b_qq ]
```

Choose a rotation `G(p, q, θ)`:

```text
[  c   s ]
[ -s   c ]
```

embedded into the full `n × n` identity.

Apply the similarity transform:

```text
B_new = G.T @ B @ G
```

Choose `c` and `s` so that:

```text
B_new[p, q] = 0
```

Also update eigenvectors:

```text
Q_new = Q @ G
```

Because `G` is orthogonal, `Q` remains orthogonal.

---

## Jacobi: Stable Rotation Formula

Given:

```text
a = B[p, p]
b = B[p, q]
d = B[q, q]
```

If `b` is already small, skip.

Otherwise:

```text
tau = (d - a) / (2b)
t   = sign(tau) / (abs(tau) + sqrt(1 + tau^2))
c   = 1 / sqrt(1 + t^2)
s   = t * c
```

Then use `c` and `s` to update rows/columns `p` and `q`.

---

## Jacobi: Pseudocode

```python
def jacobi_eigh(A, max_sweeps, tol):
    B = A.copy()
    Q = eye(n)

    for sweep in range(max_sweeps):
        offdiag_norm = compute_offdiag_norm(B)

        if offdiag_norm < tol:
            break

        for p, q in rotation_schedule(n):
            if abs(B[p, q]) < tol:
                continue

            c, s = compute_jacobi_rotation(
                B[p, p],
                B[p, q],
                B[q, q],
            )

            # B <- G.T @ B @ G
            apply_rotation_to_B(B, p, q, c, s)

            # Q <- Q @ G
            apply_rotation_to_Q(Q, p, q, c, s)

    L = diag(B)

    Q, L = sort_eigenpairs_ascending(Q, L)

    return Q, L
```

---

## Batched Jacobi

For batch input:

```text
A.shape = [batch, n, n]
```

Batched Jacobi conceptually does:

```python
def batched_jacobi_eigh(A_batch):
    B_batch = A_batch.copy()
    Q_batch = batched_identity(batch, n)

    for sweep in range(max_sweeps):
        for p, q in rotation_schedule(n):
            parallel_for b in range(batch):
                c, s = compute_rotation(B_batch[b], p, q)
                apply_rotation_to_B(B_batch[b], p, q, c, s)
                apply_rotation_to_Q(Q_batch[b], p, q, c, s)

    L_batch = diagonal(B_batch)
    Q_batch, L_batch = sort_all_batches(Q_batch, L_batch)

    return Q_batch, L_batch
```

Batching does not change the eigenproblem.

It changes execution:

- one matrix may not saturate the GPU;
- many matrices give enough work;
- each matrix may converge in a different number of sweeps;
- if using fixed sweeps, some matrices get extra unnecessary work;
- if using early stopping, branches/divergence appear.

---

## Block Jacobi

Scalar Jacobi rotates one coordinate pair `(p, q)`.

Block Jacobi rotates block pairs.

Instead of diagonalizing:

```text
[ b_pp   b_pq ]
[ b_pq   b_qq ]
```

it diagonalizes or updates a larger block-pair submatrix:

```text
[ B_ii   B_ij ]
[ B_ji   B_jj ]
```

where each `B_ij` is a tile.

Conceptual block Jacobi:

```python
for sweep in range(max_sweeps):
    for block_i, block_j in block_pair_schedule:
        S = extract_block_pair(B, block_i, block_j)

        # Solve a smaller local EVD/SVD-like problem.
        U = local_orthogonal_transform(S)

        # Apply block similarity transform.
        B = apply_block_rotation(B, block_i, block_j, U)

        # Accumulate global eigenvectors.
        Q = apply_block_rotation(Q, block_i, block_j, U)
```

Block Jacobi is more relevant for `n = 512` than pure scalar Jacobi because it creates tiled matrix operations.

---

## Jacobi: Strengths and Weaknesses

Strengths:

- very parallel;
- naturally batched;
- excellent orthogonality;
- good for small/medium matrices;
- strong fit for GPU execution.

Weaknesses:

- iterative;
- clustered eigenvalues may require many sweeps;
- convergence differs across batch elements;
- large `n` becomes expensive;
- global-memory traffic can dominate unless blocked/tiled carefully.

Competition relevance:

```text
n=32:    very strong
n=176:   strong
n=352:   strong
n=512:   very relevant, probably block/batched Jacobi
n=1024:  possible but expensive
n=2048+: less likely to be best
```

---

# Method 2: Householder Tridiagonalization Pipeline

## Summary

This is the classical dense symmetric eigensolver pipeline used by LAPACK-style algorithms.

Pipeline:

1. Reduce dense symmetric `A` to tridiagonal `T`.
2. Solve the tridiagonal eigenproblem.
3. Backtransform tridiagonal eigenvectors to eigenvectors of `A`.

This is usually the strongest theoretical approach for large dense matrices.

For the competition, this becomes more relevant as `n` grows:

- `n = 1024`: relevant
- `n = 2048`: very relevant
- `n = 4096`: very relevant, if present

---

## Tridiagonal Pipeline: Mathematical Goal

Given:

```text
A = A.T
```

Find an orthogonal matrix `U` such that:

```text
T = U.T @ A @ U
```

where `T` is tridiagonal.

Then solve:

```text
T @ Y = Y @ Λ
```

Recover original eigenvectors:

```text
Q = U @ Y
```

Check:

```text
A @ Q
= A @ U @ Y
= U @ T @ Y
= U @ Y @ Λ
= Q @ Λ
```

So `Q` contains eigenvectors of `A`.

---

## Step 1: Householder Reduction

Use Householder reflectors:

```text
H_1, H_2, ..., H_{n-2}
```

Each `H_k` zeros entries below the first subdiagonal in column `k`.

The accumulated transform is:

```text
U = H_1 H_2 ... H_{n-2}
```

The transformed matrix is:

```text
T = U.T @ A @ U
```

`T` has the form:

```text
[ d1  e1   0   0  ... ]
[ e1  d2  e2   0  ... ]
[  0  e2  d3  e3  ... ]
[  0   0  e3  d4  ... ]
[ ...                ]
```

Only two vectors are needed to store `T`:

```text
d = diagonal entries, length n
e = off-diagonal entries, length n-1
```

---

## Step 2: Solve Tridiagonal Eigenproblem

After reduction, solve:

```text
T @ Y = Y @ Λ
```

This tridiagonal stage may use:

- shifted QR iteration;
- divide-and-conquer;
- MRRR;
- bisection + inverse iteration.

The output is:

```text
Y = eigenvectors of T
L = eigenvalues of T
```

Because `T` is orthogonally similar to `A`, `T` and `A` have the same eigenvalues.

---

## Step 3: Backtransform Eigenvectors

Given:

```text
T = U.T @ A @ U
T @ Y = Y @ Λ
```

Then:

```text
Q = U @ Y
```

So:

```text
A @ Q = Q @ Λ
```

This is the final expensive step when eigenvectors are required.

If only eigenvalues were needed, backtransform could be skipped.

But this competition requires `Q`, so backtransform is mandatory for this method.

---

## Tridiagonal Pipeline: Pseudocode

```python
def tridiagonal_pipeline_eigh(A):
    # Step 1:
    # Reduce dense symmetric A to tridiagonal T.
    U, d, e = householder_tridiagonalize(A)

    # Step 2:
    # Solve tridiagonal eigenproblem.
    Y, L = tridiagonal_eigh(d, e)

    # Step 3:
    # Convert tridiagonal eigenvectors back to original-space eigenvectors.
    Q = U @ Y

    Q, L = sort_eigenpairs_ascending(Q, L)

    return Q, L
```

---

## Batched Tridiagonal Pipeline

Conceptually:

```python
def batched_tridiagonal_pipeline(A_batch):
    for b in parallel:
        U[b], d[b], e[b] = householder_tridiagonalize(A_batch[b])

    for b in parallel:
        Y[b], L[b] = tridiagonal_eigh(d[b], e[b])

    for b in parallel:
        Q[b] = U[b] @ Y[b]

    return Q, L
```

Batching does not change the math.

But it makes scheduling harder:

- each matrix has its own Householder reduction;
- tridiagonal solvers may converge at different rates;
- backtransform is large matrix multiplication per batch element;
- workspace per matrix can be large;
- pipeline has multiple phases.

---

## Tridiagonal Pipeline: Strengths and Weaknesses

Strengths:

- classical robust method;
- strong for large dense matrices;
- mature algorithms;
- good accuracy;
- common in LAPACK/MAGMA/cuSOLVER-style solvers.

Weaknesses:

- multi-stage;
- Householder reduction contains memory-bound operations;
- less naturally batched than Jacobi;
- backtransform is expensive;
- implementation is complex.

Competition relevance:

```text
n=32:    probably overkill
n=176:   probably not ideal
n=352:   maybe, but likely overkill
n=512:   possible, but batched Jacobi may be better
n=1024:  relevant
n=2048+: strong candidate
```

---

# Method 3: Shifted QR Iteration

## Summary

QR iteration is a classical eigenvalue algorithm.

For dense symmetric EVD, QR is usually not applied directly to the full dense matrix.

Instead, it is commonly used after tridiagonalization:

```text
A dense symmetric
→ tridiagonal T
→ shifted QR on T
→ eigenvalues/eigenvectors of T
→ backtransform to eigenvectors of A
```

So QR is best viewed as an inner tridiagonal eigensolver.

---

## QR Iteration: Mathematical Idea

Given current matrix `T_k`, choose shift `μ`.

Compute QR factorization:

```text
T_k - μI = Q_k R_k
```

Then form:

```text
T_{k+1} = R_k Q_k + μI
```

This is equivalent to:

```text
T_{k+1} = Q_k.T @ T_k @ Q_k
```

So `T_{k+1}` is orthogonally similar to `T_k`.

Therefore, eigenvalues are preserved.

Over many iterations, `T_k` becomes diagonal or block diagonal.

For symmetric tridiagonal `T`, the off-diagonal entries shrink until deflation is possible.

---

## QR: Deflation

For tridiagonal `T`, if an off-diagonal entry `e_i` becomes tiny:

```text
abs(e_i) ≈ 0
```

then `T` can be split:

```text
T =
[ T_1   0  ]
[  0   T_2 ]
```

Now solve `T_1` and `T_2` independently.

This is called deflation.

It reduces problem size and improves performance.

---

## QR: Eigenvectors

If eigenvectors are needed, accumulate all orthogonal QR step matrices.

For tridiagonal `T`:

```text
Z = I
```

Each QR step gives `Q_k`.

Update:

```text
Z = Z @ Q_k
```

At convergence:

```text
T_final ≈ Λ
```

`Z` contains eigenvectors of the original tridiagonal `T`.

If this QR solver is inside the dense pipeline:

```text
Q_dense = U @ Z
```

where `U` came from Householder tridiagonalization.

---

## QR Iteration: Pseudocode

```python
def tridiagonal_qr_eigh(d, e):
    # d: diagonal of tridiagonal T
    # e: off-diagonal of tridiagonal T

    T = form_tridiagonal(d, e)
    Z = eye(n)

    while not converged(T):
        mu = choose_shift(T)

        # QR factorization of shifted matrix.
        Q_step, R_step = qr(T - mu * eye(n))

        # Similarity update.
        T = R_step @ Q_step + mu * eye(n)

        # Accumulate eigenvectors.
        Z = Z @ Q_step

        # Split problem if off-diagonals become small.
        deflate_if_possible(T)

    L = diag(T)

    Z, L = sort_eigenpairs_ascending(Z, L)

    return Z, L
```

Real implementations exploit tridiagonal structure and use Givens rotations instead of dense QR.

---

## Batched QR

Batched QR-style EVD means:

```python
for b in parallel:
    run shifted QR on T[b]
```

The issue:

- each matrix may deflate differently;
- each matrix may need different iteration counts;
- convergence divergence is common;
- small tridiagonal QR is hard to keep efficient on GPU unless grouped carefully.

Batching does not create new QR theory.

It creates a scheduling problem:

- group similar active subproblems;
- avoid wasting work on already-converged matrices;
- avoid too many tiny kernels;
- keep memory accesses regular.

---

## QR: Strengths and Weaknesses

Strengths:

- classical and robust;
- very important for tridiagonal eigenproblems;
- good eigenvalue accuracy;
- deflation reduces work.

Weaknesses:

- sequential convergence behavior;
- not as naturally parallel as Jacobi;
- batching causes divergence;
- direct dense QR iteration is not the usual high-performance route.

Competition relevance:

- rarely the top-level method;
- important as part of tridiagonal pipeline;
- more relevant for `n = 1024+` than `n = 512` batched cases.

---

# Method 4: Divide-and-Conquer on Tridiagonal Matrix

## Summary

Divide-and-conquer is another classical tridiagonal eigensolver.

It is usually used after Householder tridiagonalization.

Pipeline:

```text
A dense symmetric
→ tridiagonal T
→ divide-and-conquer solve T
→ backtransform eigenvectors to original space
```

This method is strong when all eigenvectors are required.

---

## Divide-and-Conquer: Mathematical Idea

Given tridiagonal `T`, split it into two smaller tridiagonal matrices plus a rank-one correction.

Example:

```text
T =
[ T1   0  ]
[  0  T2 ] + ρ v v.T
```

where:

- `T1` is the top-left subproblem;
- `T2` is the bottom-right subproblem;
- `ρ v v.T` reconnects the two pieces.

Then:

1. solve eigendecomposition of `T1`;
2. solve eigendecomposition of `T2`;
3. merge the two solutions using the rank-one update.

---

## Divide-and-Conquer: Recursive Structure

For tridiagonal `T`:

```python
def dc_tridiagonal_eigh(T):
    if size(T) is small:
        return small_eigh(T)

    T1, T2, rho, v = split_tridiagonal(T)

    Y1, L1 = dc_tridiagonal_eigh(T1)
    Y2, L2 = dc_tridiagonal_eigh(T2)

    Y, L = merge_rank_one_update(Y1, L1, Y2, L2, rho, v)

    return Y, L
```

The recursion creates many smaller eigenproblems.

---

## Divide-and-Conquer: Merge Step

After solving the subproblems, we have:

```text
T1 = Y1 Λ1 Y1.T
T2 = Y2 Λ2 Y2.T
```

The combined block diagonal eigensystem is:

```text
D = diag(Λ1, Λ2)
```

Then the full matrix is:

```text
D + ρ z z.T
```

The merge step solves eigenvalues of this rank-one modified diagonal matrix.

The eigenvalues are found from the secular equation.

After new eigenvalues are found, eigenvectors are reconstructed from the rank-one update formula.

---

## Dense Symmetric EVD Using Divide-and-Conquer

For dense `A`:

```python
def dense_eigh_via_dc(A):
    U, d, e = householder_tridiagonalize(A)

    Y, L = divide_and_conquer_tridiagonal_eigh(d, e)

    Q = U @ Y

    Q, L = sort_eigenpairs_ascending(Q, L)

    return Q, L
```

Again:

- D&C solves the tridiagonal problem;
- Householder reduction handles dense-to-tridiagonal;
- backtransform produces eigenvectors of original `A`.

---

## Batched Divide-and-Conquer

Conceptually:

```python
for b in parallel:
    U[b], d[b], e[b] = householder_tridiagonalize(A[b])

for b in parallel:
    Y[b], L[b] = divide_and_conquer_tridiagonal_eigh(d[b], e[b])

for b in parallel:
    Q[b] = U[b] @ Y[b]
```

Batching issues:

- recursion shapes may differ across matrices;
- deflation may differ across matrices;
- merge work may be irregular;
- workspace management is more complex than Jacobi;
- better for fewer larger matrices than many medium ones.

---

## Divide-and-Conquer: Strengths and Weaknesses

Strengths:

- strong for full eigenvectors;
- good for large dense matrices;
- often faster than QR for all eigenvectors;
- mature in LAPACK/MAGMA-style dense symmetric EVD.

Weaknesses:

- complicated;
- memory/workspace heavy;
- irregular recursion;
- batching can be awkward;
- still needs dense tridiagonalization and backtransform.

Competition relevance:

```text
n=512:       maybe not ideal as top choice
n=1024:      relevant
n=2048+:     strong candidate
batch=640:   less natural than Jacobi
batch=8/60:  more plausible for large n
```

---

# Batch Dimension: What Actually Changes?

The math does not change:

```text
A_b Q_b = Q_b Λ_b
```

But execution changes strongly.

Batching improves:

- occupancy;
- amortized launch overhead;
- throughput;
- ability to saturate GPU with many medium matrices.

Batching hurts when:

- matrices converge at different rates;
- deflation patterns differ;
- recursion or active subproblem sizes differ;
- workspace per matrix is large;
- memory traffic becomes dominant.

---

# Which Method Stands Out Because of Batching?

The method most improved by batching is:

```text
Jacobi / block Jacobi
```

Reason:

Jacobi has two levels of parallelism:

1. across matrices in the batch;
2. across independent rotations or block updates within each matrix.

For GPU batched EVD, this is why Jacobi is a serious contender even if it is not always the default algorithm for one large matrix.

---

# Practical Decision Table for This Competition

| Shape regime | Most plausible method |
|---|---|
| `n = 32`, large batch | shared-memory or scalar Jacobi |
| `n = 176`, large batch | batched Jacobi |
| `n = 352`, large batch | batched Jacobi |
| `n = 512`, `batch = 640` | batched/block Jacobi |
| `n = 1024`, `batch = 60` | block Jacobi or tridiagonal pipeline |
| `n = 2048`, `batch = 8` | tridiagonal pipeline with QR/D&C inner solve |
| `n = 4096`, small batch | tridiagonal pipeline |

---

# Final Ranking of Theoretical Avenues

For this exact competition:

## 1. Batched/block Jacobi

Most batch/GPU-natural. Best fit for `n = 512`, `batch = 640`.

## 2. Tridiagonal pipeline

Best classical route for larger `n`. Relevant for `n = 1024+`.

## 3. Divide-and-conquer

Mostly an inner tridiagonal solver. Useful inside the tridiagonal pipeline.

## 4. Shifted QR

Mostly an inner tridiagonal solver. Robust but less batch/GPU-natural.

No new eigendecomposition theory appears just because the input is batched.

The main batch-specific evolution is:

```text
scalar Jacobi
→ block Jacobi
→ batched block Jacobi
```

---

# Sources / Pointers for Further Reading

- NVIDIA cuSOLVER documentation: `syevj`, `syevjBatched`, and symmetric/Hermitian eigensolvers.
- MAGMA documentation: `syevd/heevd`, tridiagonalization, and dense symmetric EVD routines.
- LAPACK Users' Guide: symmetric eigenvalue routines, tridiagonal solvers, QR, and divide-and-conquer.
- GPU batched Hermitian EVD papers discussing Jacobi, blocked batched algorithms, convergence divergence, and shared-memory limitations.
