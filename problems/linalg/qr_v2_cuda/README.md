# qr_v2_cuda

Native CUDA workspace for developing QR kernels before packaging one into
`../qr_v2/submission.py`.

This directory is for H100-side development. The final Popcorn target is B200,
so use these numbers as relative guidance, not absolute leaderboard truth.

## Layout

```text
bench.cu              hardcoded benchmark suite matching final QR shapes
qr_kernel.h           kernel launcher interface
kernels/              one `.cu` file per candidate kernel
Makefile              build one benchmark binary for one kernel
run_bench_h100.sh     Slurm runner; run from this directory via sbatch
```

Each kernel candidate must implement:

```cpp
void qr_custom_kernel_cuda(
    const float* A,
    float* H,
    float* tau,
    int batch,
    int n,
    cudaStream_t stream);
```

Build selection is by kernel name:

```bash
make KERNEL=copy_stub
make KERNEL=cusolver_geqrf
make KERNEL=cublas_geqrf_batched
make KERNEL=magma_geqrf_batched
```

This compiles `kernels/copy_stub.cu` into `build/bench_copy_stub`.

Run only through Slurm on a GPU node:

```bash
sbatch run_bench_h100.sh copy_stub
sbatch run_bench_h100.sh cusolver_geqrf
sbatch run_bench_h100.sh cublas_geqrf_batched
sbatch run_bench_h100.sh magma_geqrf_batched
```

`copy_stub` is only a placeholder for validating the native harness. It is not a
correct QR factorization.

`cusolver_geqrf` is the first real baseline. It converts row-major input into a
temporary column-major buffer, calls `cusolverDnSgeqrf` once per matrix, then
converts the compact Householder result back to row-major `H`.

`cublas_geqrf_batched` uses `cublasSgeqrfBatched`, which factors the whole batch
through one batched cuBLAS call after the same row-major to column-major layout
conversion.

`magma_geqrf_batched` uses MAGMA's `magma_sgeqrf_batched` from the micromamba
environment at `/ictstr01/home/haicu/titouan.breton/micromamba/envs/magma-qr`.
