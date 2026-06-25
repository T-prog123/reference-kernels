#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdio>
#include <cstdlib>

#include "../qr_kernel.h"

#define CHECK_CUDA_LOCAL(expr)                                                  \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(_err));                             \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define CHECK_CUSOLVER_LOCAL(expr)                                              \
    do {                                                                        \
        cusolverStatus_t _err = (expr);                                         \
        if (_err != CUSOLVER_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuSOLVER error %s:%d: status=%d\n", __FILE__, \
                         __LINE__, static_cast<int>(_err));                     \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

__global__ void row_to_col_major_kernel(const float* A, float* C, int batch, int n) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int j = static_cast<int>(idx % n);
        int i = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        size_t row_major = (static_cast<size_t>(b) * n + i) * n + j;
        size_t col_major = static_cast<size_t>(b) * n * n + i + static_cast<size_t>(j) * n;
        C[col_major] = A[row_major];
    }
}

__global__ void col_to_row_major_kernel(const float* C, float* H, int batch, int n) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int j = static_cast<int>(idx % n);
        int i = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        size_t row_major = (static_cast<size_t>(b) * n + i) * n + j;
        size_t col_major = static_cast<size_t>(b) * n * n + i + static_cast<size_t>(j) * n;
        H[row_major] = C[col_major];
    }
}

void qr_custom_kernel_cuda(const float* A, float* H, float* tau, int batch, int n,
                           cudaStream_t stream) {
    // cuSOLVER geqrf is LAPACK-style column-major. PyTorch contiguous tensors
    // are row-major, so use a temporary column-major buffer for the factorization
    // and transpose the compact Householder result back into row-major H.
    size_t matrix_elems = static_cast<size_t>(n) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;

    float* C = nullptr;
    CHECK_CUDA_LOCAL(cudaMalloc(&C, total_elems * sizeof(float)));

    int threads = 256;
    int blocks = static_cast<int>((total_elems + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    row_to_col_major_kernel<<<blocks, threads, 0, stream>>>(A, C, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    cusolverDnHandle_t handle = nullptr;
    CHECK_CUSOLVER_LOCAL(cusolverDnCreate(&handle));
    CHECK_CUSOLVER_LOCAL(cusolverDnSetStream(handle, stream));

    int lwork = 0;
    CHECK_CUSOLVER_LOCAL(cusolverDnSgeqrf_bufferSize(handle, n, n, C, n, &lwork));

    float* work = nullptr;
    int* dev_info = nullptr;
    CHECK_CUDA_LOCAL(cudaMalloc(&work, static_cast<size_t>(lwork) * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&dev_info, sizeof(int)));

    for (int b = 0; b < batch; ++b) {
        float* C_b = C + static_cast<size_t>(b) * matrix_elems;
        float* tau_b = tau + static_cast<size_t>(b) * n;
        CHECK_CUSOLVER_LOCAL(cusolverDnSgeqrf(
            handle, n, n, C_b, n, tau_b, work, lwork, dev_info));
    }

    col_to_row_major_kernel<<<blocks, threads, 0, stream>>>(C, H, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    CHECK_CUDA_LOCAL(cudaFree(dev_info));
    CHECK_CUDA_LOCAL(cudaFree(work));
    CHECK_CUSOLVER_LOCAL(cusolverDnDestroy(handle));
    CHECK_CUDA_LOCAL(cudaFree(C));
}
