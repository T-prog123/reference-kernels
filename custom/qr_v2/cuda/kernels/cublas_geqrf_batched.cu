#include <cuda_runtime.h>
#include <cublas_v2.h>

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

#define CHECK_CUBLAS_LOCAL(expr)                                                \
    do {                                                                        \
        cublasStatus_t _err = (expr);                                           \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__,   \
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

__global__ void make_pointer_arrays_kernel(float* C, float* tau, float** Aarray,
                                           float** TauArray, int batch, int n) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }

    size_t matrix_elems = static_cast<size_t>(n) * n;
    Aarray[b] = C + static_cast<size_t>(b) * matrix_elems;
    TauArray[b] = tau + static_cast<size_t>(b) * n;
}

void qr_custom_kernel_cuda(const float* A, float* H, float* tau, int batch, int n,
                           cudaStream_t stream) {
    // cublasSgeqrfBatched is LAPACK-style column-major and takes device arrays
    // of per-matrix pointers. Convert the row-major benchmark input to a
    // temporary column-major buffer, factor it in one batched call, then convert
    // the compact Householder result back to row-major H.
    size_t matrix_elems = static_cast<size_t>(n) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;

    float* C = nullptr;
    float** Aarray = nullptr;
    float** TauArray = nullptr;
    int info = 0;

    CHECK_CUDA_LOCAL(cudaMalloc(&C, total_elems * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&Aarray, static_cast<size_t>(batch) * sizeof(float*)));
    CHECK_CUDA_LOCAL(cudaMalloc(&TauArray, static_cast<size_t>(batch) * sizeof(float*)));

    int threads = 256;
    int elem_blocks = static_cast<int>((total_elems + threads - 1) / threads);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    row_to_col_major_kernel<<<elem_blocks, threads, 0, stream>>>(A, C, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    CHECK_CUDA_LOCAL(cudaStreamSynchronize(stream));

    int batch_blocks = (batch + threads - 1) / threads;
    make_pointer_arrays_kernel<<<batch_blocks, threads, 0, stream>>>(
        C, tau, Aarray, TauArray, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    CHECK_CUDA_LOCAL(cudaStreamSynchronize(stream));

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS_LOCAL(cublasCreate(&handle));
    CHECK_CUBLAS_LOCAL(cublasSetStream(handle, stream));
    CHECK_CUBLAS_LOCAL(cublasSgeqrfBatched(
        handle, n, n, Aarray, n, TauArray, &info, batch));
    if (info != 0) {
        std::fprintf(stderr, "cuBLAS geqrfBatched parameter error: info=%d\n", info);
        std::abort();
    }
    CHECK_CUDA_LOCAL(cudaStreamSynchronize(stream));

    col_to_row_major_kernel<<<elem_blocks, threads, 0, stream>>>(C, H, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    CHECK_CUDA_LOCAL(cudaStreamSynchronize(stream));

    CHECK_CUBLAS_LOCAL(cublasDestroy(handle));
    CHECK_CUDA_LOCAL(cudaFree(TauArray));
    CHECK_CUDA_LOCAL(cudaFree(Aarray));
    CHECK_CUDA_LOCAL(cudaFree(C));
}
