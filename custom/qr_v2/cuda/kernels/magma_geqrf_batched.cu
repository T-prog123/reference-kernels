#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <magma.h>
#include <magma_sbatched.h>

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

#define CHECK_MAGMA_LOCAL(expr)                                                 \
    do {                                                                        \
        magma_int_t _err = (expr);                                              \
        if (_err != MAGMA_SUCCESS) {                                            \
            std::fprintf(stderr, "MAGMA error %s:%d: status=%lld\n", __FILE__,  \
                         __LINE__, static_cast<long long>(_err));               \
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

#define CHECK_CUSPARSE_LOCAL(expr)                                              \
    do {                                                                        \
        cusparseStatus_t _err = (expr);                                         \
        if (_err != CUSPARSE_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuSPARSE error %s:%d: status=%d\n", __FILE__, \
                         __LINE__, static_cast<int>(_err));                     \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

__global__ void row_to_col_major_kernel(const float* A, float* C, int batch, int n,
                                        int ldda) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int j = static_cast<int>(idx % n);
        int i = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        size_t row_major = (static_cast<size_t>(b) * n + i) * n + j;
        size_t col_major = static_cast<size_t>(b) * ldda * n + i + static_cast<size_t>(j) * ldda;
        C[col_major] = A[row_major];
    }
}

__global__ void col_to_row_major_kernel(const float* C, float* H, int batch, int n,
                                        int ldda) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int j = static_cast<int>(idx % n);
        int i = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        size_t row_major = (static_cast<size_t>(b) * n + i) * n + j;
        size_t col_major = static_cast<size_t>(b) * ldda * n + i + static_cast<size_t>(j) * ldda;
        H[row_major] = C[col_major];
    }
}

__global__ void make_pointer_arrays_kernel(float* C, float* tau, float** Aarray,
                                           float** TauArray, int batch, int n, int ldda) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }

    size_t matrix_elems = static_cast<size_t>(ldda) * n;
    Aarray[b] = C + static_cast<size_t>(b) * matrix_elems;
    TauArray[b] = tau + static_cast<size_t>(b) * n;
}

void ensure_magma_initialized() {
    static bool initialized = false;
    if (!initialized) {
        CHECK_MAGMA_LOCAL(magma_init());
        std::atexit([]() { magma_finalize(); });
        initialized = true;
    }
}

void qr_custom_kernel_cuda(const float* A, float* H, float* tau, int batch, int n,
                           cudaStream_t stream) {
    // MAGMA geqrf_batched is LAPACK-style column-major. Convert row-major input
    // into column-major storage, factor through MAGMA, then convert compact H
    // back to row-major for the benchmark/PyTorch convention.
    magma_int_t ldda = magma_roundup(n, 32);
    size_t matrix_elems = static_cast<size_t>(ldda) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;
    size_t dense_elems = static_cast<size_t>(batch) * n * n;

    float* C = nullptr;
    float** Aarray = nullptr;
    float** TauArray = nullptr;
    magma_int_t* info = nullptr;

    CHECK_CUDA_LOCAL(cudaMalloc(&C, total_elems * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&Aarray, static_cast<size_t>(batch) * sizeof(float*)));
    CHECK_CUDA_LOCAL(cudaMalloc(&TauArray, static_cast<size_t>(batch) * sizeof(float*)));
    CHECK_CUDA_LOCAL(cudaMalloc(&info, static_cast<size_t>(batch) * sizeof(magma_int_t)));

    int threads = 256;
    CHECK_CUDA_LOCAL(cudaMemsetAsync(C, 0, total_elems * sizeof(float), stream));

    int elem_blocks = static_cast<int>((dense_elems + threads - 1) / threads);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    row_to_col_major_kernel<<<elem_blocks, threads, 0, stream>>>(A, C, batch, n, ldda);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    int batch_blocks = (batch + threads - 1) / threads;
    make_pointer_arrays_kernel<<<batch_blocks, threads, 0, stream>>>(
        C, tau, Aarray, TauArray, batch, n, ldda);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    CHECK_CUDA_LOCAL(cudaStreamSynchronize(stream));

    cublasHandle_t cublas = nullptr;
    cusparseHandle_t cusparse = nullptr;
    magma_queue_t queue = nullptr;
    int device = 0;

    CHECK_CUDA_LOCAL(cudaGetDevice(&device));
    ensure_magma_initialized();
    CHECK_CUBLAS_LOCAL(cublasCreate(&cublas));
    CHECK_CUBLAS_LOCAL(cublasSetStream(cublas, stream));
    CHECK_CUSPARSE_LOCAL(cusparseCreate(&cusparse));
    CHECK_CUSPARSE_LOCAL(cusparseSetStream(cusparse, stream));
    magma_queue_create_from_cuda(device, stream, cublas, cusparse, &queue);

    magma_int_t status = magma_sgeqrf_batched(
        n, n, Aarray, ldda, TauArray, info, batch, queue);
    if (status != MAGMA_SUCCESS) {
        std::fprintf(stderr, "MAGMA geqrf_batched failed: status=%lld\n",
                     static_cast<long long>(status));
        std::abort();
    }
    magma_queue_sync(queue);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    CHECK_CUDA_LOCAL(cudaStreamSynchronize(stream));

    std::vector<magma_int_t> h_info(static_cast<size_t>(batch));
    CHECK_CUDA_LOCAL(cudaMemcpy(h_info.data(), info,
                                static_cast<size_t>(batch) * sizeof(magma_int_t),
                                cudaMemcpyDeviceToHost));
    for (int b = 0; b < batch; ++b) {
        if (h_info[static_cast<size_t>(b)] != 0) {
            std::fprintf(stderr, "MAGMA geqrf_batched info[%d]=%lld (batch=%d n=%d ldda=%lld)\n",
                         b, static_cast<long long>(h_info[static_cast<size_t>(b)]),
                         batch, n, static_cast<long long>(ldda));
            std::abort();
        }
    }

    col_to_row_major_kernel<<<elem_blocks, threads, 0, stream>>>(C, H, batch, n, ldda);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    CHECK_CUDA_LOCAL(cudaStreamSynchronize(stream));

    magma_queue_destroy(queue);
    CHECK_CUSPARSE_LOCAL(cusparseDestroy(cusparse));
    CHECK_CUBLAS_LOCAL(cublasDestroy(cublas));

    CHECK_CUDA_LOCAL(cudaFree(info));
    CHECK_CUDA_LOCAL(cudaFree(TauArray));
    CHECK_CUDA_LOCAL(cudaFree(Aarray));
    CHECK_CUDA_LOCAL(cudaFree(C));
}
