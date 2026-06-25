#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../qr_kernel.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kPanelSize = 32;

enum class qr_precision {
    fp32,
    fp16,
    bf16,
    fp8
};

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

__device__ float block_sum(float value, float* scratch) {
    int tx = threadIdx.x;
    scratch[tx] = value;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tx < offset) {
            scratch[tx] += scratch[tx + offset];
        }
        __syncthreads();
    }
    return scratch[0];
}

__global__ void row_to_col_major_kernel(const float* A, float* C, int batch, int n) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int j = static_cast<int>(idx % n);
        int i = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        C[static_cast<size_t>(b) * n * n + i + static_cast<size_t>(j) * n] =
            A[(static_cast<size_t>(b) * n + i) * n + j];
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
        H[(static_cast<size_t>(b) * n + i) * n + j] =
            C[static_cast<size_t>(b) * n * n + i + static_cast<size_t>(j) * n];
    }
}

__global__ void panel_factor_kernel_float(
    float* A,
    float* tau,
    float* V,
    int n,
    int k,
    int ib)
{
    __shared__ float scratch[kBlockSize];
    __shared__ float tau_s;
    __shared__ float scale_s;
    __shared__ float dot_s;

    int b = blockIdx.x;
    int tx = threadIdx.x;
    int h = n - k;
    float* Ab = A + static_cast<size_t>(b) * n * n;
    float* taub = tau + static_cast<size_t>(b) * n;
    float* Vb = V + static_cast<size_t>(b) * n * kPanelSize;

    for (int j = 0; j < ib; ++j) {
        int col = k + j;
        int row0 = k + j;
        int len = h - j;
        float* x = Ab + row0 + static_cast<size_t>(col) * n;

        float local_norm2 = 0.0f;
        for (int r = tx + 1; r < len; r += blockDim.x) {
            float v = x[r];
            local_norm2 += v * v;
        }
        float xnorm2 = block_sum(local_norm2, scratch);

        if (tx == 0) {
            float alpha = x[0];
            float tau_val = 0.0f;
            float scale = 0.0f;
            float beta = alpha;

            if (xnorm2 > 0.0f) {
                float norm = sqrtf(alpha * alpha + xnorm2);
                beta = -copysignf(norm, alpha == 0.0f ? 1.0f : alpha);
                tau_val = (beta - alpha) / beta;
                scale = 1.0f / (alpha - beta);
            }

            x[0] = beta;
            taub[col] = tau_val;
            tau_s = tau_val;
            scale_s = scale;
        }
        __syncthreads();

        if (tau_s != 0.0f) {
            for (int r = tx + 1; r < len; r += blockDim.x) {
                x[r] *= scale_s;
            }
        }
        __syncthreads();

        for (int jj = j + 1; jj < ib; ++jj) {
            int update_col = k + jj;
            float* c = Ab + row0 + static_cast<size_t>(update_col) * n;
            float local_dot = (tx == 0) ? c[0] : 0.0f;
            for (int r = tx + 1; r < len; r += blockDim.x) {
                local_dot += x[r] * c[r];
            }
            float dot = block_sum(local_dot, scratch);

            if (tx == 0) {
                dot_s = tau_s * dot;
                c[0] -= dot_s;
            }
            __syncthreads();

            if (tau_s != 0.0f) {
                for (int r = tx + 1; r < len; r += blockDim.x) {
                    c[r] -= x[r] * dot_s;
                }
            }
            __syncthreads();
        }
    }

    int total_v = h * ib;
    for (int idx = tx; idx < total_v; idx += blockDim.x) {
        int col = idx / h;
        int row = idx - col * h;
        float value = 0.0f;
        if (row == col) {
            value = 1.0f;
        } else if (row > col) {
            value = Ab[(k + row) + static_cast<size_t>(k + col) * n];
        }
        Vb[row + static_cast<size_t>(col) * n] = value;
    }
}

__global__ void build_T_kernel_float(
    const float* V,
    const float* tau,
    float* T,
    int n,
    int k,
    int ib)
{
    __shared__ float scratch[kBlockSize];
    __shared__ float y[kPanelSize];

    int b = blockIdx.x;
    int tx = threadIdx.x;
    int h = n - k;
    const float* Vb = V + static_cast<size_t>(b) * n * kPanelSize;
    const float* taub = tau + static_cast<size_t>(b) * n + k;
    float* Tb = T + static_cast<size_t>(b) * kPanelSize * kPanelSize;

    for (int idx = tx; idx < kPanelSize * kPanelSize; idx += blockDim.x) {
        Tb[idx] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < ib; ++i) {
        float tau_i = taub[i];
        if (tx == 0) {
            Tb[i + static_cast<size_t>(i) * kPanelSize] = tau_i;
        }
        __syncthreads();

        for (int j = 0; j < i; ++j) {
            float local_dot = 0.0f;
            for (int r = tx; r < h; r += blockDim.x) {
                local_dot += Vb[r + static_cast<size_t>(j) * n] *
                             Vb[r + static_cast<size_t>(i) * n];
            }
            float dot = block_sum(local_dot, scratch);
            if (tx == 0) {
                y[j] = -tau_i * dot;
            }
            __syncthreads();
        }

        if (tx == 0) {
            for (int row = 0; row < i; ++row) {
                float accum = 0.0f;
                for (int col = 0; col < i; ++col) {
                    accum += Tb[row + static_cast<size_t>(col) * kPanelSize] * y[col];
                }
                Tb[row + static_cast<size_t>(i) * kPanelSize] = accum;
            }
        }
        __syncthreads();
    }
}

__global__ void build_T_from_gram_kernel_float(
    const float* G,
    const float* tau,
    float* T,
    int n,
    int k,
    int ib)
{
    int b = blockIdx.x;
    int tx = threadIdx.x;
    const float* Gb = G + static_cast<size_t>(b) * kPanelSize * kPanelSize;
    const float* taub = tau + static_cast<size_t>(b) * n + k;
    float* Tb = T + static_cast<size_t>(b) * kPanelSize * kPanelSize;

    for (int idx = tx; idx < kPanelSize * kPanelSize; idx += blockDim.x) {
        Tb[idx] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < ib; ++i) {
        float tau_i = taub[i];
        if (tx < i) {
            float accum = 0.0f;
            for (int col = 0; col < i; ++col) {
                float z = -tau_i * Gb[col + static_cast<size_t>(i) * kPanelSize];
                accum += Tb[tx + static_cast<size_t>(col) * kPanelSize] * z;
            }
            Tb[tx + static_cast<size_t>(i) * kPanelSize] = accum;
        } else if (tx == i) {
            Tb[i + static_cast<size_t>(i) * kPanelSize] = tau_i;
        }
        __syncthreads();
    }
}

void build_T_via_gram_fp32(
    cublasHandle_t handle,
    const float* V,
    const float* tau,
    float* G,
    float* T,
    int batch,
    int n,
    int k,
    int ib,
    cudaStream_t stream)
{
    int h = n - k;
    if (ib <= 0) {
        return;
    }

    const float one = 1.0f;
    const float zero = 0.0f;
    const long long stride_V = static_cast<long long>(n) * kPanelSize;
    const long long stride_G = static_cast<long long>(kPanelSize) * kPanelSize;

    CHECK_CUBLAS_LOCAL(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        ib, ib, h,
        &one,
        V, n, stride_V,
        V, n, stride_V,
        &zero,
        G, kPanelSize, stride_G,
        batch));

    build_T_from_gram_kernel_float<<<batch, kBlockSize, 0, stream>>>(
        G, tau, T, n, k, ib);
    CHECK_CUDA_LOCAL(cudaGetLastError());
}

void trailing_update_fp32(
    cublasHandle_t handle,
    float* A,
    const float* V,
    const float* T,
    float* W,
    float* W2,
    int batch,
    int n,
    int k,
    int ib)
{
    int h = n - k;
    int t = n - k - ib;
    if (t <= 0 || ib <= 0) {
        return;
    }

    const float one = 1.0f;
    const float zero = 0.0f;
    const float neg_one = -1.0f;

    const long long stride_A = static_cast<long long>(n) * n;
    const long long stride_V = static_cast<long long>(n) * kPanelSize;
    const long long stride_T = static_cast<long long>(kPanelSize) * kPanelSize;
    const long long stride_W = static_cast<long long>(kPanelSize) * n;

    const float* C = A + k + static_cast<size_t>(k + ib) * n;

    CHECK_CUBLAS_LOCAL(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        ib, t, h,
        &one,
        V, n, stride_V,
        C, n, stride_A,
        &zero,
        W, kPanelSize, stride_W,
        batch));

    CHECK_CUBLAS_LOCAL(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        ib, t, ib,
        &one,
        T, kPanelSize, stride_T,
        W, kPanelSize, stride_W,
        &zero,
        W2, kPanelSize, stride_W,
        batch));

    CHECK_CUBLAS_LOCAL(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        h, t, ib,
        &neg_one,
        V, n, stride_V,
        W2, kPanelSize, stride_W,
        &one,
        A + k + static_cast<size_t>(k + ib) * n, n, stride_A,
        batch));
}

void trailing_update_dispatch(
    cublasHandle_t handle,
    qr_precision mode,
    float* A,
    const float* V,
    const float* T,
    float* W,
    float* W2,
    int batch,
    int n,
    int k,
    int ib)
{
    switch (mode) {
        case qr_precision::fp32:
            trailing_update_fp32(handle, A, V, T, W, W2, batch, n, k, ib);
            return;
        case qr_precision::fp16:
        case qr_precision::bf16:
        case qr_precision::fp8:
            std::fprintf(stderr, "requested trailing-update precision is not implemented yet\n");
            std::abort();
    }
}

void blocked_wy_qr_float(
    const float* Arow,
    float* Hrow,
    float* tau,
    int batch,
    int n,
    cudaStream_t stream,
    qr_precision trailing_mode)
{
    size_t matrix_elems = static_cast<size_t>(n) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;

    float* Acol = nullptr;
    float* V = nullptr;
    float* G = nullptr;
    float* T = nullptr;
    float* W = nullptr;
    float* W2 = nullptr;

    CHECK_CUDA_LOCAL(cudaMalloc(&Acol, total_elems * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&V, static_cast<size_t>(batch) * n * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&G, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&T, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W, static_cast<size_t>(batch) * kPanelSize * n * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W2, static_cast<size_t>(batch) * kPanelSize * n * sizeof(float)));

    int elem_blocks = static_cast<int>((total_elems + kBlockSize - 1) / kBlockSize);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    row_to_col_major_kernel<<<elem_blocks, kBlockSize, 0, stream>>>(Arow, Acol, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS_LOCAL(cublasCreate(&handle));
    CHECK_CUBLAS_LOCAL(cublasSetStream(handle, stream));

    for (int k = 0; k < n; k += kPanelSize) {
        int ib = (n - k < kPanelSize) ? (n - k) : kPanelSize;
        panel_factor_kernel_float<<<batch, kBlockSize, 0, stream>>>(
            Acol, tau, V, n, k, ib);
        CHECK_CUDA_LOCAL(cudaGetLastError());

        if (n - k - ib > 0) {
            build_T_via_gram_fp32(
                handle, V, tau, G, T, batch, n, k, ib, stream);

            trailing_update_dispatch(
                handle, trailing_mode, Acol, V, T, W, W2, batch, n, k, ib);
        }
    }

    col_to_row_major_kernel<<<elem_blocks, kBlockSize, 0, stream>>>(Acol, Hrow, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    CHECK_CUBLAS_LOCAL(cublasDestroy(handle));
    CHECK_CUDA_LOCAL(cudaFree(W2));
    CHECK_CUDA_LOCAL(cudaFree(W));
    CHECK_CUDA_LOCAL(cudaFree(T));
    CHECK_CUDA_LOCAL(cudaFree(G));
    CHECK_CUDA_LOCAL(cudaFree(V));
    CHECK_CUDA_LOCAL(cudaFree(Acol));
}

}  // namespace

void qr_custom_kernel_cuda(const float* A, float* H, float* tau, int batch, int n,
                           cudaStream_t stream) {
    blocked_wy_qr_float(A, H, tau, batch, n, stream, qr_precision::fp32);
}
