#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../qr_kernel.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kPanelThreads = 512;
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

__device__ int panel_threads_per_column(int ncols) {
    if (ncols <= 1) return 512;
    if (ncols <= 2) return 256;
    if (ncols <= 4) return 128;
    if (ncols <= 8) return 64;
    if (ncols <= 16) return 32;
    return 16;
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

__global__ void float_to_half_contiguous_kernel(const float* src, __half* dst, size_t count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < count; idx += stride) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void trailing_float_to_half_kernel(
    const float* A,
    __half* Ahalf,
    int batch,
    int n,
    int k,
    int ib)
{
    int h = n - k;
    int t = n - k - ib;
    size_t total = static_cast<size_t>(batch) * h * t;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int row = static_cast<int>(idx % h);
        int col = static_cast<int>((idx / h) % t);
        int b = static_cast<int>(idx / (static_cast<size_t>(h) * t));
        size_t aidx = static_cast<size_t>(b) * n * n +
                      (k + row) + static_cast<size_t>(k + ib + col) * n;
        Ahalf[aidx] = __float2half(A[aidx]);
    }
}

__global__ void v_panel_float_to_half_kernel(
    const float* V,
    __half* Vhalf,
    int batch,
    int n,
    int k,
    int ib)
{
    int h = n - k;
    size_t total = static_cast<size_t>(batch) * h * ib;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int row = static_cast<int>(idx % h);
        int col = static_cast<int>((idx / h) % ib);
        int b = static_cast<int>(idx / (static_cast<size_t>(h) * ib));
        size_t vidx = static_cast<size_t>(b) * n * kPanelSize + row + static_cast<size_t>(col) * n;
        Vhalf[vidx] = __float2half(V[vidx]);
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
    __shared__ float scratch[kPanelThreads];
    __shared__ float tau_s;
    __shared__ float scale_s;
    __shared__ float dot_s[kPanelSize];

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

        int ncols = ib - j - 1;
        if (ncols > 0) {
            int tpc = panel_threads_per_column(ncols);
            int active_threads = tpc * ncols;
            int group = tx / tpc;
            int lane = tx - group * tpc;
            float local_dot = 0.0f;

            if (tx < active_threads) {
                int update_col = k + j + 1 + group;
                float* c = Ab + row0 + static_cast<size_t>(update_col) * n;
                local_dot = (lane == 0) ? c[0] : 0.0f;
                for (int r = lane + 1; r < len; r += tpc) {
                    local_dot += x[r] * c[r];
                }
            }
            scratch[tx] = local_dot;
            __syncthreads();

            for (int offset = tpc >> 1; offset > 0; offset >>= 1) {
                if (tx < active_threads && lane < offset) {
                    scratch[tx] += scratch[tx + offset];
                }
                __syncthreads();
            }

            if (tx < active_threads && lane == 0) {
                dot_s[group] = tau_s * scratch[tx];
            }
            __syncthreads();

            if (tau_s != 0.0f) {
                int total_update = len * ncols;
                for (int idx = tx; idx < total_update; idx += blockDim.x) {
                    int local_col = idx / len;
                    int r = idx - local_col * len;
                    int update_col = k + j + 1 + local_col;
                    float* c = Ab + row0 + static_cast<size_t>(update_col) * n;
                    float v = (r == 0) ? 1.0f : x[r];
                    c[r] -= v * dot_s[local_col];
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

void build_T_via_half_gram_fp32(
    cublasHandle_t handle,
    const float* V,
    const float* tau,
    __half* Vhalf,
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

    size_t live_v_count = static_cast<size_t>(batch) * h * ib;
    int v_blocks = static_cast<int>((live_v_count + kBlockSize - 1) / kBlockSize);
    v_blocks = v_blocks > 4096 ? 4096 : v_blocks;
    v_panel_float_to_half_kernel<<<v_blocks, kBlockSize, 0, stream>>>(V, Vhalf, batch, n, k, ib);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    const float one = 1.0f;
    const float zero = 0.0f;
    const long long stride_V = static_cast<long long>(n) * kPanelSize;
    const long long stride_G = static_cast<long long>(kPanelSize) * kPanelSize;

    CHECK_CUBLAS_LOCAL(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        ib, ib, h,
        &one,
        Vhalf, CUDA_R_16F, n, stride_V,
        Vhalf, CUDA_R_16F, n, stride_V,
        &zero,
        G, CUDA_R_32F, kPanelSize, stride_G,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

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

void trailing_update_fp16_conservative(
    cublasHandle_t handle,
    float* A,
    const float* V,
    const float* T,
    __half* Ahalf,
    __half* Vhalf,
    __half* Thalf,
    __half* W,
    __half* W2,
    int batch,
    int n,
    int k,
    int ib,
    cudaStream_t stream)
{
    int h = n - k;
    int t = n - k - ib;
    if (t <= 0 || ib <= 0) {
        return;
    }

    const long long stride_A = static_cast<long long>(n) * n;
    const long long stride_V = static_cast<long long>(n) * kPanelSize;
    const long long stride_T = static_cast<long long>(kPanelSize) * kPanelSize;
    const long long stride_W = static_cast<long long>(kPanelSize) * n;

    size_t t_count = static_cast<size_t>(batch) * kPanelSize * kPanelSize;
    size_t live_v_count = static_cast<size_t>(batch) * h * ib;
    int v_blocks = static_cast<int>((live_v_count + kBlockSize - 1) / kBlockSize);
    int t_blocks = static_cast<int>((t_count + kBlockSize - 1) / kBlockSize);
    v_blocks = v_blocks > 4096 ? 4096 : v_blocks;
    t_blocks = t_blocks > 4096 ? 4096 : t_blocks;

    v_panel_float_to_half_kernel<<<v_blocks, kBlockSize, 0, stream>>>(V, Vhalf, batch, n, k, ib);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    float_to_half_contiguous_kernel<<<t_blocks, kBlockSize, 0, stream>>>(T, Thalf, t_count);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    size_t trailing_count = static_cast<size_t>(batch) * h * t;
    int c_blocks = static_cast<int>((trailing_count + kBlockSize - 1) / kBlockSize);
    c_blocks = c_blocks > 4096 ? 4096 : c_blocks;
    trailing_float_to_half_kernel<<<c_blocks, kBlockSize, 0, stream>>>(
        A, Ahalf, batch, n, k, ib);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    const float one = 1.0f;
    const float zero = 0.0f;
    const float neg_one = -1.0f;
    const __half* C = Ahalf + k + static_cast<size_t>(k + ib) * n;

    CHECK_CUBLAS_LOCAL(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        ib, t, h,
        &one,
        Vhalf, CUDA_R_16F, n, stride_V,
        C, CUDA_R_16F, n, stride_A,
        &zero,
        W, CUDA_R_16F, kPanelSize, stride_W,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CHECK_CUBLAS_LOCAL(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        ib, t, ib,
        &one,
        Thalf, CUDA_R_16F, kPanelSize, stride_T,
        W, CUDA_R_16F, kPanelSize, stride_W,
        &zero,
        W2, CUDA_R_16F, kPanelSize, stride_W,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CHECK_CUBLAS_LOCAL(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        h, t, ib,
        &neg_one,
        Vhalf, CUDA_R_16F, n, stride_V,
        W2, CUDA_R_16F, kPanelSize, stride_W,
        &one,
        A + k + static_cast<size_t>(k + ib) * n, CUDA_R_32F, n, stride_A,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void trailing_update_dispatch(
    cublasHandle_t handle,
    qr_precision mode,
    float* A,
    const float* V,
    const float* T,
    float* W,
    float* W2,
    __half* Ahalf,
    __half* Vhalf,
    __half* Thalf,
    __half* Whalf,
    __half* W2half,
    int batch,
    int n,
    int k,
    int ib,
    cudaStream_t stream)
{
    switch (mode) {
        case qr_precision::fp32:
            trailing_update_fp32(handle, A, V, T, W, W2, batch, n, k, ib);
            return;
        case qr_precision::fp16:
            trailing_update_fp16_conservative(handle, A, V, T, Ahalf, Vhalf, Thalf,
                                              Whalf, W2half, batch, n, k, ib, stream);
            return;
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
    __half* Ahalf = nullptr;
    __half* Vhalf = nullptr;
    __half* Thalf = nullptr;
    __half* Whalf = nullptr;
    __half* W2half = nullptr;

    CHECK_CUDA_LOCAL(cudaMalloc(&Acol, total_elems * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&V, static_cast<size_t>(batch) * n * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&G, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&T, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W, static_cast<size_t>(batch) * kPanelSize * n * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W2, static_cast<size_t>(batch) * kPanelSize * n * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&Ahalf, total_elems * sizeof(__half)));
    CHECK_CUDA_LOCAL(cudaMalloc(&Vhalf, static_cast<size_t>(batch) * n * kPanelSize * sizeof(__half)));
    CHECK_CUDA_LOCAL(cudaMalloc(&Thalf, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(__half)));
    CHECK_CUDA_LOCAL(cudaMalloc(&Whalf, static_cast<size_t>(batch) * kPanelSize * n * sizeof(__half)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W2half, static_cast<size_t>(batch) * kPanelSize * n * sizeof(__half)));

    int elem_blocks = static_cast<int>((total_elems + kBlockSize - 1) / kBlockSize);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    row_to_col_major_kernel<<<elem_blocks, kBlockSize, 0, stream>>>(Arow, Acol, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS_LOCAL(cublasCreate(&handle));
    CHECK_CUBLAS_LOCAL(cublasSetStream(handle, stream));

    for (int k = 0; k < n; k += kPanelSize) {
        int ib = (n - k < kPanelSize) ? (n - k) : kPanelSize;
        panel_factor_kernel_float<<<batch, kPanelThreads, 0, stream>>>(
            Acol, tau, V, n, k, ib);
        CHECK_CUDA_LOCAL(cudaGetLastError());

        if (n - k - ib > 0) {
            build_T_via_half_gram_fp32(
                handle, V, tau, Vhalf, G, T, batch, n, k, ib, stream);

            trailing_update_dispatch(
                handle, trailing_mode, Acol, V, T, W, W2,
                Ahalf, Vhalf, Thalf, Whalf, W2half,
                batch, n, k, ib, stream);
        }
    }

    col_to_row_major_kernel<<<elem_blocks, kBlockSize, 0, stream>>>(Acol, Hrow, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());

    CHECK_CUBLAS_LOCAL(cublasDestroy(handle));
    CHECK_CUDA_LOCAL(cudaFree(W2half));
    CHECK_CUDA_LOCAL(cudaFree(Whalf));
    CHECK_CUDA_LOCAL(cudaFree(Thalf));
    CHECK_CUDA_LOCAL(cudaFree(Vhalf));
    CHECK_CUDA_LOCAL(cudaFree(Ahalf));
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
    blocked_wy_qr_float(A, H, tau, batch, n, stream, qr_precision::fp16);
}
