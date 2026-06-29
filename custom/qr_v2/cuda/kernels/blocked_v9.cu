#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../qr_kernel.h"

namespace {
namespace cg = cooperative_groups;

constexpr int kBlockSize = 256;
constexpr int kPanelThreads = 512;
constexpr int kPanelSize = 32;
constexpr int kDirectUpdateTileCols = 16;
constexpr int kPanelTileRows = 256;
constexpr int kMultiBlockPanelMinN = 2048;
constexpr int kMultiBlockPanelMinHeight = 1024;
constexpr int kMultiBlockPanelMaxBatch = 8;
constexpr int kTinySinglePanelCutoff = kPanelSize;
constexpr int kSmallNoTCutoff = 256;

enum class qr_path {
    tiny_single_panel,
    small_no_t,
    blocked_cublas
};

qr_path choose_qr_path(int n) {
    if (n <= kTinySinglePanelCutoff) {
        return qr_path::tiny_single_panel;
    }
    if (n <= kSmallNoTCutoff) {
        return qr_path::small_no_t;
    }
    return qr_path::blocked_cublas;
}

template <typename>
struct sixth_arg;

template <typename R, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5>
struct sixth_arg<R(A0, A1, A2, A3, A4, A5)> {
    using type = A5;
};

using ignored_launch_arg = typename sixth_arg<decltype(qr_custom_kernel_cuda)>::type;

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

__global__ void copy_row_major_kernel(const float* A, float* H, int batch, int n) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        H[idx] = A[idx];
    }
}

void copy_row_major_input(const float* Arow, float* Hrow, int batch, int n) {
    size_t total_elems = static_cast<size_t>(batch) * n * n;
    int elem_blocks = static_cast<int>((total_elems + kBlockSize - 1) / kBlockSize);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    copy_row_major_kernel<<<elem_blocks, kBlockSize>>>(Arow, Hrow, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
}

// Direct one-panel path for very small row-major matrices.
__global__ void tiny_single_panel_qr_kernel(
    const float* A,
    float* H,
    float* tau,
    int n)
{
    __shared__ float panel[kPanelSize * kPanelSize];
    __shared__ float scratch[kPanelThreads];
    __shared__ float tau_s;
    __shared__ float scale_s;
    __shared__ float dot_s[kPanelSize];

    int b = blockIdx.x;
    int tx = threadIdx.x;
    const float* Ab = A + static_cast<size_t>(b) * n * n;
    float* Hb = H + static_cast<size_t>(b) * n * n;
    float* taub = tau + static_cast<size_t>(b) * n;

    int total = n * n;
    for (int idx = tx; idx < total; idx += blockDim.x) {
        int col = idx % n;
        int row = idx / n;
        panel[row * kPanelSize + col] = Ab[row * n + col];
    }
    __syncthreads();

    for (int j = 0; j < n; ++j) {
        int len = n - j;
        float* x = panel + j * kPanelSize + j;

        float local_norm2 = 0.0f;
        for (int r = tx + 1; r < len; r += blockDim.x) {
            float v = x[r * kPanelSize];
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
            taub[j] = tau_val;
            tau_s = tau_val;
            scale_s = scale;
        }
        __syncthreads();

        if (tau_s != 0.0f) {
            for (int r = tx + 1; r < len; r += blockDim.x) {
                x[r * kPanelSize] *= scale_s;
            }
        }
        __syncthreads();

        int ncols = n - j - 1;
        if (ncols > 0) {
            int tpc = panel_threads_per_column(ncols);
            int active_threads = tpc * ncols;
            int group = tx / tpc;
            int lane = tx - group * tpc;
            float local_dot = 0.0f;

            if (tx < active_threads) {
                float* c = panel + j * kPanelSize + (j + 1 + group);
                local_dot = (lane == 0) ? c[0] : 0.0f;
                for (int r = lane + 1; r < len; r += tpc) {
                    local_dot += x[r * kPanelSize] * c[r * kPanelSize];
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
                    float* c = panel + j * kPanelSize + (j + 1 + local_col);
                    float v = (r == 0) ? 1.0f : x[r * kPanelSize];
                    c[r * kPanelSize] -= v * dot_s[local_col];
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tx; idx < total; idx += blockDim.x) {
        int col = idx % n;
        int row = idx / n;
        Hb[row * n + col] = panel[row * kPanelSize + col];
    }
}

void tiny_single_panel_qr(
    const float* Arow,
    float* Hrow,
    float* tau,
    int batch,
    int n)
{
    tiny_single_panel_qr_kernel<<<batch, kPanelThreads>>>(Arow, Hrow, tau, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
}

// Factor the current row-major panel. V is still column-major scratch.
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

        float local_norm2 = 0.0f;
        for (int r = tx + 1; r < len; r += blockDim.x) {
            float v = Ab[static_cast<size_t>(row0 + r) * n + col];
            local_norm2 += v * v;
        }
        float xnorm2 = block_sum(local_norm2, scratch);

        if (tx == 0) {
            float* x0 = Ab + static_cast<size_t>(row0) * n + col;
            float alpha = *x0;
            float tau_val = 0.0f;
            float scale = 0.0f;
            float beta = alpha;

            if (xnorm2 > 0.0f) {
                float norm = sqrtf(alpha * alpha + xnorm2);
                beta = -copysignf(norm, alpha == 0.0f ? 1.0f : alpha);
                tau_val = (beta - alpha) / beta;
                scale = 1.0f / (alpha - beta);
            }

            *x0 = beta;
            taub[col] = tau_val;
            tau_s = tau_val;
            scale_s = scale;
        }
        __syncthreads();

        if (tau_s != 0.0f) {
            for (int r = tx + 1; r < len; r += blockDim.x) {
                Ab[static_cast<size_t>(row0 + r) * n + col] *= scale_s;
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
                local_dot = (lane == 0)
                    ? Ab[static_cast<size_t>(row0) * n + update_col]
                    : 0.0f;
                for (int r = lane + 1; r < len; r += tpc) {
                    int row = row0 + r;
                    local_dot += Ab[static_cast<size_t>(row) * n + col] *
                                 Ab[static_cast<size_t>(row) * n + update_col];
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
                    int row = row0 + r;
                    int update_col = k + j + 1 + local_col;
                    float v = (r == 0)
                        ? 1.0f
                        : Ab[static_cast<size_t>(row) * n + col];
                    Ab[static_cast<size_t>(row) * n + update_col] -=
                        v * dot_s[local_col];
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
            value = Ab[static_cast<size_t>(k + row) * n + (k + col)];
        }
        Vb[row + static_cast<size_t>(col) * n] = value;
    }
}

__global__ void panel_factor_multiblock_kernel_float(
    float* A,
    float* tau,
    float* V,
    float* partial_norms,
    float* partial_dots,
    float* panel_scalars,
    int n,
    int k,
    int ib,
    int tile_rows,
    int tile_count)
{
    cg::grid_group grid = cg::this_grid();
    __shared__ float scratch[kPanelThreads];

    int b = blockIdx.x;
    int tile = blockIdx.y;
    int tx = threadIdx.x;
    int h = n - k;
    int tile_start = tile * tile_rows;
    int tile_end = tile_start + tile_rows;
    if (tile_end > h) {
        tile_end = h;
    }

    float* Ab = A + static_cast<size_t>(b) * n * n;
    float* taub = tau + static_cast<size_t>(b) * n;
    float* Vb = V + static_cast<size_t>(b) * n * kPanelSize;
    float* normb = partial_norms + static_cast<size_t>(b) * tile_count;
    float* dotb = partial_dots + static_cast<size_t>(b) * tile_count * kPanelSize;
    float* scalarb = panel_scalars + static_cast<size_t>(b) * 2;

    for (int j = 0; j < ib; ++j) {
        int col = k + j;
        int row0 = k + j;

        float local_norm2 = 0.0f;
        int norm_start = tile_start > j + 1 ? tile_start : j + 1;
        for (int r = norm_start + tx; r < tile_end; r += blockDim.x) {
            float value = Ab[static_cast<size_t>(k + r) * n + col];
            local_norm2 += value * value;
        }
        float tile_norm2 = block_sum(local_norm2, scratch);
        if (tx == 0) {
            normb[tile] = tile_norm2;
        }
        grid.sync();

        if (tile == 0) {
            float accum = 0.0f;
            for (int t = tx; t < tile_count; t += blockDim.x) {
                accum += normb[t];
            }
            float xnorm2 = block_sum(accum, scratch);

            if (tx == 0) {
                float* x0 = Ab + static_cast<size_t>(row0) * n + col;
                float alpha = *x0;
                float tau_val = 0.0f;
                float scale = 0.0f;
                float beta = alpha;

                if (xnorm2 > 0.0f) {
                    float norm = sqrtf(alpha * alpha + xnorm2);
                    beta = -copysignf(norm, alpha == 0.0f ? 1.0f : alpha);
                    tau_val = (beta - alpha) / beta;
                    scale = 1.0f / (alpha - beta);
                }

                *x0 = beta;
                taub[col] = tau_val;
                scalarb[0] = tau_val;
                scalarb[1] = scale;
            }
        }
        grid.sync();

        float tau_j = scalarb[0];
        float scale_j = scalarb[1];
        if (tau_j != 0.0f) {
            int scale_start = tile_start > j + 1 ? tile_start : j + 1;
            for (int r = scale_start + tx; r < tile_end; r += blockDim.x) {
                Ab[static_cast<size_t>(k + r) * n + col] *= scale_j;
            }
        }
        grid.sync();

        int ncols = ib - j - 1;
        if (ncols > 0) {
            int tpc = panel_threads_per_column(ncols);
            int active_threads = tpc * ncols;
            int group = tx / tpc;
            int lane = tx - group * tpc;
            float local_dot = 0.0f;
            int dot_start = tile_start > j ? tile_start : j;

            if (tx < active_threads) {
                int update_col = k + j + 1 + group;
                for (int r = dot_start + lane; r < tile_end; r += tpc) {
                    float v = (r == j)
                        ? 1.0f
                        : Ab[static_cast<size_t>(k + r) * n + col];
                    float c = Ab[static_cast<size_t>(k + r) * n + update_col];
                    local_dot += v * c;
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
                dotb[tile * kPanelSize + group] = scratch[tx];
            }
            grid.sync();

            if (tile == 0) {
                for (int idx = tx; idx < ncols; idx += blockDim.x) {
                    float dot = 0.0f;
                    for (int t = 0; t < tile_count; ++t) {
                        dot += dotb[t * kPanelSize + idx];
                    }
                    dotb[idx] = tau_j * dot;
                }
            }
            grid.sync();

            if (tau_j != 0.0f) {
                int update_start = tile_start > j ? tile_start : j;
                int rows = tile_end - update_start;
                int total_update = rows * ncols;
                for (int idx = tx; idx < total_update; idx += blockDim.x) {
                    int local_col = idx / rows;
                    int row_offset = idx - local_col * rows;
                    int r = update_start + row_offset;
                    int update_col = k + j + 1 + local_col;
                    float v = (r == j)
                        ? 1.0f
                        : Ab[static_cast<size_t>(k + r) * n + col];
                    float* c = Ab + static_cast<size_t>(k + r) * n + update_col;
                    *c -= v * dotb[local_col];
                }
            }
            grid.sync();
        }
    }

    int rows = tile_end - tile_start;
    int total_v = rows * ib;
    for (int idx = tx; idx < total_v; idx += blockDim.x) {
        int col = idx / rows;
        int local_row = idx - col * rows;
        int row = tile_start + local_row;
        float value = 0.0f;
        if (row == col) {
            value = 1.0f;
        } else if (row > col) {
            value = Ab[static_cast<size_t>(k + row) * n + (k + col)];
        }
        Vb[row + static_cast<size_t>(col) * n] = value;
    }
}

bool use_multiblock_panel(int batch, int n, int h) {
    return n >= kMultiBlockPanelMinN &&
           h >= kMultiBlockPanelMinHeight &&
           batch <= kMultiBlockPanelMaxBatch;
}

void launch_panel_factor(
    float* A,
    float* tau,
    float* V,
    float* partial_norms,
    float* partial_dots,
    float* panel_scalars,
    int batch,
    int n,
    int k,
    int ib)
{
    int h = n - k;
    if (use_multiblock_panel(batch, n, h)) {
        int tile_count = (h + kPanelTileRows - 1) / kPanelTileRows;
        int tile_rows = kPanelTileRows;
        dim3 grid(batch, tile_count);
        void* args[] = {
            &A, &tau, &V, &partial_norms, &partial_dots, &panel_scalars,
            &n, &k, &ib, &tile_rows, &tile_count
        };
        cudaError_t coop_err = cudaLaunchCooperativeKernel(
            (void*)panel_factor_multiblock_kernel_float,
            grid,
            dim3(kPanelThreads),
            args);
        if (coop_err == cudaErrorCooperativeLaunchTooLarge ||
            coop_err == cudaErrorNotSupported) {
            (void)cudaGetLastError();
            panel_factor_kernel_float<<<batch, kPanelThreads>>>(
                A, tau, V, n, k, ib);
            CHECK_CUDA_LOCAL(cudaGetLastError());
        } else {
            CHECK_CUDA_LOCAL(coop_err);
            CHECK_CUDA_LOCAL(cudaGetLastError());
        }
    } else {
        panel_factor_kernel_float<<<batch, kPanelThreads>>>(
            A, tau, V, n, k, ib);
        CHECK_CUDA_LOCAL(cudaGetLastError());
    }
}

// Direct Householder update for small matrices: no G, no T, no compact-WY GEMMs.
__global__ void direct_reflector_update_kernel(
    float* A,
    const float* tau,
    const float* V,
    int n,
    int k,
    int ib)
{
    __shared__ float scratch[kPanelThreads];
    __shared__ float dot_s[kDirectUpdateTileCols];

    int b = blockIdx.x;
    int tile = blockIdx.y;
    int tx = threadIdx.x;
    int h = n - k;
    int col_start = k + ib + tile * kDirectUpdateTileCols;
    int ncols = n - col_start;
    if (ncols > kDirectUpdateTileCols) {
        ncols = kDirectUpdateTileCols;
    }
    if (ncols <= 0) {
        return;
    }

    float* Ab = A + static_cast<size_t>(b) * n * n;
    const float* taub = tau + static_cast<size_t>(b) * n + k;
    const float* Vb = V + static_cast<size_t>(b) * n * kPanelSize;

    for (int j = 0; j < ib; ++j) {
        int len = h - j;
        float tau_j = taub[j];
        int tpc = panel_threads_per_column(ncols);
        int active_threads = tpc * ncols;
        int group = tx / tpc;
        int lane = tx - group * tpc;
        float local_dot = 0.0f;

        if (tau_j != 0.0f && tx < active_threads) {
            int update_col = col_start + group;
            const float* c = Ab + static_cast<size_t>(k + j) * n + update_col;
            for (int r = lane; r < len; r += tpc) {
                int vrow = j + r;
                local_dot += Vb[vrow + static_cast<size_t>(j) * n] *
                             c[static_cast<size_t>(r) * n];
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
            dot_s[group] = tau_j * scratch[tx];
        }
        __syncthreads();

        if (tau_j != 0.0f) {
            int total_update = len * ncols;
            for (int idx = tx; idx < total_update; idx += blockDim.x) {
                int local_col = idx / len;
                int r = idx - local_col * len;
                int update_col = col_start + local_col;
                int vrow = j + r;
                float* c = Ab + static_cast<size_t>(k + j) * n + update_col;
                c[static_cast<size_t>(r) * n] -=
                    Vb[vrow + static_cast<size_t>(j) * n] * dot_s[local_col];
            }
        }
        __syncthreads();
    }
}

void apply_direct_reflectors(
    float* A,
    const float* tau,
    const float* V,
    int batch,
    int n,
    int k,
    int ib)
{
    int trailing_cols = n - k - ib;
    if (trailing_cols <= 0 || ib <= 0) {
        return;
    }
    int tile_blocks = (trailing_cols + kDirectUpdateTileCols - 1) / kDirectUpdateTileCols;
    dim3 grid(batch, tile_blocks);
    direct_reflector_update_kernel<<<grid, kPanelThreads>>>(
        A, tau, V, n, k, ib);
    CHECK_CUDA_LOCAL(cudaGetLastError());
}

// Kernel to build the small triangular T matrix for blocked update.
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

// V/G/T remain column-major scratch so this gram build can stay conventional.
void build_T_via_gram(
    cublasHandle_t handle,
    const float* V,
    const float* tau,
    float* G,
    float* T,
    int batch,
    int n,
    int k,
    int ib)
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

    build_T_from_gram_kernel_float<<<batch, kBlockSize>>>(
        G, tau, T, n, k, ib);
    CHECK_CUDA_LOCAL(cudaGetLastError());
}

// Row-major trailing update. cuBLAS sees C, W, W2 as transposed column-major views.
void trailing_update(
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

    float* C = A + static_cast<size_t>(k) * n + (k + ib);

    // W^T = C^T V. W is row-major ib x t, viewed by cuBLAS as column-major t x ib.
    CHECK_CUBLAS_LOCAL(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        t, ib, h,
        &one,
        C, n, stride_A,
        V, n, stride_V,
        &zero,
        W, n, stride_W,
        batch));

    // W2^T = W^T T, equivalent to W2 = T^T W.
    CHECK_CUBLAS_LOCAL(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        t, ib, ib,
        &one,
        W, n, stride_W,
        T, kPanelSize, stride_T,
        &zero,
        W2, n, stride_W,
        batch));

    // C^T <- C^T - W2^T V^T, equivalent to C <- C - V W2.
    CHECK_CUBLAS_LOCAL(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        t, h, ib,
        &neg_one,
        W2, n, stride_W,
        V, n, stride_V,
        &one,
        C, n, stride_A,
        batch));
}

void blocked_qr_small_no_t(
    const float* Arow,
    float* Hrow,
    float* tau,
    int batch,
    int n)
{
    float* V = nullptr;
    CHECK_CUDA_LOCAL(cudaMalloc(&V, static_cast<size_t>(batch) * n * kPanelSize * sizeof(float)));

    copy_row_major_input(Arow, Hrow, batch, n);

    for (int k = 0; k < n; k += kPanelSize) {
        int ib = (n - k < kPanelSize) ? (n - k) : kPanelSize;
        panel_factor_kernel_float<<<batch, kPanelThreads>>>(
            Hrow, tau, V, n, k, ib);
        CHECK_CUDA_LOCAL(cudaGetLastError());

        if (n - k - ib > 0) {
            apply_direct_reflectors(Hrow, tau, V, batch, n, k, ib);
        }
    }

    CHECK_CUDA_LOCAL(cudaFree(V));
}

void blocked_wy_qr_cublas(
    const float* Arow,
    float* Hrow,
    float* tau,
    int batch,
    int n)
{
    float* V = nullptr;
    float* G = nullptr;
    float* T = nullptr;
    float* W = nullptr;
    float* W2 = nullptr;
    float* panel_norms = nullptr;
    float* panel_dots = nullptr;
    float* panel_scalars = nullptr;
    bool has_multiblock_panel = n >= kMultiBlockPanelMinN &&
                                batch <= kMultiBlockPanelMaxBatch;
    int max_tile_count = (n + kPanelTileRows - 1) / kPanelTileRows;

    CHECK_CUDA_LOCAL(cudaMalloc(&V, static_cast<size_t>(batch) * n * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&G, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&T, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W, static_cast<size_t>(batch) * kPanelSize * n * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W2, static_cast<size_t>(batch) * kPanelSize * n * sizeof(float)));
    if (has_multiblock_panel) {
        CHECK_CUDA_LOCAL(cudaMalloc(&panel_norms,
            static_cast<size_t>(batch) * max_tile_count * sizeof(float)));
        CHECK_CUDA_LOCAL(cudaMalloc(&panel_dots,
            static_cast<size_t>(batch) * max_tile_count * kPanelSize * sizeof(float)));
        CHECK_CUDA_LOCAL(cudaMalloc(&panel_scalars,
            static_cast<size_t>(batch) * 2 * sizeof(float)));
    }

    copy_row_major_input(Arow, Hrow, batch, n);

    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS_LOCAL(cublasCreate(&handle));

    for (int k = 0; k < n; k += kPanelSize) {
        int ib = (n - k < kPanelSize) ? (n - k) : kPanelSize;
        launch_panel_factor(
            Hrow, tau, V, panel_norms, panel_dots, panel_scalars,
            batch, n, k, ib);

        if (n - k - ib > 0) {
            build_T_via_gram(
                handle, V, tau, G, T, batch, n, k, ib);

            trailing_update(
                handle, Hrow, V, T, W, W2, batch, n, k, ib);
        }
    }

    CHECK_CUBLAS_LOCAL(cublasDestroy(handle));
    if (has_multiblock_panel) {
        CHECK_CUDA_LOCAL(cudaFree(panel_scalars));
        CHECK_CUDA_LOCAL(cudaFree(panel_dots));
        CHECK_CUDA_LOCAL(cudaFree(panel_norms));
    }
    CHECK_CUDA_LOCAL(cudaFree(W2));
    CHECK_CUDA_LOCAL(cudaFree(W));
    CHECK_CUDA_LOCAL(cudaFree(T));
    CHECK_CUDA_LOCAL(cudaFree(G));
    CHECK_CUDA_LOCAL(cudaFree(V));
}

}  // namespace

void qr_custom_kernel_cuda(const float* A, float* H, float* tau, int batch, int n,
                           ignored_launch_arg) {
    switch (choose_qr_path(n)) {
        case qr_path::tiny_single_panel:
            tiny_single_panel_qr(A, H, tau, batch, n);
            return;
        case qr_path::small_no_t:
            blocked_qr_small_no_t(A, H, tau, batch, n);
            return;
        case qr_path::blocked_cublas:
            blocked_wy_qr_cublas(A, H, tau, batch, n);
            return;
    }
}
