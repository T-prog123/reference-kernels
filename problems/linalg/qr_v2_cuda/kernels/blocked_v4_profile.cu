#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../qr_kernel.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kPanelThreads = 512;
constexpr int kPanelSize = 32;

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

using HostClock = std::chrono::steady_clock;

struct ProfileRecord {
    int batch = 0;
    int n = 0;
    long long calls = 0;
    long long panels = 0;
    long long updates = 0;
    double host_total_ms = 0.0;
    double alloc_ms = 0.0;
    double cublas_create_ms = 0.0;
    double cublas_destroy_ms = 0.0;
    double free_ms = 0.0;
    double row_to_col_ms = 0.0;
    double panel_ms = 0.0;
    double gram_ms = 0.0;
    double t_recur_ms = 0.0;
    double gemm_vtc_ms = 0.0;
    double gemm_tw_ms = 0.0;
    double gemm_update_ms = 0.0;
    double col_to_row_ms = 0.0;
};

constexpr int kMaxProfileRecords = 64;
ProfileRecord g_profile_records[kMaxProfileRecords];
int g_profile_record_count = 0;
bool g_profile_registered = false;

double elapsed_host_ms(HostClock::time_point start) {
    return std::chrono::duration<double, std::milli>(HostClock::now() - start).count();
}

double gpu_timer_stop(cudaEvent_t start, cudaEvent_t stop) {
    CHECK_CUDA_LOCAL(cudaEventRecord(stop));
    CHECK_CUDA_LOCAL(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_LOCAL(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms);
}

void print_profile_report() {
    if (g_profile_record_count == 0) {
        return;
    }

    std::printf("\nblocked_v4_profile phase breakdown\n");
    std::printf("note: diagnostic build; phase timing synchronizes after each measured phase.\n");
    std::printf("note: this kernel target uses the same default execution path as current blocked_v4.\n");
    std::printf("%7s %6s %7s %7s %11s %8s %8s %8s %8s %8s %8s %8s %8s %11s\n",
                "batch", "n", "calls", "panels", "gpu_ms/call",
                "panel%", "gram%", "Trec%", "VtC%", "TW%", "upd%", "layout%",
                "other%", "host_alloc");

    for (int i = 0; i < g_profile_record_count; ++i) {
        const ProfileRecord& r = g_profile_records[i];
        double calls = static_cast<double>(r.calls > 0 ? r.calls : 1);
        double layout_ms = r.row_to_col_ms + r.col_to_row_ms;
        double gpu_ms = layout_ms + r.panel_ms + r.gram_ms + r.t_recur_ms +
                        r.gemm_vtc_ms + r.gemm_tw_ms + r.gemm_update_ms;
        double denom = gpu_ms > 0.0 ? gpu_ms : 1.0;
        double host_alloc = (r.alloc_ms + r.free_ms + r.cublas_create_ms +
                             r.cublas_destroy_ms) / calls;
        double other_pct = 100.0 * (r.host_total_ms - gpu_ms) /
                           (r.host_total_ms > 0.0 ? r.host_total_ms : 1.0);
        if (other_pct < 0.0) {
            other_pct = 0.0;
        }

        std::printf("%7d %6d %7lld %7lld %11.4f %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %11.4f\n",
                    r.batch, r.n, r.calls, r.panels,
                    gpu_ms / calls,
                    100.0 * r.panel_ms / denom,
                    100.0 * r.gram_ms / denom,
                    100.0 * r.t_recur_ms / denom,
                    100.0 * r.gemm_vtc_ms / denom,
                    100.0 * r.gemm_tw_ms / denom,
                    100.0 * r.gemm_update_ms / denom,
                    100.0 * layout_ms / denom,
                    other_pct,
                    host_alloc);
    }
    std::fflush(stdout);
}

ProfileRecord* get_profile_record(int batch, int n) {
    if (!g_profile_registered) {
        std::atexit(print_profile_report);
        g_profile_registered = true;
    }

    for (int i = 0; i < g_profile_record_count; ++i) {
        if (g_profile_records[i].batch == batch && g_profile_records[i].n == n) {
            return &g_profile_records[i];
        }
    }

    if (g_profile_record_count >= kMaxProfileRecords) {
        return &g_profile_records[kMaxProfileRecords - 1];
    }

    ProfileRecord* rec = &g_profile_records[g_profile_record_count++];
    *rec = ProfileRecord{};
    rec->batch = batch;
    rec->n = n;
    return rec;
}

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

void build_T_via_gram(
    cublasHandle_t handle,
    const float* V,
    const float* tau,
    float* G,
    float* T,
    int batch,
    int n,
    int k,
    int ib,
    ProfileRecord* profile,
    cudaEvent_t timer_start,
    cudaEvent_t timer_stop)
{
    int h = n - k;
    if (ib <= 0) {
        return;
    }

    const float one = 1.0f;
    const float zero = 0.0f;
    const long long stride_V = static_cast<long long>(n) * kPanelSize;
    const long long stride_G = static_cast<long long>(kPanelSize) * kPanelSize;

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
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
    profile->gram_ms += gpu_timer_stop(timer_start, timer_stop);

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
    build_T_from_gram_kernel_float<<<batch, kBlockSize>>>(
        G, tau, T, n, k, ib);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->t_recur_ms += gpu_timer_stop(timer_start, timer_stop);
}

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
    int ib,
    ProfileRecord* profile,
    cudaEvent_t timer_start,
    cudaEvent_t timer_stop)
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

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
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
    profile->gemm_vtc_ms += gpu_timer_stop(timer_start, timer_stop);

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
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
    profile->gemm_tw_ms += gpu_timer_stop(timer_start, timer_stop);

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
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
    profile->gemm_update_ms += gpu_timer_stop(timer_start, timer_stop);
}

void blocked_wy_qr_float(
    const float* Arow,
    float* Hrow,
    float* tau,
    int batch,
    int n)
{
    ProfileRecord* profile = get_profile_record(batch, n);
    profile->calls += 1;
    auto host_total_start = HostClock::now();

    cudaEvent_t timer_start = nullptr;
    cudaEvent_t timer_stop = nullptr;
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_start));
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_stop));

    size_t matrix_elems = static_cast<size_t>(n) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;

    float* Acol = nullptr;
    float* V = nullptr;
    float* G = nullptr;
    float* T = nullptr;
    float* W = nullptr;
    float* W2 = nullptr;

    auto host_phase_start = HostClock::now();
    CHECK_CUDA_LOCAL(cudaMalloc(&Acol, total_elems * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&V, static_cast<size_t>(batch) * n * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&G, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&T, static_cast<size_t>(batch) * kPanelSize * kPanelSize * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W, static_cast<size_t>(batch) * kPanelSize * n * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&W2, static_cast<size_t>(batch) * kPanelSize * n * sizeof(float)));
    profile->alloc_ms += elapsed_host_ms(host_phase_start);

    int elem_blocks = static_cast<int>((total_elems + kBlockSize - 1) / kBlockSize);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
    row_to_col_major_kernel<<<elem_blocks, kBlockSize>>>(Arow, Acol, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->row_to_col_ms += gpu_timer_stop(timer_start, timer_stop);

    cublasHandle_t handle = nullptr;
    host_phase_start = HostClock::now();
    CHECK_CUBLAS_LOCAL(cublasCreate(&handle));
    profile->cublas_create_ms += elapsed_host_ms(host_phase_start);

    for (int k = 0; k < n; k += kPanelSize) {
        int ib = (n - k < kPanelSize) ? (n - k) : kPanelSize;
        profile->panels += 1;
        CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
        panel_factor_kernel_float<<<batch, kPanelThreads>>>(
            Acol, tau, V, n, k, ib);
        CHECK_CUDA_LOCAL(cudaGetLastError());
        profile->panel_ms += gpu_timer_stop(timer_start, timer_stop);

        if (n - k - ib > 0) {
            profile->updates += 1;
            build_T_via_gram(
                handle, V, tau, G, T, batch, n, k, ib,
                profile, timer_start, timer_stop);

            trailing_update(
                handle, Acol, V, T, W, W2, batch, n, k, ib,
                profile, timer_start, timer_stop);
        }
    }

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
    col_to_row_major_kernel<<<elem_blocks, kBlockSize>>>(Acol, Hrow, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->col_to_row_ms += gpu_timer_stop(timer_start, timer_stop);

    host_phase_start = HostClock::now();
    CHECK_CUBLAS_LOCAL(cublasDestroy(handle));
    profile->cublas_destroy_ms += elapsed_host_ms(host_phase_start);

    host_phase_start = HostClock::now();
    CHECK_CUDA_LOCAL(cudaFree(W2));
    CHECK_CUDA_LOCAL(cudaFree(W));
    CHECK_CUDA_LOCAL(cudaFree(T));
    CHECK_CUDA_LOCAL(cudaFree(G));
    CHECK_CUDA_LOCAL(cudaFree(V));
    CHECK_CUDA_LOCAL(cudaFree(Acol));
    profile->free_ms += elapsed_host_ms(host_phase_start);

    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_stop));
    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_start));
    profile->host_total_ms += elapsed_host_ms(host_total_start);
}

}  // namespace

void qr_custom_kernel_cuda(const float* A, float* H, float* tau, int batch, int n,
                           ignored_launch_arg) {
    blocked_wy_qr_float(A, H, tau, batch, n);
}
