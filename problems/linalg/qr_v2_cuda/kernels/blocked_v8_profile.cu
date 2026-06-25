#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

using HostClock = std::chrono::steady_clock;

struct ProfileRecord {
    int batch = 0;
    int n = 0;
    long long calls = 0;
    long long tiny_calls = 0;
    long long small_calls = 0;
    long long cublas_calls = 0;
    long long panels = 0;
    long long multiblock_panels = 0;
    long long updates = 0;
    double host_total_ms = 0.0;
    double alloc_ms = 0.0;
    double cublas_create_ms = 0.0;
    double cublas_destroy_ms = 0.0;
    double free_ms = 0.0;
    double tiny_ms = 0.0;
    double row_to_col_ms = 0.0;
    double panel_ms = 0.0;
    double panel_multiblock_ms = 0.0;
    double direct_update_ms = 0.0;
    double gram_ms = 0.0;
    double t_recur_ms = 0.0;
    double gemm_vtc_ms = 0.0;
    double gemm_tw_ms = 0.0;
    double gemm_update_ms = 0.0;
    double col_to_row_ms = 0.0;
};

struct ProfileSample {
    int batch = 0;
    int n = 0;
    int call_index = 0;
    double host_total_ms = 0.0;
    double alloc_ms = 0.0;
    double cublas_create_ms = 0.0;
    double cublas_destroy_ms = 0.0;
    double free_ms = 0.0;
    double tiny_ms = 0.0;
    double row_to_col_ms = 0.0;
    double panel_ms = 0.0;
    double panel_multiblock_ms = 0.0;
    double direct_update_ms = 0.0;
    double gram_ms = 0.0;
    double t_recur_ms = 0.0;
    double gemm_vtc_ms = 0.0;
    double gemm_tw_ms = 0.0;
    double gemm_update_ms = 0.0;
    double col_to_row_ms = 0.0;
};

constexpr int kMaxProfileRecords = 64;
constexpr int kMaxProfileSamples = 8192;
ProfileRecord g_profile_records[kMaxProfileRecords];
int g_profile_record_count = 0;
bool g_profile_registered = false;
ProfileSample g_profile_samples[kMaxProfileSamples];
int g_profile_sample_count = 0;

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

const char* path_name(const ProfileRecord& r) {
    if (r.tiny_calls > 0) return "tiny";
    if (r.small_calls > 0) return "small";
    return "cublas";
}

bool profile_variance_enabled() {
    const char* mode = std::getenv("QR_PROFILE_MODE");
    return mode != nullptr && std::strcmp(mode, "variance") == 0;
}

bool profile_record_enabled() {
    const char* record = std::getenv("QR_PROFILE_RECORD");
    return record == nullptr || std::strcmp(record, "0") != 0;
}

double sample_gpu_ms(const ProfileSample& s) {
    return s.tiny_ms + s.row_to_col_ms + s.panel_ms + s.panel_multiblock_ms +
           s.direct_update_ms + s.gram_ms + s.t_recur_ms + s.gemm_vtc_ms +
           s.gemm_tw_ms + s.gemm_update_ms + s.col_to_row_ms;
}

double sample_host_phase_ms(const ProfileSample& s) {
    return s.alloc_ms + s.free_ms + s.cublas_create_ms + s.cublas_destroy_ms;
}

constexpr int kVariancePhaseCount = 14;

const char* variance_phase_name(int phase) {
    switch (phase) {
        case 0: return "alloc";
        case 1: return "cublas";
        case 2: return "free";
        case 3: return "layout";
        case 4: return "panel";
        case 5: return "panMB";
        case 6: return "dirUpd";
        case 7: return "gram";
        case 8: return "Trec";
        case 9: return "VtC";
        case 10: return "TW";
        case 11: return "upd";
        case 12: return "tiny";
        case 13: return "hostOther";
        default: return "unknown";
    }
}

double variance_phase_value(const ProfileSample& s, int phase) {
    switch (phase) {
        case 0: return s.alloc_ms;
        case 1: return s.cublas_create_ms + s.cublas_destroy_ms;
        case 2: return s.free_ms;
        case 3: return s.row_to_col_ms + s.col_to_row_ms;
        case 4: return s.panel_ms;
        case 5: return s.panel_multiblock_ms;
        case 6: return s.direct_update_ms;
        case 7: return s.gram_ms;
        case 8: return s.t_recur_ms;
        case 9: return s.gemm_vtc_ms;
        case 10: return s.gemm_tw_ms;
        case 11: return s.gemm_update_ms;
        case 12: return s.tiny_ms;
        case 13: return s.host_total_ms - sample_host_phase_ms(s) - sample_gpu_ms(s);
        default: return 0.0;
    }
}

bool same_sample_case(const ProfileSample& a, const ProfileSample& b) {
    return a.batch == b.batch && a.n == b.n;
}

void case_phase_means(
    int batch,
    int n,
    double* host_mean,
    double phase_means[kVariancePhaseCount])
{
    int count = 0;
    double host_sum = 0.0;
    for (int p = 0; p < kVariancePhaseCount; ++p) {
        phase_means[p] = 0.0;
    }

    for (int i = 0; i < g_profile_sample_count; ++i) {
        const ProfileSample& s = g_profile_samples[i];
        if (s.batch != batch || s.n != n) {
            continue;
        }
        count += 1;
        host_sum += s.host_total_ms;
        for (int p = 0; p < kVariancePhaseCount; ++p) {
            phase_means[p] += variance_phase_value(s, p);
        }
    }

    if (count <= 0) {
        *host_mean = 0.0;
        return;
    }

    *host_mean = host_sum / static_cast<double>(count);
    for (int p = 0; p < kVariancePhaseCount; ++p) {
        phase_means[p] /= static_cast<double>(count);
    }
}

void record_profile_sample(const ProfileRecord& after, const ProfileRecord& before) {
    if (!profile_variance_enabled() || !profile_record_enabled() ||
        g_profile_sample_count >= kMaxProfileSamples) {
        return;
    }

    ProfileSample& s = g_profile_samples[g_profile_sample_count++];
    s.batch = after.batch;
    s.n = after.n;
    s.call_index = static_cast<int>(after.calls);
    s.host_total_ms = after.host_total_ms - before.host_total_ms;
    s.alloc_ms = after.alloc_ms - before.alloc_ms;
    s.cublas_create_ms = after.cublas_create_ms - before.cublas_create_ms;
    s.cublas_destroy_ms = after.cublas_destroy_ms - before.cublas_destroy_ms;
    s.free_ms = after.free_ms - before.free_ms;
    s.tiny_ms = after.tiny_ms - before.tiny_ms;
    s.row_to_col_ms = after.row_to_col_ms - before.row_to_col_ms;
    s.panel_ms = after.panel_ms - before.panel_ms;
    s.panel_multiblock_ms = after.panel_multiblock_ms - before.panel_multiblock_ms;
    s.direct_update_ms = after.direct_update_ms - before.direct_update_ms;
    s.gram_ms = after.gram_ms - before.gram_ms;
    s.t_recur_ms = after.t_recur_ms - before.t_recur_ms;
    s.gemm_vtc_ms = after.gemm_vtc_ms - before.gemm_vtc_ms;
    s.gemm_tw_ms = after.gemm_tw_ms - before.gemm_tw_ms;
    s.gemm_update_ms = after.gemm_update_ms - before.gemm_update_ms;
    s.col_to_row_ms = after.col_to_row_ms - before.col_to_row_ms;
}

void print_variance_report() {
    if (!profile_variance_enabled() || g_profile_sample_count == 0) {
        return;
    }

    std::printf("\nblocked_v8_profile variance attribution\n");
    std::printf("note: samples are per timed qr_custom_kernel_cuda call; warmups are skipped by the harness.\n");
    std::printf("note: top phase = largest per-call sample stddev inside that case; max is that phase's worst single-call value.\n");
    std::printf("%7s %6s %7s %9s %9s %7s %9s %9s %8s %9s %8s %9s %8s %9s %8s\n",
                "batch", "n", "samples", "hostMean", "hostStd", "cv%",
                "hostMax", "phase1", "std1", "max1", "std2", "max2", "std3", "max3", "top3");

    bool printed_cases[kMaxProfileSamples] = {};
    for (int base = 0; base < g_profile_sample_count; ++base) {
        if (printed_cases[base]) {
            continue;
        }
        const ProfileSample& first = g_profile_samples[base];
        int count = 0;
        double host_sum = 0.0;
        double host_sum_sq = 0.0;
        double host_max = 0.0;
        double phase_sum[kVariancePhaseCount] = {};
        double phase_sum_sq[kVariancePhaseCount] = {};
        double phase_max[kVariancePhaseCount] = {};

        for (int i = base; i < g_profile_sample_count; ++i) {
            if (!same_sample_case(first, g_profile_samples[i])) {
                continue;
            }
            printed_cases[i] = true;
            const ProfileSample& s = g_profile_samples[i];
            count += 1;
            host_sum += s.host_total_ms;
            host_sum_sq += s.host_total_ms * s.host_total_ms;
            if (s.host_total_ms > host_max) {
                host_max = s.host_total_ms;
            }
            for (int p = 0; p < kVariancePhaseCount; ++p) {
                double v = variance_phase_value(s, p);
                phase_sum[p] += v;
                phase_sum_sq[p] += v * v;
                if (v > phase_max[p]) {
                    phase_max[p] = v;
                }
            }
        }

        if (count <= 0) {
            continue;
        }

        double denom = static_cast<double>(count);
        double host_mean = host_sum / denom;
        double host_var = host_sum_sq / denom - host_mean * host_mean;
        double host_std = host_var > 0.0 ? std::sqrt(host_var) : 0.0;
        double host_cv = host_mean > 0.0 ? 100.0 * host_std / host_mean : 0.0;

        double phase_std[kVariancePhaseCount] = {};
        for (int p = 0; p < kVariancePhaseCount; ++p) {
            double mean = phase_sum[p] / denom;
            double var = phase_sum_sq[p] / denom - mean * mean;
            phase_std[p] = var > 0.0 ? std::sqrt(var) : 0.0;
        }

        int top[3] = {-1, -1, -1};
        for (int p = 0; p < kVariancePhaseCount; ++p) {
            for (int slot = 0; slot < 3; ++slot) {
                if (top[slot] < 0 || phase_std[p] > phase_std[top[slot]]) {
                    for (int move = 2; move > slot; --move) {
                        top[move] = top[move - 1];
                    }
                    top[slot] = p;
                    break;
                }
            }
        }

        std::printf("%7d %6d %7d %9.3f %9.3f %6.2f%% %9.3f %9s %8.3f %9.3f %8.3f %9.3f %8.3f %9.3f %s/%s/%s\n",
                    first.batch, first.n, count, host_mean, host_std, host_cv,
                    host_max,
                    variance_phase_name(top[0]), phase_std[top[0]], phase_max[top[0]],
                    phase_std[top[1]], phase_max[top[1]],
                    phase_std[top[2]], phase_max[top[2]],
                    variance_phase_name(top[0]),
                    variance_phase_name(top[1]),
                    variance_phase_name(top[2]));
    }

    std::printf("\nblocked_v8_profile slowest outliers explained\n");
    std::printf("note: deltas are phase_time - that case's phase_mean; this is the actual spike attribution.\n");
    std::printf("%7s %6s %6s %9s %9s %10s %9s %10s %9s %10s %9s\n",
                "batch", "n", "call", "host_ms", "host+avg",
                "cause1", "delta1", "cause2", "delta2", "cause3", "delta3");

    bool printed_samples[kMaxProfileSamples] = {};
    int top_count = g_profile_sample_count < 20 ? g_profile_sample_count : 20;
    for (int rank = 0; rank < top_count; ++rank) {
        int best = -1;
        for (int i = 0; i < g_profile_sample_count; ++i) {
            if (printed_samples[i]) {
                continue;
            }
            if (best < 0 || g_profile_samples[i].host_total_ms >
                            g_profile_samples[best].host_total_ms) {
                best = i;
            }
        }
        if (best < 0) {
            break;
        }
        printed_samples[best] = true;
        const ProfileSample& s = g_profile_samples[best];
        double host_mean = 0.0;
        double phase_mean[kVariancePhaseCount] = {};
        case_phase_means(s.batch, s.n, &host_mean, phase_mean);

        int cause[3] = {-1, -1, -1};
        double delta[3] = {0.0, 0.0, 0.0};
        for (int p = 0; p < kVariancePhaseCount; ++p) {
            double d = variance_phase_value(s, p) - phase_mean[p];
            for (int slot = 0; slot < 3; ++slot) {
                if (cause[slot] < 0 || d > delta[slot]) {
                    for (int move = 2; move > slot; --move) {
                        cause[move] = cause[move - 1];
                        delta[move] = delta[move - 1];
                    }
                    cause[slot] = p;
                    delta[slot] = d;
                    break;
                }
            }
        }

        std::printf("%7d %6d %6d %9.3f %+9.3f %10s %+9.3f %10s %+9.3f %10s %+9.3f\n",
                    s.batch, s.n, s.call_index, s.host_total_ms,
                    s.host_total_ms - host_mean,
                    variance_phase_name(cause[0]), delta[0],
                    variance_phase_name(cause[1]), delta[1],
                    variance_phase_name(cause[2]), delta[2]);
    }
    std::fflush(stdout);
}

void print_profile_report() {
    if (g_profile_record_count == 0) {
        return;
    }

    std::printf("\nblocked_v8_profile phase breakdown\n");
    std::printf("note: diagnostic build; phase timing synchronizes after each measured phase.\n");
    std::printf("note: multiblock panel active when n>=%d, h>=%d, batch<=%d.\n",
                kMultiBlockPanelMinN, kMultiBlockPanelMinHeight, kMultiBlockPanelMaxBatch);
    std::printf("%7s %6s %7s %7s %7s %7s %11s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %11s\n",
                "batch", "n", "path", "calls", "panels", "mbPan", "gpu_ms/call",
                "tiny%", "panel%", "panelMB%", "dirUpd%", "gram%", "Trec%",
                "VtC%", "TW%", "upd%", "layout%", "other%", "host_alloc");

    for (int i = 0; i < g_profile_record_count; ++i) {
        const ProfileRecord& r = g_profile_records[i];
        double calls = static_cast<double>(r.calls > 0 ? r.calls : 1);
        double layout_ms = r.row_to_col_ms + r.col_to_row_ms;
        double gpu_ms = r.tiny_ms + layout_ms + r.panel_ms + r.panel_multiblock_ms +
                        r.direct_update_ms + r.gram_ms + r.t_recur_ms +
                        r.gemm_vtc_ms + r.gemm_tw_ms + r.gemm_update_ms;
        double denom = gpu_ms > 0.0 ? gpu_ms : 1.0;
        double host_alloc = (r.alloc_ms + r.free_ms + r.cublas_create_ms +
                             r.cublas_destroy_ms) / calls;
        double other_pct = 100.0 * (r.host_total_ms - gpu_ms) /
                           (r.host_total_ms > 0.0 ? r.host_total_ms : 1.0);
        if (other_pct < 0.0) {
            other_pct = 0.0;
        }

        std::printf("%7d %6d %7s %7lld %7lld %7lld %11.4f %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %11.4f\n",
                    r.batch, r.n, path_name(r), r.calls, r.panels, r.multiblock_panels,
                    gpu_ms / calls,
                    100.0 * r.tiny_ms / denom,
                    100.0 * r.panel_ms / denom,
                    100.0 * r.panel_multiblock_ms / denom,
                    100.0 * r.direct_update_ms / denom,
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
    print_variance_report();
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


// from python and pytroch row-major to cuBLAS column-major
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

// from cublas column-major to python and pytorch row-major.
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

// direct one-pannel path for very small matrices.
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
    ProfileRecord* profile = get_profile_record(batch, n);
    ProfileRecord before = *profile;
    profile->calls += 1;
    profile->tiny_calls += 1;
    auto host_total_start = HostClock::now();

    cudaEvent_t timer_start = nullptr;
    cudaEvent_t timer_stop = nullptr;
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_start));
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_stop));

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
    tiny_single_panel_qr_kernel<<<batch, kPanelThreads>>>(Arow, Hrow, tau, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->tiny_ms += gpu_timer_stop(timer_start, timer_stop);

    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_stop));
    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_start));
    profile->host_total_ms += elapsed_host_ms(host_total_start);
    record_profile_sample(*profile, before);
}

// custom kernel that factorises the current pannel and builds the householder reflectors (step 1)
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
    int h = n - k; // pannel height
    float* Ab = A + static_cast<size_t>(b) * n * n;
    float* taub = tau + static_cast<size_t>(b) * n;
    float* Vb = V + static_cast<size_t>(b) * n * kPanelSize;

    for (int j = 0; j < ib; ++j) { // loop over pannel columns: veyr large loop
        int col = k + j;
        int row0 = k + j;
        int len = h - j;
        float* x = Ab + row0 + static_cast<size_t>(col) * n;
        
        // compute the tail norm
        float local_norm2 = 0.0f;
        for (int r = tx + 1; r < len; r += blockDim.x) {
            float v = x[r];
            local_norm2 += v * v;
        }
        float xnorm2 = block_sum(local_norm2, scratch);
        
        // thread 0 computes the scalar values, notablue tau
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
        
        // store the new v vector (the normal vector of the hyperplane)
        if (tau_s != 0.0f) {
            for (int r = tx + 1; r < len; r += blockDim.x) {
                x[r] *= scale_s;
            }
        }
        __syncthreads();

        // apply the reflection to the remaining panel columns
        int ncols = ib - j - 1;
        if (ncols > 0) {
            int tpc = panel_threads_per_column(ncols); // seems to be the V4 trick that makes things faster...
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
                // update C
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

    // build V explicitly.
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
        int len = h - j;
        float* x = Ab + row0 + static_cast<size_t>(col) * n;

        float local_norm2 = 0.0f;
        int norm_start = tile_start > j + 1 ? tile_start : j + 1;
        for (int r = norm_start + tx; r < tile_end; r += blockDim.x) {
            float value = Ab[(k + r) + static_cast<size_t>(col) * n];
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
                Ab[(k + r) + static_cast<size_t>(col) * n] *= scale_j;
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
                            : Ab[(k + r) + static_cast<size_t>(col) * n];
                    float c = Ab[(k + r) + static_cast<size_t>(update_col) * n];
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
                            : Ab[(k + r) + static_cast<size_t>(col) * n];
                    float* c = Ab + (k + r) + static_cast<size_t>(update_col) * n;
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
            value = Ab[(k + row) + static_cast<size_t>(k + col) * n];
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
    int ib,
    ProfileRecord* profile,
    cudaEvent_t timer_start,
    cudaEvent_t timer_stop)
{
    int h = n - k;
    profile->panels += 1;
    if (use_multiblock_panel(batch, n, h)) {
        int tile_count = (h + kPanelTileRows - 1) / kPanelTileRows;
        int tile_rows = kPanelTileRows;
        dim3 grid(batch, tile_count);
        void* args[] = {
            &A, &tau, &V, &partial_norms, &partial_dots, &panel_scalars,
            &n, &k, &ib, &tile_rows, &tile_count
        };
        CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
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
            profile->panel_ms += gpu_timer_stop(timer_start, timer_stop);
        } else {
            CHECK_CUDA_LOCAL(coop_err);
            CHECK_CUDA_LOCAL(cudaGetLastError());
            profile->multiblock_panels += 1;
            profile->panel_multiblock_ms += gpu_timer_stop(timer_start, timer_stop);
        }
    } else {
        CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
        panel_factor_kernel_float<<<batch, kPanelThreads>>>(
            A, tau, V, n, k, ib);
        CHECK_CUDA_LOCAL(cudaGetLastError());
        profile->panel_ms += gpu_timer_stop(timer_start, timer_stop);
    }
}

// direct Householder update for small matrices: no G, no T, no compact-WY GEMMs.
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
            float* c = Ab + (k + j) + static_cast<size_t>(update_col) * n;
            for (int r = lane; r < len; r += tpc) {
                int vrow = j + r;
                local_dot += Vb[vrow + static_cast<size_t>(j) * n] * c[r];
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
                float* c = Ab + (k + j) + static_cast<size_t>(update_col) * n;
                c[r] -= Vb[vrow + static_cast<size_t>(j) * n] * dot_s[local_col];
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
    int ib,
    ProfileRecord* profile,
    cudaEvent_t timer_start,
    cudaEvent_t timer_stop)
{
    int trailing_cols = n - k - ib;
    if (trailing_cols <= 0 || ib <= 0) {
        return;
    }
    int tile_blocks = (trailing_cols + kDirectUpdateTileCols - 1) / kDirectUpdateTileCols;
    dim3 grid(batch, tile_blocks);
    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
    direct_reflector_update_kernel<<<grid, kPanelThreads>>>(
        A, tau, V, n, k, ib);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->direct_update_ms += gpu_timer_stop(timer_start, timer_stop);
}


// kernel to build the small trianguler T matrix for blocked update
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

// builds the T matrix so block QR can be applied to the trail (step 2)
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

// update the trail (once T is computed) with 3 CUBLASS operations
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

void blocked_qr_small_no_t(
    const float* Arow,
    float* Hrow,
    float* tau,
    int batch,
    int n)
{
    ProfileRecord* profile = get_profile_record(batch, n);
    ProfileRecord before = *profile;
    profile->calls += 1;
    profile->small_calls += 1;
    auto host_total_start = HostClock::now();

    cudaEvent_t timer_start = nullptr;
    cudaEvent_t timer_stop = nullptr;
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_start));
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_stop));

    size_t matrix_elems = static_cast<size_t>(n) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;


    // memory allocation
    float* Acol = nullptr;
    float* V = nullptr;

    auto host_phase_start = HostClock::now();
    CHECK_CUDA_LOCAL(cudaMalloc(&Acol, total_elems * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&V, static_cast<size_t>(batch) * n * kPanelSize * sizeof(float)));
    profile->alloc_ms += elapsed_host_ms(host_phase_start);

    int elem_blocks = static_cast<int>((total_elems + kBlockSize - 1) / kBlockSize);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
    row_to_col_major_kernel<<<elem_blocks, kBlockSize>>>(Arow, Acol, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->row_to_col_ms += gpu_timer_stop(timer_start, timer_stop);
    
    // outer loop over the pannels
    for (int k = 0; k < n; k += kPanelSize) {
        int ib = (n - k < kPanelSize) ? (n - k) : kPanelSize;
        // custom kernel that factorises the current pannel and builds the householder reflectors
        // the number of blocks is the batch... so for 2 matrices, it has 2 blocks! 
        CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
        panel_factor_kernel_float<<<batch, kPanelThreads>>>(
            Acol, tau, V, n, k, ib);
        CHECK_CUDA_LOCAL(cudaGetLastError());
        profile->panels += 1;
        profile->panel_ms += gpu_timer_stop(timer_start, timer_stop);

        if (n - k - ib > 0) {   // checks if there is still a trail to update. 
            profile->updates += 1;
            apply_direct_reflectors(
                Acol, tau, V, batch, n, k, ib,
                profile, timer_start, timer_stop);
        }
    }

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start));
    col_to_row_major_kernel<<<elem_blocks, kBlockSize>>>(Acol, Hrow, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->col_to_row_ms += gpu_timer_stop(timer_start, timer_stop);

    host_phase_start = HostClock::now();
    CHECK_CUDA_LOCAL(cudaFree(V));
    CHECK_CUDA_LOCAL(cudaFree(Acol));
    profile->free_ms += elapsed_host_ms(host_phase_start);

    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_stop));
    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_start));
    profile->host_total_ms += elapsed_host_ms(host_total_start);
    record_profile_sample(*profile, before);
}

// Main orchestrator function
void blocked_wy_qr_cublas(
    const float* Arow,
    float* Hrow,
    float* tau,
    int batch,
    int n)
{
    ProfileRecord* profile = get_profile_record(batch, n);
    ProfileRecord before = *profile;
    profile->calls += 1;
    profile->cublas_calls += 1;
    auto host_total_start = HostClock::now();

    cudaEvent_t timer_start = nullptr;
    cudaEvent_t timer_stop = nullptr;
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_start));
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_stop));

    size_t matrix_elems = static_cast<size_t>(n) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;


    // memory allocation
    float* Acol = nullptr;
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

    auto host_phase_start = HostClock::now();
    CHECK_CUDA_LOCAL(cudaMalloc(&Acol, total_elems * sizeof(float)));
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
    
    // outer loop over the pannels
    for (int k = 0; k < n; k += kPanelSize) {
        int ib = (n - k < kPanelSize) ? (n - k) : kPanelSize;
        // custom kernel that factorises the current pannel and builds the householder reflectors
        // the number of blocks is the batch... so for 2 matrices, it has 2 blocks! 
        launch_panel_factor(
            Acol, tau, V, panel_norms, panel_dots, panel_scalars,
            batch, n, k, ib, profile, timer_start, timer_stop);

        if (n - k - ib > 0) {   // checks if there is still a trail to update. 
            profile->updates += 1;
            build_T_via_gram(   // builds the T matrix so block QR can be applied to the trail
                handle, V, tau, G, T, batch, n, k, ib,
                profile, timer_start, timer_stop);

            trailing_update(  // update the trail 
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
    CHECK_CUDA_LOCAL(cudaFree(Acol));
    profile->free_ms += elapsed_host_ms(host_phase_start);

    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_stop));
    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_start));
    profile->host_total_ms += elapsed_host_ms(host_total_start);
    record_profile_sample(*profile, before);
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
