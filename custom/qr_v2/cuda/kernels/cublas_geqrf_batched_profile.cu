#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../qr_kernel.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxRecords = 64;
constexpr int kMaxSamples = 8192;
constexpr int kPhaseCount = 9;

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
    double host_total_ms = 0.0;
    double alloc_ms = 0.0;
    double row_to_col_ms = 0.0;
    double ptrs_ms = 0.0;
    double cublas_create_ms = 0.0;
    double geqrf_ms = 0.0;
    double col_to_row_ms = 0.0;
    double cublas_destroy_ms = 0.0;
    double free_ms = 0.0;
};

struct ProfileSample {
    int batch = 0;
    int n = 0;
    int call_index = 0;
    double host_total_ms = 0.0;
    double alloc_ms = 0.0;
    double row_to_col_ms = 0.0;
    double ptrs_ms = 0.0;
    double cublas_create_ms = 0.0;
    double geqrf_ms = 0.0;
    double col_to_row_ms = 0.0;
    double cublas_destroy_ms = 0.0;
    double free_ms = 0.0;
};

ProfileRecord g_records[kMaxRecords];
int g_record_count = 0;
bool g_registered = false;
ProfileSample g_samples[kMaxSamples];
int g_sample_count = 0;

double elapsed_host_ms(HostClock::time_point start) {
    return std::chrono::duration<double, std::milli>(HostClock::now() - start).count();
}

double gpu_timer_stop(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream) {
    CHECK_CUDA_LOCAL(cudaEventRecord(stop, stream));
    CHECK_CUDA_LOCAL(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_LOCAL(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms);
}

bool variance_enabled() {
    const char* mode = std::getenv("QR_PROFILE_MODE");
    return mode != nullptr && std::strcmp(mode, "variance") == 0;
}

bool record_enabled() {
    const char* record = std::getenv("QR_PROFILE_RECORD");
    return record == nullptr || std::strcmp(record, "0") != 0;
}

double gpu_phase_ms(const ProfileSample& s) {
    return s.row_to_col_ms + s.ptrs_ms + s.geqrf_ms + s.col_to_row_ms;
}

double host_phase_ms(const ProfileSample& s) {
    return s.alloc_ms + s.cublas_create_ms + s.cublas_destroy_ms + s.free_ms;
}

const char* phase_name(int phase) {
    switch (phase) {
        case 0: return "alloc";
        case 1: return "row2col";
        case 2: return "ptrs";
        case 3: return "create";
        case 4: return "geqrf";
        case 5: return "col2row";
        case 6: return "destroy";
        case 7: return "free";
        case 8: return "hostOther";
        default: return "unknown";
    }
}

double phase_value(const ProfileSample& s, int phase) {
    switch (phase) {
        case 0: return s.alloc_ms;
        case 1: return s.row_to_col_ms;
        case 2: return s.ptrs_ms;
        case 3: return s.cublas_create_ms;
        case 4: return s.geqrf_ms;
        case 5: return s.col_to_row_ms;
        case 6: return s.cublas_destroy_ms;
        case 7: return s.free_ms;
        case 8: return s.host_total_ms - host_phase_ms(s) - gpu_phase_ms(s);
        default: return 0.0;
    }
}

bool same_case(const ProfileSample& a, const ProfileSample& b) {
    return a.batch == b.batch && a.n == b.n;
}

void case_means(int batch, int n, double* host_mean, double phase_means[kPhaseCount]) {
    int count = 0;
    double host_sum = 0.0;
    for (int p = 0; p < kPhaseCount; ++p) {
        phase_means[p] = 0.0;
    }

    for (int i = 0; i < g_sample_count; ++i) {
        const ProfileSample& s = g_samples[i];
        if (s.batch != batch || s.n != n) {
            continue;
        }
        count += 1;
        host_sum += s.host_total_ms;
        for (int p = 0; p < kPhaseCount; ++p) {
            phase_means[p] += phase_value(s, p);
        }
    }

    if (count <= 0) {
        *host_mean = 0.0;
        return;
    }
    *host_mean = host_sum / static_cast<double>(count);
    for (int p = 0; p < kPhaseCount; ++p) {
        phase_means[p] /= static_cast<double>(count);
    }
}

void print_variance_report() {
    if (!variance_enabled() || g_sample_count == 0) {
        return;
    }

    std::printf("\ncublas_geqrf_batched_profile variance attribution\n");
    std::printf("note: top phase = largest per-call sample stddev inside that case.\n");
    std::printf("%7s %6s %7s %9s %9s %7s %9s %9s %8s %9s %8s %9s %8s %9s %8s\n",
                "batch", "n", "samples", "hostMean", "hostStd", "cv%",
                "hostMax", "phase1", "std1", "max1", "std2", "max2",
                "std3", "max3", "top3");

    bool printed_cases[kMaxSamples] = {};
    for (int base = 0; base < g_sample_count; ++base) {
        if (printed_cases[base]) {
            continue;
        }
        const ProfileSample& first = g_samples[base];
        int count = 0;
        double host_sum = 0.0;
        double host_sum_sq = 0.0;
        double host_max = 0.0;
        double phase_sum[kPhaseCount] = {};
        double phase_sum_sq[kPhaseCount] = {};
        double phase_max[kPhaseCount] = {};

        for (int i = base; i < g_sample_count; ++i) {
            if (!same_case(first, g_samples[i])) {
                continue;
            }
            printed_cases[i] = true;
            const ProfileSample& s = g_samples[i];
            count += 1;
            host_sum += s.host_total_ms;
            host_sum_sq += s.host_total_ms * s.host_total_ms;
            if (s.host_total_ms > host_max) {
                host_max = s.host_total_ms;
            }
            for (int p = 0; p < kPhaseCount; ++p) {
                double v = phase_value(s, p);
                phase_sum[p] += v;
                phase_sum_sq[p] += v * v;
                if (v > phase_max[p]) {
                    phase_max[p] = v;
                }
            }
        }

        double denom = static_cast<double>(count);
        double host_mean = host_sum / denom;
        double host_var = host_sum_sq / denom - host_mean * host_mean;
        double host_std = host_var > 0.0 ? std::sqrt(host_var) : 0.0;
        double host_cv = host_mean > 0.0 ? 100.0 * host_std / host_mean : 0.0;
        double phase_std[kPhaseCount] = {};
        for (int p = 0; p < kPhaseCount; ++p) {
            double mean = phase_sum[p] / denom;
            double var = phase_sum_sq[p] / denom - mean * mean;
            phase_std[p] = var > 0.0 ? std::sqrt(var) : 0.0;
        }

        int top[3] = {-1, -1, -1};
        for (int p = 0; p < kPhaseCount; ++p) {
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
                    host_max, phase_name(top[0]), phase_std[top[0]],
                    phase_max[top[0]], phase_std[top[1]], phase_max[top[1]],
                    phase_std[top[2]], phase_max[top[2]], phase_name(top[0]),
                    phase_name(top[1]), phase_name(top[2]));
    }

    std::printf("\ncublas_geqrf_batched_profile slowest outliers explained\n");
    std::printf("note: deltas are phase_time - that case's phase_mean.\n");
    std::printf("%7s %6s %6s %9s %9s %10s %9s %10s %9s %10s %9s\n",
                "batch", "n", "call", "host_ms", "host+avg",
                "cause1", "delta1", "cause2", "delta2", "cause3", "delta3");

    bool printed[kMaxSamples] = {};
    int top_count = g_sample_count < 20 ? g_sample_count : 20;
    for (int rank = 0; rank < top_count; ++rank) {
        int best = -1;
        for (int i = 0; i < g_sample_count; ++i) {
            if (printed[i]) {
                continue;
            }
            if (best < 0 || g_samples[i].host_total_ms > g_samples[best].host_total_ms) {
                best = i;
            }
        }
        if (best < 0) {
            break;
        }
        printed[best] = true;
        const ProfileSample& s = g_samples[best];
        double host_mean = 0.0;
        double phase_mean[kPhaseCount] = {};
        case_means(s.batch, s.n, &host_mean, phase_mean);

        int cause[3] = {-1, -1, -1};
        double delta[3] = {};
        for (int p = 0; p < kPhaseCount; ++p) {
            double d = phase_value(s, p) - phase_mean[p];
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
                    s.host_total_ms - host_mean, phase_name(cause[0]), delta[0],
                    phase_name(cause[1]), delta[1], phase_name(cause[2]), delta[2]);
    }
}

void print_report() {
    if (g_record_count == 0) {
        return;
    }
    std::printf("\ncublas_geqrf_batched_profile phase breakdown\n");
    std::printf("%7s %6s %7s %11s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n",
                "batch", "n", "calls", "host_ms/call", "alloc%", "row2col%",
                "ptrs%", "create%", "geqrf%", "col2row%", "destroy%", "free%",
                "other%");
    for (int i = 0; i < g_record_count; ++i) {
        const ProfileRecord& r = g_records[i];
        double calls = static_cast<double>(r.calls > 0 ? r.calls : 1);
        double gpu_ms = r.row_to_col_ms + r.ptrs_ms + r.geqrf_ms + r.col_to_row_ms;
        double host_ms = r.alloc_ms + r.cublas_create_ms + r.cublas_destroy_ms + r.free_ms;
        double denom = r.host_total_ms > 0.0 ? r.host_total_ms : 1.0;
        double other = r.host_total_ms - gpu_ms - host_ms;
        if (other < 0.0) {
            other = 0.0;
        }
        std::printf("%7d %6d %7lld %11.4f %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n",
                    r.batch, r.n, r.calls, r.host_total_ms / calls,
                    100.0 * r.alloc_ms / denom,
                    100.0 * r.row_to_col_ms / denom,
                    100.0 * r.ptrs_ms / denom,
                    100.0 * r.cublas_create_ms / denom,
                    100.0 * r.geqrf_ms / denom,
                    100.0 * r.col_to_row_ms / denom,
                    100.0 * r.cublas_destroy_ms / denom,
                    100.0 * r.free_ms / denom,
                    100.0 * other / denom);
    }
    print_variance_report();
    std::fflush(stdout);
}

ProfileRecord* get_record(int batch, int n) {
    if (!g_registered) {
        std::atexit(print_report);
        g_registered = true;
    }
    for (int i = 0; i < g_record_count; ++i) {
        if (g_records[i].batch == batch && g_records[i].n == n) {
            return &g_records[i];
        }
    }
    if (g_record_count >= kMaxRecords) {
        return &g_records[kMaxRecords - 1];
    }
    ProfileRecord* rec = &g_records[g_record_count++];
    *rec = ProfileRecord{};
    rec->batch = batch;
    rec->n = n;
    return rec;
}

void record_sample(const ProfileRecord& after, const ProfileRecord& before) {
    if (!variance_enabled() || !record_enabled() || g_sample_count >= kMaxSamples) {
        return;
    }
    ProfileSample& s = g_samples[g_sample_count++];
    s.batch = after.batch;
    s.n = after.n;
    s.call_index = static_cast<int>(after.calls);
    s.host_total_ms = after.host_total_ms - before.host_total_ms;
    s.alloc_ms = after.alloc_ms - before.alloc_ms;
    s.row_to_col_ms = after.row_to_col_ms - before.row_to_col_ms;
    s.ptrs_ms = after.ptrs_ms - before.ptrs_ms;
    s.cublas_create_ms = after.cublas_create_ms - before.cublas_create_ms;
    s.geqrf_ms = after.geqrf_ms - before.geqrf_ms;
    s.col_to_row_ms = after.col_to_row_ms - before.col_to_row_ms;
    s.cublas_destroy_ms = after.cublas_destroy_ms - before.cublas_destroy_ms;
    s.free_ms = after.free_ms - before.free_ms;
}

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

}  // namespace

void qr_custom_kernel_cuda(const float* A, float* H, float* tau, int batch, int n,
                           cudaStream_t stream) {
    ProfileRecord* profile = get_record(batch, n);
    ProfileRecord before = *profile;
    profile->calls += 1;
    auto host_total_start = HostClock::now();

    cudaEvent_t timer_start = nullptr;
    cudaEvent_t timer_stop = nullptr;
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_start));
    CHECK_CUDA_LOCAL(cudaEventCreate(&timer_stop));

    size_t matrix_elems = static_cast<size_t>(n) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;

    float* C = nullptr;
    float** Aarray = nullptr;
    float** TauArray = nullptr;
    int info = 0;

    auto host_phase_start = HostClock::now();
    CHECK_CUDA_LOCAL(cudaMalloc(&C, total_elems * sizeof(float)));
    CHECK_CUDA_LOCAL(cudaMalloc(&Aarray, static_cast<size_t>(batch) * sizeof(float*)));
    CHECK_CUDA_LOCAL(cudaMalloc(&TauArray, static_cast<size_t>(batch) * sizeof(float*)));
    profile->alloc_ms += elapsed_host_ms(host_phase_start);

    int elem_blocks = static_cast<int>((total_elems + kBlockSize - 1) / kBlockSize);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start, stream));
    row_to_col_major_kernel<<<elem_blocks, kBlockSize, 0, stream>>>(A, C, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->row_to_col_ms += gpu_timer_stop(timer_start, timer_stop, stream);

    int batch_blocks = (batch + kBlockSize - 1) / kBlockSize;
    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start, stream));
    make_pointer_arrays_kernel<<<batch_blocks, kBlockSize, 0, stream>>>(
        C, tau, Aarray, TauArray, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->ptrs_ms += gpu_timer_stop(timer_start, timer_stop, stream);

    cublasHandle_t handle = nullptr;
    host_phase_start = HostClock::now();
    CHECK_CUBLAS_LOCAL(cublasCreate(&handle));
    CHECK_CUBLAS_LOCAL(cublasSetStream(handle, stream));
    profile->cublas_create_ms += elapsed_host_ms(host_phase_start);

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start, stream));
    CHECK_CUBLAS_LOCAL(cublasSgeqrfBatched(
        handle, n, n, Aarray, n, TauArray, &info, batch));
    if (info != 0) {
        std::fprintf(stderr, "cuBLAS geqrfBatched parameter error: info=%d\n", info);
        std::abort();
    }
    profile->geqrf_ms += gpu_timer_stop(timer_start, timer_stop, stream);

    CHECK_CUDA_LOCAL(cudaEventRecord(timer_start, stream));
    col_to_row_major_kernel<<<elem_blocks, kBlockSize, 0, stream>>>(C, H, batch, n);
    CHECK_CUDA_LOCAL(cudaGetLastError());
    profile->col_to_row_ms += gpu_timer_stop(timer_start, timer_stop, stream);

    host_phase_start = HostClock::now();
    CHECK_CUBLAS_LOCAL(cublasDestroy(handle));
    profile->cublas_destroy_ms += elapsed_host_ms(host_phase_start);

    host_phase_start = HostClock::now();
    CHECK_CUDA_LOCAL(cudaFree(TauArray));
    CHECK_CUDA_LOCAL(cudaFree(Aarray));
    CHECK_CUDA_LOCAL(cudaFree(C));
    profile->free_ms += elapsed_host_ms(host_phase_start);

    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_stop));
    CHECK_CUDA_LOCAL(cudaEventDestroy(timer_start));
    profile->host_total_ms += elapsed_host_ms(host_total_start);
    record_sample(*profile, before);
}
