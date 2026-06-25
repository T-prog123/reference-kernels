#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "qr_kernel.h"

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(_err));                             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

enum class CaseKind {
    Dense,
    Mixed,
    RankDef,
    Clustered,
    NearRank,
};

struct BenchCase {
    const char* name;
    int batch;
    int n;
    int cond;
    CaseKind kind;
    int warmup;
    int iters;
};

struct BenchStats {
    int runs;
    double mean;
    double std;
    double err;
    double best;
    double worst;
    bool finite;
};

struct HarnessProfile {
    const char* name;
    int batch;
    int n;
    int warmup;
    int iters;
    double total_wall_ms;
    double alloc_ms;
    double fill_wall_ms;
    double fill_gpu_ms;
    double warmup_wall_ms;
    double event_setup_ms;
    double timed_qr_total_ms;
    double timed_qr_mean_ms;
    double finite_check_ms;
    double free_ms;
    bool finite;
};

enum class ProfileMode {
    Aggregate,
    Variance,
};

struct ProfileConfig {
    ProfileMode mode;
    const char* mode_name;
    const char* case_filter;
    int variance_iters;
};

using HostClock = std::chrono::steady_clock;

static const BenchCase kCases[] = {
    {"dense_b20_n32_c1", 20, 32, 1, CaseKind::Dense, 10, 100},
    {"dense_b40_n176_c1", 40, 176, 1, CaseKind::Dense, 5, 50},
    {"dense_b40_n352_c1", 40, 352, 1, CaseKind::Dense, 5, 30},
    {"dense_b640_n512_c2", 640, 512, 2, CaseKind::Dense, 3, 10},
    {"dense_b60_n1024_c2", 60, 1024, 2, CaseKind::Dense, 3, 8},
    {"dense_b8_n2048_c1", 8, 2048, 1, CaseKind::Dense, 2, 5},
    {"dense_b2_n4096_c1", 2, 4096, 1, CaseKind::Dense, 1, 3},
    {"mixed_b640_n512_c2", 640, 512, 2, CaseKind::Mixed, 3, 10},
    {"mixed_b60_n1024_c2", 60, 1024, 2, CaseKind::Mixed, 3, 8},
    {"rankdef_b640_n512_c0", 640, 512, 0, CaseKind::RankDef, 3, 10},
    {"clustered_b640_n512_c0", 640, 512, 0, CaseKind::Clustered, 3, 10},
    {"nearrank_b60_n1024_c0", 60, 1024, 0, CaseKind::NearRank, 3, 8},
};

double elapsed_host_ms(HostClock::time_point start) {
    return std::chrono::duration<double, std::milli>(HostClock::now() - start).count();
}

double stop_gpu_timer_ms(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream) {
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return static_cast<double>(elapsed_ms);
}

ProfileConfig parse_config(int argc, char** argv) {
    ProfileConfig cfg{};
    cfg.mode = ProfileMode::Aggregate;
    cfg.mode_name = "aggregate";
    cfg.case_filter = argc >= 3 ? argv[2] : nullptr;
    cfg.variance_iters = 200;

    if (argc >= 2) {
        if (std::strcmp(argv[1], "variance") == 0) {
            cfg.mode = ProfileMode::Variance;
            cfg.mode_name = "variance";
        } else if (std::strcmp(argv[1], "aggregate") != 0) {
            std::fprintf(stderr, "unknown profile mode '%s' (use aggregate or variance)\n", argv[1]);
            std::exit(2);
        }
    }
    return cfg;
}

bool case_selected(const BenchCase& c, const ProfileConfig& cfg) {
    return cfg.case_filter == nullptr || std::strcmp(c.name, cfg.case_filter) == 0;
}

__device__ unsigned lcg(unsigned x) {
    return 1664525u * x + 1013904223u;
}

__device__ float rand_centered(unsigned seed) {
    seed = lcg(seed);
    float u = static_cast<float>(seed & 0x00ffffffu) / 16777216.0f;
    return 2.0f * u - 1.0f;
}

__global__ void fill_input_kernel(float* A, int batch, int n, int cond, int kind) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int col = static_cast<int>(idx % n);
        int row = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        int local_kind = kind;
        if (kind == static_cast<int>(CaseKind::Mixed)) {
            int tag = b % 4;
            local_kind = tag == 0 ? static_cast<int>(CaseKind::Dense)
                       : tag == 1 ? static_cast<int>(CaseKind::RankDef)
                       : tag == 2 ? static_cast<int>(CaseKind::Clustered)
                                  : static_cast<int>(CaseKind::NearRank);
        }

        float v = rand_centered(static_cast<unsigned>(idx) ^ 0x9e3779b9u);
        float scale = 1.0f;
        if (cond > 0 && n > 1) {
            scale = powf(10.0f, -static_cast<float>(cond) * col / (n - 1));
        }

        if (local_kind == static_cast<int>(CaseKind::RankDef)) {
            int rank = max(1, (3 * n) / 4);
            v = col >= rank ? 0.0f : v * scale;
        } else if (local_kind == static_cast<int>(CaseKind::Clustered)) {
            float eps = 1.1920928955078125e-7f;
            if (col >= n / 2) scale = 4.0f * eps;
            if (n >= 8 && col >= n / 2 - 2 && col < n / 2 + 2) scale = sqrtf(eps);
            v *= scale;
        } else if (local_kind == static_cast<int>(CaseKind::NearRank)) {
            int rank = max(1, (3 * n) / 4);
            if (col >= rank) {
                int src_col = col - rank;
                unsigned src_idx = static_cast<unsigned>((static_cast<size_t>(b) * n + row) * n + src_col);
                v = rand_centered(src_idx ^ 0x9e3779b9u) + 1.0e-4f * v;
            }
        } else {
            v *= scale;
        }

        A[idx] = v;
    }
}

__global__ void finite_check_kernel(const float* H, const float* tau, size_t h_count,
                                    size_t tau_count, int* bad) {
    size_t total = h_count + tau_count;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < total; idx += stride) {
        float v = idx < h_count ? H[idx] : tau[idx - h_count];
        if (!isfinite(v)) {
            atomicExch(bad, 1);
        }
    }
}

void fill_input(float* A, const BenchCase& c, cudaStream_t stream) {
    size_t total = static_cast<size_t>(c.batch) * c.n * c.n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    fill_input_kernel<<<blocks, threads, 0, stream>>>(
        A, c.batch, c.n, c.cond, static_cast<int>(c.kind));
}

bool finite_check(const float* H, const float* tau, size_t h_count, size_t tau_count,
                  cudaStream_t stream) {
    int* d_bad = nullptr;
    int h_bad = 0;
    CHECK_CUDA(cudaMalloc(&d_bad, sizeof(int)));
    CHECK_CUDA(cudaMemsetAsync(d_bad, 0, sizeof(int), stream));
    size_t total = h_count + tau_count;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    finite_check_kernel<<<blocks, threads, 0, stream>>>(H, tau, h_count, tau_count, d_bad);
    CHECK_CUDA(cudaMemcpyAsync(&h_bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(d_bad));
    return h_bad == 0;
}

BenchStats calculate_stats(const std::vector<double>& durations_ms, bool finite) {
    BenchStats stats{};
    stats.runs = static_cast<int>(durations_ms.size());
    stats.finite = finite;
    if (durations_ms.empty()) {
        return stats;
    }

    double total = 0.0;
    stats.best = durations_ms[0];
    stats.worst = durations_ms[0];
    for (double value : durations_ms) {
        total += value;
        stats.best = std::min(stats.best, value);
        stats.worst = std::max(stats.worst, value);
    }
    stats.mean = total / static_cast<double>(stats.runs);

    double variance = 0.0;
    for (double value : durations_ms) {
        double diff = value - stats.mean;
        variance += diff * diff;
    }
    stats.std = stats.runs > 1 ? std::sqrt(variance / static_cast<double>(stats.runs - 1)) : 0.0;
    stats.err = stats.runs > 0 ? stats.std / std::sqrt(static_cast<double>(stats.runs)) : 0.0;
    return stats;
}

void print_harness_profile(const std::vector<HarnessProfile>& profiles) {
    std::printf("\nharness-level profile\n");
    std::printf("note: benchmark matrices are generated directly on GPU; matrix H2D bytes = 0, matrix D2H bytes = 0.\n");
    std::printf("note: finite_check copies one int back to host per case; that cost is included in finite_ms.\n");
    std::printf("%-28s %9s %9s %9s %9s %9s %10s %9s %9s %8s\n",
                "case", "total_ms", "alloc", "fill_gpu", "fill_wall",
                "warmup", "timed_qr", "finite", "free", "timed%");

    for (const HarnessProfile& p : profiles) {
        double timed_pct = p.total_wall_ms > 0.0
                         ? 100.0 * p.timed_qr_total_ms / p.total_wall_ms
                         : 0.0;
        std::printf("%-28s %9.3f %9.3f %9.3f %9.3f %9.3f %10.3f %9.3f %9.3f %7.2f%%\n",
                    p.name, p.total_wall_ms, p.alloc_ms, p.fill_gpu_ms,
                    p.fill_wall_ms, p.warmup_wall_ms, p.timed_qr_total_ms,
                    p.finite_check_ms, p.free_ms, timed_pct);
    }

    std::fflush(stdout);
}

BenchStats run_case(const BenchCase& c, cudaStream_t stream, HarnessProfile* profile,
                    const ProfileConfig& cfg) {
    auto total_start = HostClock::now();
    *profile = HarnessProfile{};
    profile->name = c.name;
    profile->batch = c.batch;
    profile->n = c.n;
    profile->warmup = c.warmup;
    profile->iters = cfg.mode == ProfileMode::Variance ? cfg.variance_iters : c.iters;

    std::printf("running %-28s batch=%d n=%d\n", c.name, c.batch, c.n);
    std::fflush(stdout);

    size_t h_count = static_cast<size_t>(c.batch) * c.n * c.n;
    size_t tau_count = static_cast<size_t>(c.batch) * c.n;
    float* A = nullptr;
    float* H = nullptr;
    float* tau = nullptr;

    auto phase_start = HostClock::now();
    CHECK_CUDA(cudaMalloc(&A, h_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&H, h_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&tau, tau_count * sizeof(float)));
    profile->alloc_ms = elapsed_host_ms(phase_start);

    cudaEvent_t phase_event_start, phase_event_stop;
    CHECK_CUDA(cudaEventCreate(&phase_event_start));
    CHECK_CUDA(cudaEventCreate(&phase_event_stop));

    phase_start = HostClock::now();
    CHECK_CUDA(cudaEventRecord(phase_event_start, stream));
    fill_input(A, c, stream);
    CHECK_CUDA(cudaGetLastError());
    profile->fill_gpu_ms = stop_gpu_timer_ms(phase_event_start, phase_event_stop, stream);
    profile->fill_wall_ms = elapsed_host_ms(phase_start);

    phase_start = HostClock::now();
    if (cfg.mode == ProfileMode::Variance) {
        setenv("QR_PROFILE_RECORD", "0", 1);
    }
    for (int i = 0; i < c.warmup; ++i) {
        qr_custom_kernel_cuda(A, H, tau, c.batch, c.n, stream);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    profile->warmup_wall_ms = elapsed_host_ms(phase_start);

    phase_start = HostClock::now();
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    profile->event_setup_ms = elapsed_host_ms(phase_start);

    std::vector<double> durations_ms;
    durations_ms.reserve(static_cast<size_t>(profile->iters));
    if (cfg.mode == ProfileMode::Variance) {
        setenv("QR_PROFILE_RECORD", "1", 1);
    }
    for (int i = 0; i < profile->iters; ++i) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        qr_custom_kernel_cuda(A, H, tau, c.batch, c.n, stream);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        durations_ms.push_back(static_cast<double>(elapsed_ms));
        profile->timed_qr_total_ms += static_cast<double>(elapsed_ms);
    }
    if (cfg.mode == ProfileMode::Variance) {
        setenv("QR_PROFILE_RECORD", "0", 1);
    }
    phase_start = HostClock::now();
    bool finite = finite_check(H, tau, h_count, tau_count, stream);
    profile->finite_check_ms = elapsed_host_ms(phase_start);
    BenchStats stats = calculate_stats(durations_ms, finite);
    profile->timed_qr_mean_ms = stats.mean;
    profile->finite = finite;

    std::printf("%-28s %6d %6d %8d %10.4f %s\n",
                c.name, c.batch, c.n, stats.runs, stats.mean,
                finite ? "finite" : "bad");
    std::fflush(stdout);

    phase_start = HostClock::now();
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(phase_event_start));
    CHECK_CUDA(cudaEventDestroy(phase_event_stop));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(H));
    CHECK_CUDA(cudaFree(tau));
    profile->free_ms = elapsed_host_ms(phase_start);
    profile->total_wall_ms = elapsed_host_ms(total_start);
    return stats;
}

int main(int argc, char** argv) {
    ProfileConfig cfg = parse_config(argc, argv);
    if (cfg.mode == ProfileMode::Variance) {
        setenv("QR_PROFILE_MODE", "variance", 1);
        setenv("QR_PROFILE_RECORD", "0", 1);
    } else {
        setenv("QR_PROFILE_MODE", "aggregate", 1);
    }

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::printf("device: %s\n", prop.name);
    std::printf("profile_mode: %s\n", cfg.mode_name);
    if (cfg.case_filter != nullptr) {
        std::printf("case_filter: %s\n", cfg.case_filter);
    }
    std::printf("%-28s %6s %6s %8s %10s %s\n",
                "case", "batch", "n", "runs", "mean_ms", "status");
    std::fflush(stdout);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    std::vector<double> mean_ms;
    std::vector<HarnessProfile> profiles;
    profiles.reserve(sizeof(kCases) / sizeof(kCases[0]));
    bool passed = true;
    for (const BenchCase& c : kCases) {
        if (!case_selected(c, cfg)) {
            continue;
        }
        HarnessProfile profile{};
        BenchStats stats = run_case(c, stream, &profile, cfg);
        profiles.push_back(profile);
        if (stats.finite && stats.mean > 0.0) {
            mean_ms.push_back(stats.mean);
        } else {
            passed = false;
        }
    }
    CHECK_CUDA(cudaStreamDestroy(stream));

    if (!mean_ms.empty()) {
        double log_sum = 0.0;
        for (double value : mean_ms) {
            log_sum += std::log(value);
        }
        double score = std::exp(log_sum / static_cast<double>(mean_ms.size()));
        std::printf("score: %.9f\n", score);
    }
    std::printf("check: %s\n", passed ? "pass" : "fail");
    print_harness_profile(profiles);
    return passed ? 0 : 1;
}
