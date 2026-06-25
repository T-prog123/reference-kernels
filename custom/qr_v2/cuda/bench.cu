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
    double p50;
    double p95;
    double worst;
    bool finite;
    bool converged;
};

enum class BenchMode {
    Quick,
    Convergence,
    Leaderboard,
};

struct BenchConfig {
    BenchMode mode;
    const char* mode_name;
    const char* case_filter;
    int max_repeats;
    double target_rel_err;
    double max_measured_ms;
    double min_wall_ms;
    double max_wall_ms;
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

BenchStats calculate_stats(const std::vector<double>& durations_ms, bool finite,
                           bool converged = false) {
    BenchStats stats{};
    stats.runs = static_cast<int>(durations_ms.size());
    stats.finite = finite;
    stats.converged = converged;
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

    std::vector<double> sorted = durations_ms;
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double p) {
        int idx = static_cast<int>(std::ceil(p * static_cast<double>(sorted.size()))) - 1;
        idx = std::max(0, std::min(idx, static_cast<int>(sorted.size()) - 1));
        return sorted[static_cast<size_t>(idx)];
    };
    stats.p50 = percentile(0.50);
    stats.p95 = percentile(0.95);
    return stats;
}

BenchConfig parse_config(int argc, char** argv) {
    BenchConfig cfg{};
    cfg.mode = BenchMode::Quick;
    cfg.mode_name = "quick";
    cfg.case_filter = argc >= 3 ? argv[2] : nullptr;
    cfg.max_repeats = 200;
    cfg.target_rel_err = 0.001;
    cfg.max_measured_ms = 10000.0;
    cfg.min_wall_ms = 100.0;
    cfg.max_wall_ms = 120000.0;

    if (argc >= 2) {
        if (std::strcmp(argv[1], "convergence") == 0) {
            cfg.mode = BenchMode::Convergence;
            cfg.mode_name = "convergence";
        } else if (std::strcmp(argv[1], "leaderboard") == 0) {
            cfg.mode = BenchMode::Leaderboard;
            cfg.mode_name = "leaderboard";
            cfg.max_repeats = 1000;
            cfg.max_measured_ms = 30000.0;
        } else if (std::strcmp(argv[1], "quick") != 0) {
            std::fprintf(stderr, "unknown bench mode '%s' (use quick, convergence, leaderboard)\n", argv[1]);
            std::exit(2);
        }
    }
    return cfg;
}

bool case_selected(const BenchCase& c, const BenchConfig& cfg) {
    return cfg.case_filter == nullptr || std::strcmp(c.name, cfg.case_filter) == 0;
}

BenchStats run_case(const BenchCase& c, cudaStream_t stream, const BenchConfig& cfg) {
    std::printf("running %-28s batch=%d n=%d\n", c.name, c.batch, c.n);
    std::fflush(stdout);

    size_t h_count = static_cast<size_t>(c.batch) * c.n * c.n;
    size_t tau_count = static_cast<size_t>(c.batch) * c.n;
    float* A = nullptr;
    float* H = nullptr;
    float* tau = nullptr;

    CHECK_CUDA(cudaMalloc(&A, h_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&H, h_count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&tau, tau_count * sizeof(float)));

    fill_input(A, c, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    for (int i = 0; i < c.warmup; ++i) {
        qr_custom_kernel_cuda(A, H, tau, c.batch, c.n, stream);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<double> durations_ms;
    int repeat_limit = cfg.mode == BenchMode::Quick ? c.iters : cfg.max_repeats;
    durations_ms.reserve(static_cast<size_t>(repeat_limit));
    bool converged = false;
    auto bm_start = HostClock::now();
    for (int i = 0; i < repeat_limit; ++i) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        qr_custom_kernel_cuda(A, H, tau, c.batch, c.n, stream);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        durations_ms.push_back(static_cast<double>(elapsed_ms));

        if (cfg.mode != BenchMode::Quick && i > 1) {
            double wall_ms = elapsed_host_ms(bm_start);
            if (wall_ms > cfg.min_wall_ms) {
                BenchStats partial = calculate_stats(durations_ms, true);
                double rel_err = partial.mean > 0.0 ? partial.err / partial.mean : 0.0;
                if (rel_err < cfg.target_rel_err) {
                    converged = true;
                    break;
                }
                if (partial.mean * partial.runs > cfg.max_measured_ms ||
                    wall_ms > cfg.max_wall_ms) {
                    break;
                }
            }
        }
    }
    bool finite = finite_check(H, tau, h_count, tau_count, stream);
    BenchStats stats = calculate_stats(durations_ms, finite, converged);

    double err_pct = stats.mean > 0.0 ? 100.0 * stats.err / stats.mean : 0.0;
    std::printf("%-28s %6d %6d %8d %10.4f %7.3f %10.4f %10.4f %10.4f %10.4f %9s %s\n",
                c.name, c.batch, c.n, stats.runs, stats.mean, err_pct,
                stats.best, stats.p50, stats.p95, stats.worst,
                stats.converged ? "yes" : "no",
                finite ? "finite" : "bad");
    std::fflush(stdout);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(H));
    CHECK_CUDA(cudaFree(tau));
    return stats;
}

int main(int argc, char** argv) {
    BenchConfig cfg = parse_config(argc, argv);
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::printf("device: %s\n", prop.name);
    std::printf("mode: %s\n", cfg.mode_name);
    if (cfg.case_filter != nullptr) {
        std::printf("case_filter: %s\n", cfg.case_filter);
    }
    std::printf("%-28s %6s %6s %8s %10s %7s %10s %10s %10s %10s %9s %s\n",
                "case", "batch", "n", "runs", "mean_ms", "err%",
                "best", "p50", "p95", "worst", "converged", "status");
    std::fflush(stdout);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    std::vector<double> mean_ms;
    bool passed = true;
    for (const BenchCase& c : kCases) {
        if (!case_selected(c, cfg)) {
            continue;
        }
        BenchStats stats = run_case(c, stream, cfg);
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
    return passed ? 0 : 1;
}
