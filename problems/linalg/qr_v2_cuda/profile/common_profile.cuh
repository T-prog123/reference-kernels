#pragma once

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../qr_kernel.h"

#define PROFILE_CHECK_CUDA(expr)                                                \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(_err));                             \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

enum class ProfileCaseKind {
    Dense,
    Mixed,
    RankDef,
    Clustered,
    NearRank,
};

struct ProfileCase {
    const char* name;
    int batch;
    int n;
    int cond;
    ProfileCaseKind kind;
    int warmup;
    int iters;
};

struct ProfileStats {
    int runs = 0;
    double mean_ms = 0.0;
    double best_ms = 0.0;
    double worst_ms = 0.0;
};

struct ProfileInput {
    float* A = nullptr;
    float* H = nullptr;
    float* tau = nullptr;
};

__device__ unsigned profile_lcg(unsigned x) {
    return 1664525u * x + 1013904223u;
}

__device__ float profile_rand_centered(unsigned seed) {
    seed = profile_lcg(seed);
    float u = static_cast<float>(seed & 0x00ffffffu) / 16777216.0f;
    return 2.0f * u - 1.0f;
}

__global__ void profile_fill_input_kernel(
    float* A,
    int batch,
    int n,
    int cond,
    int kind,
    unsigned seed)
{
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int col = static_cast<int>(idx % n);
        int row = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        int local_kind = kind;
        if (kind == static_cast<int>(ProfileCaseKind::Mixed)) {
            int tag = b % 4;
            local_kind = tag == 0 ? static_cast<int>(ProfileCaseKind::Dense)
                       : tag == 1 ? static_cast<int>(ProfileCaseKind::RankDef)
                       : tag == 2 ? static_cast<int>(ProfileCaseKind::Clustered)
                                  : static_cast<int>(ProfileCaseKind::NearRank);
        }

        float v = profile_rand_centered(static_cast<unsigned>(idx) ^ seed);
        float scale = 1.0f;
        if (cond > 0 && n > 1) {
            scale = powf(10.0f, -static_cast<float>(cond) * col / (n - 1));
        }

        if (local_kind == static_cast<int>(ProfileCaseKind::RankDef)) {
            int rank = max(1, (3 * n) / 4);
            v = col >= rank ? 0.0f : v * scale;
        } else if (local_kind == static_cast<int>(ProfileCaseKind::Clustered)) {
            float eps = 1.1920928955078125e-7f;
            if (col >= n / 2) scale = 4.0f * eps;
            if (n >= 8 && col >= n / 2 - 2 && col < n / 2 + 2) scale = sqrtf(eps);
            v *= scale;
        } else if (local_kind == static_cast<int>(ProfileCaseKind::NearRank)) {
            int rank = max(1, (3 * n) / 4);
            if (col >= rank) {
                int src_col = col - rank;
                unsigned src_idx = static_cast<unsigned>((static_cast<size_t>(b) * n + row) * n + src_col);
                v = profile_rand_centered(src_idx ^ seed) + 1.0e-4f * v;
            }
        } else {
            v *= scale;
        }

        A[idx] = v;
    }
}

int profile_outer_input_count(const ProfileCase& c) {
    constexpr size_t kInputBytesTarget = 256ull * 1024ull * 1024ull;
    constexpr int kMaxInputs = 50;
    size_t bytes = static_cast<size_t>(c.batch) * c.n * c.n * sizeof(float);
    if (bytes == 0) {
        return 1;
    }
    return std::max(1, std::min(kMaxInputs, static_cast<int>(kInputBytesTarget / bytes)));
}

void profile_fill_input(float* A, const ProfileCase& c, unsigned seed, cudaStream_t stream) {
    size_t total = static_cast<size_t>(c.batch) * c.n * c.n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    profile_fill_input_kernel<<<blocks, threads, 0, stream>>>(
        A, c.batch, c.n, c.cond, static_cast<int>(c.kind), seed);
}

ProfileStats run_profile_case(const ProfileCase& c) {
    cudaStream_t stream;
    PROFILE_CHECK_CUDA(cudaStreamCreate(&stream));

    int outer_count = profile_outer_input_count(c);
    size_t h_count = static_cast<size_t>(c.batch) * c.n * c.n;
    size_t tau_count = static_cast<size_t>(c.batch) * c.n;
    std::vector<ProfileInput> inputs(static_cast<size_t>(outer_count));

    for (int i = 0; i < outer_count; ++i) {
        PROFILE_CHECK_CUDA(cudaMalloc(&inputs[i].A, h_count * sizeof(float)));
        PROFILE_CHECK_CUDA(cudaMalloc(&inputs[i].H, h_count * sizeof(float)));
        PROFILE_CHECK_CUDA(cudaMalloc(&inputs[i].tau, tau_count * sizeof(float)));
        profile_fill_input(inputs[i].A, c, 0x9e3779b9u + static_cast<unsigned>(42 * i), stream);
    }
    PROFILE_CHECK_CUDA(cudaGetLastError());
    PROFILE_CHECK_CUDA(cudaStreamSynchronize(stream));

    for (int i = 0; i < c.warmup; ++i) {
        for (ProfileInput& input : inputs) {
            qr_custom_kernel_cuda(input.A, input.H, input.tau, c.batch, c.n, stream);
        }
    }
    PROFILE_CHECK_CUDA(cudaGetLastError());
    PROFILE_CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    PROFILE_CHECK_CUDA(cudaEventCreate(&start));
    PROFILE_CHECK_CUDA(cudaEventCreate(&stop));

    ProfileStats stats{};
    stats.runs = c.iters;
    stats.best_ms = INFINITY;

    PROFILE_CHECK_CUDA(cudaProfilerStart());
    for (int i = 0; i < c.iters; ++i) {
        PROFILE_CHECK_CUDA(cudaEventRecord(start, stream));
        for (ProfileInput& input : inputs) {
            qr_custom_kernel_cuda(input.A, input.H, input.tau, c.batch, c.n, stream);
        }
        PROFILE_CHECK_CUDA(cudaEventRecord(stop, stream));
        PROFILE_CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        PROFILE_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double per_input_ms = static_cast<double>(elapsed_ms) / static_cast<double>(outer_count);
        stats.mean_ms += per_input_ms;
        stats.best_ms = std::min(stats.best_ms, per_input_ms);
        stats.worst_ms = std::max(stats.worst_ms, per_input_ms);
    }
    PROFILE_CHECK_CUDA(cudaProfilerStop());
    stats.mean_ms /= static_cast<double>(c.iters);

    PROFILE_CHECK_CUDA(cudaEventDestroy(start));
    PROFILE_CHECK_CUDA(cudaEventDestroy(stop));
    for (ProfileInput& input : inputs) {
        PROFILE_CHECK_CUDA(cudaFree(input.A));
        PROFILE_CHECK_CUDA(cudaFree(input.H));
        PROFILE_CHECK_CUDA(cudaFree(input.tau));
    }
    PROFILE_CHECK_CUDA(cudaStreamDestroy(stream));
    return stats;
}

int profile_main(const ProfileCase& c) {
    int device = 0;
    PROFILE_CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    PROFILE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int outer_count = profile_outer_input_count(c);
    std::printf("device: %s\n", prop.name);
    std::printf("profile_case: %s\n", c.name);
    std::printf("batch: %d\n", c.batch);
    std::printf("n: %d\n", c.n);
    std::printf("outer_inputs: %d\n", outer_count);

    ProfileStats stats = run_profile_case(c);
    std::printf("runs: %d\n", stats.runs);
    std::printf("mean_ms: %.6f\n", stats.mean_ms);
    std::printf("best_ms: %.6f\n", stats.best_ms);
    std::printf("worst_ms: %.6f\n", stats.worst_ms);
    return 0;
}
