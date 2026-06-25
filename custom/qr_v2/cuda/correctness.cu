#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

#define CHECK_CUSOLVER(expr)                                                     \
    do {                                                                        \
        cusolverStatus_t _err = (expr);                                         \
        if (_err != CUSOLVER_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuSOLVER error %s:%d: status=%d\n", __FILE__, \
                         __LINE__, static_cast<int>(_err));                     \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(expr)                                                       \
    do {                                                                        \
        cublasStatus_t _err = (expr);                                           \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__,   \
                         __LINE__, static_cast<int>(_err));                     \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

enum class CaseKind {
    Dense,
    RankDef,
    NearRank,
    Clustered,
    Band,
    RowScale,
    NearCollinear,
    Upper,
    Mixed,
};

struct TestCase {
    const char* name;
    int batch;
    int n;
    int cond;
    unsigned seed;
    CaseKind kind;
};

static const TestCase kCases[] = {
    {"dense_b20_n32_c1_s53124", 20, 32, 1, 53124u, CaseKind::Dense},
    {"dense_b40_n176_c1_s3321", 40, 176, 1, 3321u, CaseKind::Dense},
    {"dense_b40_n352_c1_s1200", 40, 352, 1, 1200u, CaseKind::Dense},
    {"dense_b16_n512_c2_s32523", 16, 512, 2, 32523u, CaseKind::Dense},
    {"dense_b4_n1024_c2_s4327", 4, 1024, 2, 4327u, CaseKind::Dense},
    {"dense_b1_n4096_c1_s75342", 1, 4096, 1, 75342u, CaseKind::Dense},
    {"dense_b16_n512_c4_s32524", 16, 512, 4, 32524u, CaseKind::Dense},
    {"rankdef_b16_n512_s32525", 16, 512, 0, 32525u, CaseKind::RankDef},
    {"clustered_b16_n512_s32526", 16, 512, 0, 32526u, CaseKind::Clustered},
    {"band_b16_n512_s32527", 16, 512, 0, 32527u, CaseKind::Band},
    {"rowscale_b16_n512_s32528", 16, 512, 0, 32528u, CaseKind::RowScale},
    {"nearcollinear_b16_n512_s32529", 16, 512, 0, 32529u, CaseKind::NearCollinear},
    {"dense_b4_n1024_c4_s4328", 4, 1024, 4, 4328u, CaseKind::Dense},
    {"rankdef_b4_n1024_s4329", 4, 1024, 0, 4329u, CaseKind::RankDef},
    {"nearrank_b4_n1024_s4330", 4, 1024, 0, 4330u, CaseKind::NearRank},
    {"clustered_b4_n1024_s4331", 4, 1024, 0, 4331u, CaseKind::Clustered},
    {"dense_b2_n2048_c2_s224466", 2, 2048, 2, 224466u, CaseKind::Dense},
    {"rankdef_b2_n2048_s224467", 2, 2048, 0, 224467u, CaseKind::RankDef},
    {"upper_b1_n4096_s75343", 1, 4096, 0, 75343u, CaseKind::Upper},
    {"mixed_b16_n512_c2_s32530", 16, 512, 2, 32530u, CaseKind::Mixed},
    {"mixed_b4_n1024_c2_s4332", 4, 1024, 2, 4332u, CaseKind::Mixed},
    {"mixed_b2_n2048_c2_s224468", 2, 2048, 2, 224468u, CaseKind::Mixed},
};

__device__ unsigned lcg(unsigned x) {
    return 1664525u * x + 1013904223u;
}

__device__ float rand_centered(unsigned seed) {
    seed = lcg(seed);
    float u = static_cast<float>(seed & 0x00ffffffu) / 16777216.0f;
    return 2.0f * u - 1.0f;
}

__device__ float col_scale(int col, int n, int cond) {
    if (cond <= 0 || n <= 1) return 1.0f;
    return powf(10.0f, -static_cast<float>(cond) * col / (n - 1));
}

__device__ int mixed_kind_for_batch(int b) {
    int tag = b % 7;
    if (tag < 3) return static_cast<int>(CaseKind::Dense);
    if (tag == 3) return static_cast<int>(CaseKind::RankDef);
    if (tag == 4) return static_cast<int>(CaseKind::NearRank);
    if (tag == 5) return static_cast<int>(CaseKind::Clustered);
    return static_cast<int>(CaseKind::NearCollinear);
}

__global__ void fill_input_kernel(float* A, int batch, int n, int cond,
                                  unsigned seed, int kind) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int col = static_cast<int>(idx % n);
        int row = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        int local_kind = kind == static_cast<int>(CaseKind::Mixed)
                             ? mixed_kind_for_batch(b)
                             : kind;
        float v = rand_centered(seed ^ static_cast<unsigned>(idx) ^ 0x9e3779b9u);
        float scale = col_scale(col, n, cond);

        if (local_kind == static_cast<int>(CaseKind::Upper)) {
            v = row <= col ? v * scale : 0.0f;
            if (row == col) {
                float t = n > 1 ? static_cast<float>(row) / (n - 1) : 0.0f;
                v += 1.0f - 0.75f * t;
            }
        } else if (local_kind == static_cast<int>(CaseKind::RankDef)) {
            int rank = max(1, (3 * n) / 4);
            v = col >= rank ? 0.0f : v * scale;
        } else if (local_kind == static_cast<int>(CaseKind::NearRank)) {
            int rank = max(1, (3 * n) / 4);
            if (col >= rank) {
                int src_col = col - rank;
                size_t src_idx = (static_cast<size_t>(b) * n + row) * n + src_col;
                v = rand_centered(seed ^ static_cast<unsigned>(src_idx) ^ 0x9e3779b9u) + 1.0e-5f * v;
            }
            v *= scale;
        } else if (local_kind == static_cast<int>(CaseKind::Clustered)) {
            float eps = 1.1920928955078125e-7f;
            scale = 1.0f;
            if (col >= n / 2) scale = 4.0f * eps;
            if (n >= 8 && col >= n / 2 - 2 && col < n / 2 + 2) scale = sqrtf(eps);
            v *= scale;
        } else if (local_kind == static_cast<int>(CaseKind::Band)) {
            int bandwidth = max(2, min(32, n / 32));
            v = abs(row - col) <= bandwidth ? v * scale : 0.0f;
            if (row == col) {
                float t = n > 1 ? static_cast<float>(row) / (n - 1) : 0.0f;
                v += 1.0f - 0.5f * t;
            }
        } else if (local_kind == static_cast<int>(CaseKind::RowScale)) {
            v *= col_scale(row, n, max(cond, 4));
        } else if (local_kind == static_cast<int>(CaseKind::NearCollinear)) {
            unsigned base_idx = seed ^ static_cast<unsigned>((static_cast<size_t>(b) * n + row) * n) ^ 0x9e3779b9u;
            v = rand_centered(base_idx) + 1.0e-4f * v;
            v *= scale;
        } else {
            v *= scale;
        }
        A[idx] = v;
    }
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

__global__ void float_to_double_kernel(const float* src, double* dst, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < n; idx += stride) {
        dst[idx] = static_cast<double>(src[idx]);
    }
}

__global__ void extract_r_kernel(const float* Hcol, double* R, int n) {
    size_t total = static_cast<size_t>(n) * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < total; idx += stride) {
        int i = static_cast<int>(idx % n);
        int j = static_cast<int>(idx / n);
        R[idx] = i <= j ? static_cast<double>(Hcol[idx]) : 0.0;
    }
}

__global__ void subtract_kernel(double* value, const double* rhs, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < n; idx += stride) {
        value[idx] -= rhs[idx];
    }
}

__global__ void subtract_eye_kernel(double* value, int n) {
    size_t total = static_cast<size_t>(n) * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < total; idx += stride) {
        int i = static_cast<int>(idx % n);
        int j = static_cast<int>(idx / n);
        if (i == j) value[idx] -= 1.0;
    }
}

__global__ void column_abs_sums_kernel(const double* M, double* sums, int n) {
    extern __shared__ double scratch[];
    int col = blockIdx.x;
    double sum = 0.0;
    for (int row = threadIdx.x; row < n; row += blockDim.x) {
        sum += fabs(M[col * static_cast<size_t>(n) + row]);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[col] = scratch[0];
}

double matrix_l1_norm(const double* M, double* sums, std::vector<double>& host_sums,
                      int n, cudaStream_t stream) {
    column_abs_sums_kernel<<<n, 256, 256 * sizeof(double), stream>>>(M, sums, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpyAsync(host_sums.data(), sums, static_cast<size_t>(n) * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return *std::max_element(host_sums.begin(), host_sums.end());
}

void fill_input(float* A, const TestCase& c, cudaStream_t stream) {
    size_t total = static_cast<size_t>(c.batch) * c.n * c.n;
    int threads = 256;
    int blocks = std::min<int>((total + threads - 1) / threads, 4096);
    fill_input_kernel<<<blocks, threads, 0, stream>>>(
        A, c.batch, c.n, c.cond, c.seed, static_cast<int>(c.kind));
    CHECK_CUDA(cudaGetLastError());
}

void run_case(const TestCase& c, cusolverDnHandle_t solver, cublasHandle_t blas,
              cudaStream_t stream) {
    int n = c.n;
    size_t matrix_elems = static_cast<size_t>(n) * n;
    size_t total_elems = static_cast<size_t>(c.batch) * matrix_elems;
    const double eps32 = 1.1920928955078125e-7;

    float *Arow = nullptr, *Acol = nullptr, *Hrow = nullptr, *Hcol = nullptr;
    float *tau = nullptr, *Qf = nullptr, *orgqr_work = nullptr;
    double *Ad = nullptr, *Qd = nullptr, *R = nullptr, *Work = nullptr, *sums = nullptr;
    int* info = nullptr;

    CHECK_CUDA(cudaMalloc(&Arow, total_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Acol, total_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Hrow, total_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Hcol, total_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&tau, static_cast<size_t>(c.batch) * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Qf, matrix_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Ad, matrix_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Qd, matrix_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&R, matrix_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&Work, matrix_elems * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&sums, static_cast<size_t>(n) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&info, sizeof(int)));

    fill_input(Arow, c, stream);
    int elem_blocks = std::min<int>((total_elems + 255) / 256, 4096);
    row_to_col_major_kernel<<<elem_blocks, 256, 0, stream>>>(Arow, Acol, c.batch, n);
    CHECK_CUDA(cudaGetLastError());

    qr_custom_kernel_cuda(Arow, Hrow, tau, c.batch, n, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    row_to_col_major_kernel<<<elem_blocks, 256, 0, stream>>>(Hrow, Hcol, c.batch, n);
    CHECK_CUDA(cudaGetLastError());

    int orgqr_lwork = 0;
    CHECK_CUSOLVER(cusolverDnSorgqr_bufferSize(
        solver, n, n, n, Hcol, n, tau, &orgqr_lwork));
    CHECK_CUDA(cudaMalloc(&orgqr_work, static_cast<size_t>(orgqr_lwork) * sizeof(float)));

    std::vector<double> host_sums(static_cast<size_t>(n));
    double worst_factor_scaled = 0.0;
    double worst_orth_scaled = 0.0;
    const double one = 1.0;
    const double zero = 0.0;

    for (int b = 0; b < c.batch; ++b) {
        const float* Ab = Acol + static_cast<size_t>(b) * matrix_elems;
        const float* Hb = Hcol + static_cast<size_t>(b) * matrix_elems;
        const float* taub = tau + static_cast<size_t>(b) * n;

        CHECK_CUDA(cudaMemcpyAsync(Qf, Hb, matrix_elems * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        CHECK_CUSOLVER(cusolverDnSorgqr(solver, n, n, n, Qf, n, taub, orgqr_work,
                                        orgqr_lwork, info));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        int blocks = std::min<int>((matrix_elems + 255) / 256, 4096);
        float_to_double_kernel<<<blocks, 256, 0, stream>>>(Ab, Ad, matrix_elems);
        float_to_double_kernel<<<blocks, 256, 0, stream>>>(Qf, Qd, matrix_elems);
        extract_r_kernel<<<blocks, 256, 0, stream>>>(Hb, R, n);
        CHECK_CUDA(cudaGetLastError());

        double a_norm = matrix_l1_norm(Ad, sums, host_sums, n, stream);
        CHECK_CUBLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                                 &one, Qd, n, Ad, n, &zero, Work, n));
        subtract_kernel<<<blocks, 256, 0, stream>>>(R, Work, matrix_elems);
        CHECK_CUDA(cudaGetLastError());
        double factor_residual = matrix_l1_norm(R, sums, host_sums, n, stream);
        double factor_scaled = factor_residual / (eps32 * std::max(n, 1) * std::max(a_norm, 1.0e-30));
        worst_factor_scaled = std::max(worst_factor_scaled, factor_scaled);

        CHECK_CUBLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                                 &one, Qd, n, Qd, n, &zero, Work, n));
        subtract_eye_kernel<<<blocks, 256, 0, stream>>>(Work, n);
        CHECK_CUDA(cudaGetLastError());
        double orth_residual = matrix_l1_norm(Work, sums, host_sums, n, stream);
        double orth_scaled = orth_residual / (eps32 * std::max(n, 1));
        worst_orth_scaled = std::max(worst_orth_scaled, orth_scaled);
    }

    const char* status = (worst_factor_scaled <= 20.0 && worst_orth_scaled <= 100.0) ? "pass" : "fail";
    std::printf("%-32s batch=%4d n=%4d scaled_factor_residual=%9.4g "
                "factor_budget=%7.3f%% scaled_orthogonality_residual=%9.4g "
                "orth_budget=%7.3f%% %s\n",
                c.name, c.batch, c.n, worst_factor_scaled,
                100.0 * worst_factor_scaled / 20.0,
                worst_orth_scaled, 100.0 * worst_orth_scaled / 100.0, status);
    std::fflush(stdout);

    CHECK_CUDA(cudaFree(orgqr_work));
    CHECK_CUDA(cudaFree(info));
    CHECK_CUDA(cudaFree(sums));
    CHECK_CUDA(cudaFree(Work));
    CHECK_CUDA(cudaFree(R));
    CHECK_CUDA(cudaFree(Qd));
    CHECK_CUDA(cudaFree(Ad));
    CHECK_CUDA(cudaFree(Qf));
    CHECK_CUDA(cudaFree(tau));
    CHECK_CUDA(cudaFree(Hcol));
    CHECK_CUDA(cudaFree(Hrow));
    CHECK_CUDA(cudaFree(Acol));
    CHECK_CUDA(cudaFree(Arow));
}

int main() {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::printf("device: %s\n", prop.name);
    std::printf("selected kernel scaled residuals; pass limits are factor<=20 and orth<=100\n");

    cudaStream_t stream;
    cusolverDnHandle_t solver = nullptr;
    cublasHandle_t blas = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUSOLVER(cusolverDnCreate(&solver));
    CHECK_CUSOLVER(cusolverDnSetStream(solver, stream));
    CHECK_CUBLAS(cublasCreate(&blas));
    CHECK_CUBLAS(cublasSetStream(blas, stream));

    for (const TestCase& c : kCases) {
        run_case(c, solver, blas, stream);
    }

    CHECK_CUBLAS(cublasDestroy(blas));
    CHECK_CUSOLVER(cusolverDnDestroy(solver));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
