import os

import torch
from task import input_t, output_t


MAGMA_HOME = os.environ.get(
    "MAGMA_HOME",
    "/ictstr01/home/haicu/titouan.breton/micromamba/envs/magma-qr",
)

_qr_magma_ext = None


def _load_qr_magma_ext():
    global _qr_magma_ext
    if _qr_magma_ext is not None:
        return _qr_magma_ext

    from torch.utils.cpp_extension import load_inline

    cpp_src = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> magma_qr(torch::Tensor A);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("magma_qr", &magma_qr, "MAGMA batched geqrf");
}
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

#include <magma.h>
#include <magma_sbatched.h>

#define CHECK_CUDA(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        TORCH_CHECK(_err == cudaSuccess, "CUDA error: ",                       \
                    cudaGetErrorString(_err));                                  \
    } while (0)

#define CHECK_MAGMA(expr)                                                       \
    do {                                                                        \
        magma_int_t _err = (expr);                                              \
        TORCH_CHECK(_err == MAGMA_SUCCESS, "MAGMA error: ",                    \
                    static_cast<long long>(_err));                              \
    } while (0)

__global__ void row_to_col_major_kernel(const float* A, float* C, int batch, int n,
                                        int ldda) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int j = static_cast<int>(idx % n);
        int i = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        size_t row_major = (static_cast<size_t>(b) * n + i) * n + j;
        size_t col_major = static_cast<size_t>(b) * ldda * n + i +
                           static_cast<size_t>(j) * ldda;
        C[col_major] = A[row_major];
    }
}

__global__ void col_to_row_major_kernel(const float* C, float* H, int batch, int n,
                                        int ldda) {
    size_t total = static_cast<size_t>(batch) * n * n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int j = static_cast<int>(idx % n);
        int i = static_cast<int>((idx / n) % n);
        int b = static_cast<int>(idx / (static_cast<size_t>(n) * n));
        size_t row_major = (static_cast<size_t>(b) * n + i) * n + j;
        size_t col_major = static_cast<size_t>(b) * ldda * n + i +
                           static_cast<size_t>(j) * ldda;
        H[row_major] = C[col_major];
    }
}

__global__ void make_pointer_arrays_kernel(float* C, float* tau, float** Aarray,
                                           float** TauArray, int batch, int n,
                                           int ldda) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }

    size_t matrix_elems = static_cast<size_t>(ldda) * n;
    Aarray[b] = C + static_cast<size_t>(b) * matrix_elems;
    TauArray[b] = tau + static_cast<size_t>(b) * n;
}

void ensure_magma_initialized() {
    static bool initialized = false;
    if (!initialized) {
        CHECK_MAGMA(magma_init());
        std::atexit([]() { magma_finalize(); });
        initialized = true;
    }
}

std::vector<torch::Tensor> magma_qr(torch::Tensor A) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(A.dim() == 3, "A must have shape (batch, n, n)");
    TORCH_CHECK(A.size(1) == A.size(2), "A must be square");

    c10::cuda::CUDAGuard device_guard(A.device());

    auto A_contig = A.contiguous();
    int batch = static_cast<int>(A_contig.size(0));
    int n = static_cast<int>(A_contig.size(1));
    magma_int_t ldda = magma_roundup(n, 32);

    auto H = torch::empty_like(A_contig);
    auto tau = torch::empty({batch, n}, A_contig.options());

    size_t matrix_elems = static_cast<size_t>(ldda) * n;
    size_t total_elems = static_cast<size_t>(batch) * matrix_elems;
    size_t dense_elems = static_cast<size_t>(batch) * n * n;

    float* C = nullptr;
    float** Aarray = nullptr;
    float** TauArray = nullptr;
    magma_int_t* info = nullptr;

    CHECK_CUDA(cudaMalloc(&C, total_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&Aarray, static_cast<size_t>(batch) * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&TauArray, static_cast<size_t>(batch) * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&info, static_cast<size_t>(batch) * sizeof(magma_int_t)));

    int threads = 256;
    int elem_blocks = static_cast<int>((dense_elems + threads - 1) / threads);
    elem_blocks = elem_blocks > 4096 ? 4096 : elem_blocks;
    int batch_blocks = (batch + threads - 1) / threads;

    CHECK_CUDA(cudaMemset(C, 0, total_elems * sizeof(float)));
    row_to_col_major_kernel<<<elem_blocks, threads>>>(
        A_contig.data_ptr<float>(), C, batch, n, ldda);
    CHECK_CUDA(cudaGetLastError());
    make_pointer_arrays_kernel<<<batch_blocks, threads>>>(
        C, tau.data_ptr<float>(), Aarray, TauArray, batch, n, ldda);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    magma_queue_t queue = nullptr;
    int device = 0;

    CHECK_CUDA(cudaGetDevice(&device));
    ensure_magma_initialized();
    magma_queue_create(device, &queue);

    magma_int_t status = magma_sgeqrf_batched(
        n, n, Aarray, ldda, TauArray, info, batch, queue);
    TORCH_CHECK(status == MAGMA_SUCCESS, "MAGMA geqrf_batched failed: ",
                static_cast<long long>(status));
    magma_queue_sync(queue);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<magma_int_t> h_info(static_cast<size_t>(batch));
    CHECK_CUDA(cudaMemcpy(h_info.data(), info,
                          static_cast<size_t>(batch) * sizeof(magma_int_t),
                          cudaMemcpyDeviceToHost));
    for (int b = 0; b < batch; ++b) {
        TORCH_CHECK(h_info[static_cast<size_t>(b)] == 0,
                    "MAGMA geqrf_batched info[", b, "]=",
                    static_cast<long long>(h_info[static_cast<size_t>(b)]),
                    " for batch=", batch, " n=", n, " ldda=",
                    static_cast<long long>(ldda));
    }

    col_to_row_major_kernel<<<elem_blocks, threads>>>(
        C, H.data_ptr<float>(), batch, n, ldda);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    magma_queue_destroy(queue);

    CHECK_CUDA(cudaFree(info));
    CHECK_CUDA(cudaFree(TauArray));
    CHECK_CUDA(cudaFree(Aarray));
    CHECK_CUDA(cudaFree(C));

    return {H, tau};
}
"""

    lib_dir = os.path.join(MAGMA_HOME, "lib")
    target_lib_dir = os.path.join(MAGMA_HOME, "targets", "x86_64-linux", "lib")

    _qr_magma_ext = load_inline(
        name="qr_v2_magma_geqrf_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_include_paths=[os.path.join(MAGMA_HOME, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=[
            f"-L{lib_dir}",
            f"-L{target_lib_dir}",
            f"-Wl,-rpath,{lib_dir}",
            f"-Wl,-rpath,{target_lib_dir}",
            "-lmagma",
            "-lcusolver",
            "-lcublas",
            "-lcusparse",
        ],
        verbose=False,
    )
    return _qr_magma_ext


def custom_kernel(data: input_t) -> output_t:
    h, tau = _load_qr_magma_ext().magma_qr(data)
    return h, tau


# def custom_kernel(data: input_t) -> output_t:
#     return torch.geqrf(data)
