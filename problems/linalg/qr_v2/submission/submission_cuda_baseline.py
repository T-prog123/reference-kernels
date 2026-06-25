import torch
from task import input_t, output_t


_qr_cusolver_ext = None


def _load_qr_cusolver_ext():
    global _qr_cusolver_ext
    if _qr_cusolver_ext is not None:
        return _qr_cusolver_ext

    from torch.utils.cpp_extension import load_inline

    cpp_src = r"""
#include <torch/extension.h>

torch::Tensor geqrf_inplace(torch::Tensor A_colmajor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("geqrf_inplace", &geqrf_inplace, "minimal cuSOLVER geqrf");
}
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <cusolverDn.h>

#include <vector>

#define CHECK_CUSOLVER(expr)                                                    \
    do {                                                                        \
        cusolverStatus_t status = (expr);                                       \
        TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS,                          \
                    "cuSOLVER error: ", static_cast<int>(status));             \
    } while (0)

torch::Tensor geqrf_inplace(torch::Tensor A_colmajor) {
    TORCH_CHECK(A_colmajor.is_cuda(), "A must be CUDA");
    TORCH_CHECK(A_colmajor.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(A_colmajor.dim() == 3, "A must be (batch, n, n)");
    TORCH_CHECK(A_colmajor.size(1) == A_colmajor.size(2), "A must be square");
    TORCH_CHECK(A_colmajor.is_contiguous(), "A must be contiguous");

    int batch = static_cast<int>(A_colmajor.size(0));
    int n = static_cast<int>(A_colmajor.size(1));

    auto tau = torch::empty({batch, n}, A_colmajor.options());
    auto info = torch::empty({1}, torch::dtype(torch::kInt32).device(A_colmajor.device()));

    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(
        handle, n, n, A_colmajor.data_ptr<float>(), n, &lwork));
    auto work = torch::empty({lwork}, A_colmajor.options());

    for (int b = 0; b < batch; ++b) {
        float* A_b = A_colmajor.data_ptr<float>() + static_cast<size_t>(b) * n * n;
        float* tau_b = tau.data_ptr<float>() + static_cast<size_t>(b) * n;
        CHECK_CUSOLVER(cusolverDnSgeqrf(
            handle, n, n, A_b, n, tau_b, work.data_ptr<float>(), lwork,
            info.data_ptr<int>()));
    }

    CHECK_CUSOLVER(cusolverDnDestroy(handle));
    return tau;
}
"""

    _qr_cusolver_ext = load_inline(
        name="qr_v2_cusolver_min_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcusolver"],
        verbose=False,
    )
    return _qr_cusolver_ext


def custom_kernel(data: input_t) -> output_t:
    # Row-major PyTorch storage for data.T is column-major storage for data.
    h_colmajor = data.transpose(-2, -1).contiguous()
    tau = _load_qr_cusolver_ext().geqrf_inplace(h_colmajor)
    h = h_colmajor.transpose(-2, -1).contiguous()
    return h, tau
