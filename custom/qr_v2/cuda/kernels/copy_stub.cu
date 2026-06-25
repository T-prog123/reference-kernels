#include <cuda_runtime.h>

#include "../qr_kernel.h"

__global__ void copy_stub_kernel(const float* A, float* H, float* tau, size_t h_count,
                                 size_t tau_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = tid; idx < h_count; idx += stride) {
        H[idx] = A[idx];
    }
    for (size_t idx = tid; idx < tau_count; idx += stride) {
        tau[idx] = 0.0f;
    }
}

void qr_custom_kernel_cuda(const float* A, float* H, float* tau, int batch, int n,
                           cudaStream_t stream) {
    size_t h_count = static_cast<size_t>(batch) * n * n;
    size_t tau_count = static_cast<size_t>(batch) * n;
    int threads = 256;
    int blocks = static_cast<int>((h_count + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    copy_stub_kernel<<<blocks, threads, 0, stream>>>(A, H, tau, h_count, tau_count);
}
