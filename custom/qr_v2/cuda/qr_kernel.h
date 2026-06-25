#pragma once

#include <cuda_runtime.h>

void qr_custom_kernel_cuda(
    const float* A,
    float* H,
    float* tau,
    int batch,
    int n,
    cudaStream_t stream);
