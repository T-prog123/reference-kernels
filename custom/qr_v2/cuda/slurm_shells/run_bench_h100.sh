#!/bin/bash
#SBATCH --job-name=qr_cuda_bench
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

CUDA_DIR="${CUDA_DIR:-/home/haicu/titouan.breton/holder/reference-kernels/custom/qr_v2/cuda}"
cd "${CUDA_DIR}"

REQUESTED_KERNEL="${1:-copy_stub}"
MODE="${2:-quick}"
CASE_FILTER="${3:-}"
KERNEL="${REQUESTED_KERNEL}"
LOG_KERNEL="${REQUESTED_KERNEL}"
if [[ "${REQUESTED_KERNEL}" == "cuBLAS" || "${REQUESTED_KERNEL}" == "cublas" ]]; then
    KERNEL="cublas_geqrf_batched"
    LOG_KERNEL="cuBLAS"
fi
MAGMA_HOME="${MAGMA_HOME:-/ictstr01/home/haicu/titouan.breton/micromamba/envs/magma-qr}"
export LD_LIBRARY_PATH="${MAGMA_HOME}/lib:${MAGMA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p logs
LOG_SUFFIX="${LOG_KERNEL}_${MODE}"
if [[ -n "${CASE_FILTER}" ]]; then
    LOG_SUFFIX="${LOG_SUFFIX}_${CASE_FILTER}"
fi
exec >"logs/bench_${LOG_SUFFIX}.out" 2>"logs/bench_${LOG_SUFFIX}.err"

echo "kernel: ${KERNEL}"
if [[ "${REQUESTED_KERNEL}" != "${KERNEL}" ]]; then
    echo "requested_kernel: ${REQUESTED_KERNEL}"
fi
echo "mode: ${MODE}"
if [[ -n "${CASE_FILTER}" ]]; then
    echo "case_filter: ${CASE_FILTER}"
fi
echo "job_id: ${SLURM_JOB_ID:-local}"
make KERNEL="${KERNEL}"
if [[ -n "${CASE_FILTER}" ]]; then
    "./build/bench_${KERNEL}" "${MODE}" "${CASE_FILTER}"
else
    "./build/bench_${KERNEL}" "${MODE}"
fi
