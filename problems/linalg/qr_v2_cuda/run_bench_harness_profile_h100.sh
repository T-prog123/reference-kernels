#!/bin/bash
#SBATCH --job-name=qr_cuda_harness_profile
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

REQUESTED_KERNEL="${1:-copy_stub}"
MODE="${2:-aggregate}"
CASE_FILTER="${3:-}"
KERNEL="${REQUESTED_KERNEL}"
LOG_KERNEL="${REQUESTED_KERNEL}"
if [[ "${REQUESTED_KERNEL}" == "cuBLAS" || "${REQUESTED_KERNEL}" == "cublas" ]]; then
    KERNEL="cublas_geqrf_batched_profile"
    LOG_KERNEL="cuBLAS"
fi
MAGMA_HOME="${MAGMA_HOME:-/ictstr01/home/haicu/titouan.breton/micromamba/envs/magma-qr}"
export LD_LIBRARY_PATH="${MAGMA_HOME}/lib:${MAGMA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p logs
LOG_BASENAME="profile_${LOG_KERNEL}"
if [[ "${KERNEL}" == "blocked_v4_profile" ]]; then
    LOG_BASENAME="profile_block_v4"
elif [[ "${KERNEL}" == "blocked_v7_profile" ]]; then
    LOG_BASENAME="profile_block_v7"
elif [[ "${KERNEL}" == "blocked_v8_profile" ]]; then
    LOG_BASENAME="profile_block_v8"
fi
LOG_BASENAME="${LOG_BASENAME}_${MODE}"
if [[ -n "${CASE_FILTER}" ]]; then
    LOG_BASENAME="${LOG_BASENAME}_${CASE_FILTER}"
fi
exec >"logs/${LOG_BASENAME}.out" 2>"logs/${LOG_BASENAME}.err"

echo "kernel: ${KERNEL}"
if [[ "${REQUESTED_KERNEL}" != "${KERNEL}" ]]; then
    echo "requested_kernel: ${REQUESTED_KERNEL}"
fi
echo "mode: ${MODE}"
if [[ -n "${CASE_FILTER}" ]]; then
    echo "case_filter: ${CASE_FILTER}"
fi
echo "job_id: ${SLURM_JOB_ID:-local}"
make -f Makefile.harness_profile KERNEL="${KERNEL}"
if [[ -n "${CASE_FILTER}" ]]; then
    "./build/bench_harness_profile_${KERNEL}" "${MODE}" "${CASE_FILTER}"
else
    "./build/bench_harness_profile_${KERNEL}" "${MODE}"
fi
