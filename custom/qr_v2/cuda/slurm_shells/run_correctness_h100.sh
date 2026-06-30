#!/bin/bash
#SBATCH --job-name=qr_cuda_correct
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

CUDA_DIR="${CUDA_DIR:-/home/haicu/titouan.breton/holder/reference-kernels/custom/qr_v2/cuda}"
cd "${CUDA_DIR}"

KERNEL="${1:-cusolver_geqrf}"
MAGMA_HOME="${MAGMA_HOME:-/ictstr01/home/haicu/titouan.breton/micromamba/envs/magma-qr}"
export LD_LIBRARY_PATH="${MAGMA_HOME}/lib:${MAGMA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p logs
exec >"logs/correctness_${KERNEL}.out" 2>"logs/correctness_${KERNEL}.err"

echo "kernel: ${KERNEL}"
echo "job_id: ${SLURM_JOB_ID:-local}"
make correctness KERNEL="${KERNEL}"
"./build/correctness_${KERNEL}"
