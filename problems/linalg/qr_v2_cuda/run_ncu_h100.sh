#!/bin/bash
#SBATCH --job-name=qr_cuda_ncu
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

KERNEL="${1:-blocked_v8}"
PROFILE="${2:-blocked}"
JOB_ID="${SLURM_JOB_ID:-local}"

mkdir -p logs
LOG_BASENAME="ncu_${KERNEL}_${PROFILE}_${JOB_ID}"
exec >"logs/${LOG_BASENAME}.out" 2>"logs/${LOG_BASENAME}.err"

echo "kernel: ${KERNEL}"
echo "profile: ${PROFILE}"
echo "job_id: ${JOB_ID}"

make profile KERNEL="${KERNEL}" PROFILE="${PROFILE}"

set +e
ncu --profile-from-start off -o "logs/${LOG_BASENAME}" "./build/profile_${PROFILE}"
NCU_STATUS=$?
set -e

echo "ncu_exit_code: ${NCU_STATUS}"
exit "${NCU_STATUS}"
