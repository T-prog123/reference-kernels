#!/bin/bash
#SBATCH --job-name=diag_nsys
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:02:00
#SBATCH --output=logs/diag_nsys_h100_%j.out
#SBATCH --error=logs/diag_nsys_h100_%j.err

set -x

hostname
which nsys || true
nsys --version || true
readlink -f "$(which nsys)" || true
/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nsys --version || true
/opt/nvidia/nsight-systems/2024.4.2/target-linux-x64/nsys --version || true

echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"

ls -l /usr/local/cuda-12.6/bin/nsys || true
ls -ld /usr/local/cuda-12.6/nsight-systems* /opt/nvidia/nsight-systems* 2>/dev/null || true
