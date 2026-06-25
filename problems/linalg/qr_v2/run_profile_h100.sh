#!/bin/bash
#SBATCH --job-name=qr_profile
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/profile_qr_%j.out
#SBATCH --error=logs/profile_qr_%j.err

cd /home/haicu/titouan.breton/reference-kernels/problems/linalg/qr_v2

mkdir -p logs

/home/haicu/titouan.breton/venv/qr_cuda_py312/bin/python3 profile_qr.py --only-n 512 --warmup 2 --repeat 1

echo "profile_qr.py exit code: $?"
