#!/bin/bash
#SBATCH --job-name=eval_h100
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/logs.out
#SBATCH --error=logs/logs.err

mkdir -p logs

exec 3> logs/benchmark_popcorn.txt
export POPCORN_FD=3
export POPCORN_SEED=42

/home/haicu/titouan.breton/venv/qr_cuda_py312/bin/python3 eval.py benchmark benchmark.txt

echo "eval.py exit code: $?"

exec 3>&-