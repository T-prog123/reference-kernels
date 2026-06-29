#!/bin/bash
#SBATCH --job-name=qr_cuda_nsys
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

CUDA_DIR="${CUDA_DIR:-/home/haicu/titouan.breton/reference-kernels/custom/qr_v2/cuda}"
cd "${CUDA_DIR}"

KERNEL="${1:-blocked_v8}"
ARCH_TAG="${2:-H}"
JOB_ID="${SLURM_JOB_ID:-local}"

mkdir -p logs nsys_rep
exec >"logs/nsys_${KERNEL}_blocked_${ARCH_TAG}_${JOB_ID}.out" 2>"logs/nsys_${KERNEL}_blocked_${ARCH_TAG}_${JOB_ID}.err"

find_nsys() {
    for path in \
        /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
        /opt/nvidia/nsight-systems/*/bin/nsys \
        /usr/local/cuda-*/bin/nsys; do
        if [[ -x "${path}" ]] && "${path}" --version >/dev/null 2>&1; then
            echo "${path}"
            return 0
        fi
    done
    return 1
}

if [[ -z "${NSYS:-}" ]]; then
    if ! NSYS="$(find_nsys)"; then
        echo "error: could not find nsys; set NSYS=/path/to/nsys or check Nsight Systems install" >&2
        exit 1
    fi
fi

echo "kernel: ${KERNEL}"
echo "arch_tag: ${ARCH_TAG}"
echo "job_id: ${JOB_ID}"
echo "nsys: ${NSYS}"
"${NSYS}" --version

run_case() {
    local case_n="$1"
    local binary="profile_blocked_${case_n}"
    local report="nsys_rep/nsys_${KERNEL}_blocked_${case_n}_${ARCH_TAG}"

    echo "case: ${case_n}"
    make profile KERNEL="${KERNEL}" PROFILE=blocked \
        PROFILE_FLAGS="-DPROFILE_BLOCKED_CASE=${case_n}" \
        PROFILE_OUT="${binary}"

    "${NSYS}" profile \
        --capture-range=cudaProfilerApi \
        --force-overwrite=true \
        -o "${report}" \
        "./build/${binary}"
}

run_case 512
run_case 4096
