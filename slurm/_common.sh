#!/bin/bash
# Shared SLURM job preamble for HHD-Ethiopic.
# All SLURM scripts in this project should:
#   source slurm/_common.sh
# after their #SBATCH directives.
#
# Behaviour:
#   * set -euo pipefail and trap unhandled errors
#   * activate the project conda env (default: amharic_htr, aliased "hhd-env")
#   * print a small banner with host, GPU, env, git hash
#   * make slurm_logs/ if missing

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

echo "==========================================================="
echo "Job:     ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-no-id})"
echo "Host:    $(hostname)"
echo "Started: $(date -Iseconds)"
echo "PWD:     $PWD"
echo "User:    $USER"
echo "==========================================================="

# Conda env activation; works around the cluster's PYTHONHOME leakage
unset PYTHONHOME PYTHONPATH
module purge >/dev/null 2>&1 || true
module load conda/25.7.0 >/dev/null 2>&1 || module load conda >/dev/null 2>&1
CONDA_BASE="$(conda info --base 2>/dev/null)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-amharic_htr}"

echo "Conda:   ${CONDA_DEFAULT_ENV} @ ${CONDA_PREFIX}"
echo "Python:  $(python --version)  ($(which python))"
echo "Torch:   $(python -c 'import torch; print(torch.__version__, "cuda=", torch.version.cuda)')"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null | head -10 || true
fi
echo "==========================================================="

export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/HHD-Ethiopic-data}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
