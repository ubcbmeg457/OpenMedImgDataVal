#!/bin/bash
#SBATCH --job-name=xray-shapley
#SBATCH --account=rrg-timsbc_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ---- Alliance Canada / Fir cluster — SLURM batch script ----
# Runs: xray-shapley-py/main.py (full data-valuation pipeline)
#
# First-time setup (run once on a login node):
#   module load gcc/12.3 python/3.11 cuda/12.2 arrow/17.0.0
#   pip install uv
#   cd ~/project/OpenMedImgDataVal && uv sync --all-packages --all-extras
#
# Submit from the repo root:
#   sbatch xray-shapley-py/job.sh

set -euo pipefail

PROJECT_DIR=~/project/OpenMedImgDataVal

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
module purge
module load gcc/12.3 python/3.11 cuda/12.2 arrow/17.0.0

# ---------------------------------------------------------------------------
# Use scratch for kagglehub cache (home quota is too small for 45GB dataset)
# ---------------------------------------------------------------------------
export XDG_CACHE_HOME=/scratch/jsiu/.cache

# ---------------------------------------------------------------------------
# Run the pipeline using the uv-managed venv
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR/xray-shapley-py"
source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Job $SLURM_JOB_ID started on $(hostname) at $(date) ==="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

python main.py

echo "=== Job finished at $(date) ==="
