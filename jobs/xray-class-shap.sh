#!/bin/bash
#SBATCH --job-name=xray-class-shap
#SBATCH --account=rrg-timsbc_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ---- Alliance Canada / Fir cluster — SLURM batch script ----
# Runs: X-ray classification with KNN-Shapley data valuation
#
# Submit from the repo root (~/scratch/OpenMedImgDataVal):
#   sbatch jobs/xray-class-shap.sh

set -euo pipefail

PROJECT_DIR=~/scratch/OpenMedImgDataVal
DATA_DIR=~/scratch/data/chestxray14

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
module purge
module load gcc/12.3 python/3.11 cuda/12.2 arrow/17.0.0

# ---------------------------------------------------------------------------
# Install dependencies (idempotent — skips if already up to date)
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
pip install --quiet uv
uv sync --all-packages --all-extras

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Job $SLURM_JOB_ID started on $(hostname) at $(date) ==="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

python src/main.py --modality xray --task class --dv shap --input "$DATA_DIR"

echo "=== Job finished at $(date) ==="
