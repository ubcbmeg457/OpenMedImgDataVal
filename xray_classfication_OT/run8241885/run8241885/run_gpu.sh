#!/bin/bash
#SBATCH --job-name=nih_dens
#SBATCH --account=st-rohling-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Date: $(date)"

# Activate virtual environment
source /home/daggarwa/venvs/xray_gpu/bin/activate

echo "Python: $(which python)"
python -V

# Define run directory
RUN_DIR="/scratch/st-rohling-1/2025_capstone/xray_runs/run_001"

# Redirect all caches to scratch (compute nodes have restricted home)
export TORCH_HOME="${RUN_DIR}/.torch_cache"
export XDG_CACHE_HOME="${RUN_DIR}/.cache"
export HF_HOME="${RUN_DIR}/.hf_cache"
export MPLCONFIGDIR="${RUN_DIR}/.mpl_cache"

mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR"

# Confirm GPU
nvidia-smi

# Paths
DATA_ROOT="/arc/project/st-rohling-1/2025_capstone/archive"
OUT_DIR="${RUN_DIR}/out"
PRETRAINED_PATH="${RUN_DIR}/.torch_cache/hub/checkpoints/densenet121-a639ec97.pth"

mkdir -p "$OUT_DIR"

python "${RUN_DIR}/xray_densenet121_ot.py" \
  --data_root "$DATA_ROOT" \
  --out_dir "$OUT_DIR" \
  --epochs 3 \
  --batch_size 32 \
  --num_workers 8 \
  --pretrained_path "$PRETRAINED_PATH"

echo "Finished at: $(date)"
