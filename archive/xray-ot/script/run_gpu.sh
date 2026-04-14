#!/bin/bash
#SBATCH --job-name=nih_dens_bin_shap
#SBATCH --account=st-rohling-1-gpu
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# More CPU helps data loading + augmentation.
#SBATCH --cpus-per-task=16

# More RAM helps caching/prefetching and avoids OOM in Python side.
#SBATCH --mem=32G

# Increase walltime so it can finish the full pipeline.
#SBATCH --time=36:00:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Date: $(date)"

# Activate virtual environment
source /home/daggarwa/venvs/xray_gpu/bin/activate

echo "Python: $(which python)"
python -V

# Run directory (where your script + caches live)
RUN_DIR="/scratch/st-rohling-1/2025_capstone/final_runs/Shapley/"

# Redirect caches to scratch (compute nodes often restrict $HOME writes)
export TORCH_HOME="${RUN_DIR}/.torch_cache"
export XDG_CACHE_HOME="${RUN_DIR}/.cache"
export HF_HOME="${RUN_DIR}/.hf_cache"
export MPLCONFIGDIR="${RUN_DIR}/.mpl_cache"

mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR"

# CPU threading settings (helps avoid oversubscription)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Confirm GPU visibility
nvidia-smi

# Paths
DATA_ROOT="/arc/project/st-rohling-1/2025_capstone/archive"
OUT_DIR="${RUN_DIR}/out_${SLURM_JOB_ID}"

# This path must already exist (downloaded previously); script will fail if missing
PRETRAINED_PATH="/scratch/st-rohling-1/2025_capstone/xray_runs/run_001/.torch_cache/hub/checkpoints/densenet121-a639ec97.pth"


mkdir -p "$OUT_DIR"

echo "RUN_DIR: $RUN_DIR"
echo "DATA_ROOT: $DATA_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "PRETRAINED_PATH: $PRETRAINED_PATH"
echo "TORCH_HOME: $TORCH_HOME"
echo "CPUS: $SLURM_CPUS_PER_TASK"
echo "MEM: 32G"
echo "TIME: 36:00:00"

python "${RUN_DIR}/xray_densenet121_shap.py" \
  --data_root "$DATA_ROOT" \
  --out_dir "$OUT_DIR" \
  --epochs 300 \
  --batch_size 64 \
  --num_workers 16 \
  --early_stop_patience 5 \
  --use_scheduler \
  --pretrained_path "$PRETRAINED_PATH" \
  --torch_home "$TORCH_HOME" \
  --dropout 0.4 \
  --freeze_backbone \
  --unfreeze_last_block \
  --unfreeze_epoch 100 \
  --head_lr 1e-3 \
  --finetune_lr 1e-4 \
  --weight_decay_head 0.0 \
  --weight_decay_finetune 5e-4 \
  --shapley_k 10 \
  --shapley_mstar 2000 \
  --shapley_batch_val 64

echo "Finished at: $(date)"