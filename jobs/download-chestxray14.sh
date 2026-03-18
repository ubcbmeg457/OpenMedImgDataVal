#!/bin/bash
#SBATCH --job-name=dl-chestxray14
#SBATCH --account=rrg-timsbc_gpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ---- Download NIH ChestXray14 from Kaggle ----
# Submit:
#   sbatch jobs/download-chestxray14.sh

set -euo pipefail

module purge
module load python/3.11

pip install --quiet kaggle

DATA_DIR=~/scratch/data/chestxray14
mkdir -p "$DATA_DIR"

echo "=== Download started at $(date) ==="
kaggle datasets download -d nih-chest-xrays/data -p "$DATA_DIR" --unzip
echo "=== Download finished at $(date) ==="

echo "Contents:"
ls -lh "$DATA_DIR"
du -sh "$DATA_DIR"
