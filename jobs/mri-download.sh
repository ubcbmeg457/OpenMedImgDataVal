#!/bin/bash
#SBATCH --job-name=mri-download
#SBATCH --account=def-timsbc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_DIR=~/scratch/OpenMedImgDataVal

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
# Run
# ---------------------------------------------------------------------------
source "$PROJECT_DIR/.venv/bin/activate"

# Delete incomplete data
rm -rf src/data/brats2023

# Download
PYTHONPATH=src python -c "from mri_seg.data import download_dataset; path = download_dataset(); print(f'Resolved path: {path}')"

# Print the full directory structure
echo ""
echo "=== Directory structure ==="
find src/data -type d

echo ""
echo "=== File counts per directory ==="
find src/data -type d | while read d; do
  count=$(find "$d" -maxdepth 1 -type f | wc -l)
  if [ "$count" -gt 0 ]; then
    echo "$d: $count files"
  fi
done
