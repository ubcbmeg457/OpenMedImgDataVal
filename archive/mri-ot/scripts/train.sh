#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=rrg-timsbc
#SBATCH --time=0-16:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=64G
#SBATCH --output=train.out

DATA_PATH="/home/chloechr/scratch/OTMRIWORKING/sliced"
source /home/chloechr/scratch/venvs/BMEG457_scratch/bin/activate

python -u trainMRIsegBottom.py \
        --data_path ${DATA_PATH} \
        --task segmentation \
        --modality MRI \
        --DV OT \
        --out_dir otmri_results \
        --epochs 100 \
        --batch_size 32 \
        --early_stop_patience 10\
        --ot_reg 0.01 \
        --seed 42