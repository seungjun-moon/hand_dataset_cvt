#!/bin/bash
#SBATCH --job-name=cvt_mtc_t
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j-%x.log
#SBATCH --error=logs/%j-%x.err
#SBATCH --comment="convert MTC training split to egodex format"

cd /rlwrld3/home/seungjun/hand_dataset_cvt
source ~/.bashrc
source /rlwrld3/home/seungjun/miniconda3/etc/profile.d/conda.sh
conda activate ego_pipeline
echo "Conda environment activated."

mkdir -p logs

python scripts/convert_mtc.py \
    --src ../mtc_dataset/mtc_video_dataset \
    --dst CONVERTED/mtc_train \
    --split training
