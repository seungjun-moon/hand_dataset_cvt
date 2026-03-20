#!/bin/bash
#SBATCH --job-name=cvt_whim
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j-%x.log
#SBATCH --error=logs/%j-%x.err
#SBATCH --comment="download and convert WHIM train videos"

cd /rlwrld3/home/seungjun/hand_dataset_cvt
source ~/.bashrc
source /rlwrld3/home/seungjun/miniconda3/etc/profile.d/conda.sh
conda activate ego_pipeline
echo "Conda environment activated."

mkdir -p logs

python scripts/convert_whim.py \
    --src ../WiLoR \
    --dst CONVERTED/whim_train \
    --mode train
