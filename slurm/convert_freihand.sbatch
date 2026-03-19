#!/bin/bash
#SBATCH --job-name=cvt_freihand
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --output=logs/%j-%x.log
#SBATCH --error=logs/%j-%x.err
#SBATCH --comment="convert freihand to egodex format"

cd /rlwrld3/home/seungjun/hand_dataset_cvt
source ~/.bashrc
source /rlwrld3/home/seungjun/miniconda3/etc/profile.d/conda.sh
conda activate ego_pipeline
echo "Conda environment activated."

mkdir -p logs

python scripts/convert_freihand.py \
    --src /rlwrld3/home/seungjun/freihand/FreiHAND_pub_v2 \
    --dst-train CONVERTED/freihand_train \
    --dst-eval CONVERTED/freihand_eval
