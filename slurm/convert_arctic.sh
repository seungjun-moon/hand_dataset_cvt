#!/bin/bash
#SBATCH --job-name=convert_arctic
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/convert_arctic_%j.log

cd /rlwrld3/home/seungjun/hand_dataset_cvt

/rlwrld3/home/seungjun/miniconda3/envs/hawor/bin/python scripts/convert_arctic.py \
    --src ../arctic/downloads/data \
    --dst CONVERTED/arctic \
    --mano-model-dir ../arctic/unpack/body_models/mano \
    --fps 30.0

echo "Conversion complete!"
