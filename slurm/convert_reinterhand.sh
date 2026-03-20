#!/bin/bash
#SBATCH --job-name=cvt_reinterhand
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/convert_reinterhand_%j.out
#SBATCH --error=logs/convert_reinterhand_%j.err

cd /rlwrld3/home/seungjun/hand_dataset_cvt

mkdir -p logs

python scripts/convert_reinterhand.py \
    --src ../InterWild/tool/ReInterHand/download \
    --dst CONVERTED/reinterhand \
    --chunk-size 28

echo "Conversion complete!"
