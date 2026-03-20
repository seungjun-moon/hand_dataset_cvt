#!/bin/bash
#SBATCH --job-name=cvt_ih26m
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/convert_interhand26m_%j.out
#SBATCH --error=logs/convert_interhand26m_%j.err

SPLIT=${1:-train}

cd /rlwrld3/home/seungjun/hand_dataset_cvt

mkdir -p logs

python scripts/convert_interhand26m.py \
    --src ../InterWild/data/InterHand26M \
    --dst "CONVERTED/interhand26m_${SPLIT}" \
    --split "${SPLIT}" \
    --chunk-size 28
