#!/bin/bash
#SBATCH --job-name=backup_dexycb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --output=logs/%j-%x.log
#SBATCH --error=logs/%j-%x.err
#SBATCH --comment="compress dex_ycb backup"

cd /rlwrld3/home/seungjun/hand_dataset_cvt

echo "Starting compression of RAW/dex_ycb ($(date))"
tar cf - RAW/dex_ycb | pigz -p 8 > dex_ycb_backup.tar.gz
echo "Done ($(date)). Output: $(ls -lh dex_ycb_backup.tar.gz)"
