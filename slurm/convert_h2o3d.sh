#!/bin/bash
#SBATCH --job-name=convert_h2o3d
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/convert_h2o3d_%j.log

cd /rlwrld3/home/seungjun/hand_dataset_cvt

# Extract dataset if not already done
H2O3D_DIR="../ho3d/data/h2o3d"
if [ ! -d "${H2O3D_DIR}/train" ]; then
    echo "Extracting h2o3d_v1.zip..."
    unzip -q "${H2O3D_DIR}/h2o3d_v1.zip" -d "${H2O3D_DIR}"
    echo "Extraction complete."
fi

/rlwrld3/home/seungjun/miniconda3/envs/hawor/bin/python -u scripts/convert_h2o3d.py \
    --src "${H2O3D_DIR}" \
    --dst CONVERTED/h2o3d \
    --fps 30.0

echo "Conversion complete!"
