#!/bin/bash
#SBATCH --job-name=depth_dexycb
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=v100
#SBATCH --output=logs/%j-%x-%a.log
#SBATCH --error=logs/%j-%x-%a.err
#SBATCH --comment="estimate depth for converted dex_ycb"

cd /rlwrld3/home/seungjun/hand_dataset_cvt
source ~/.bashrc
source /rlwrld3/home/seungjun/miniconda3/etc/profile.d/conda.sh
conda activate ego_pipeline
echo "Conda environment activated. GPU $SLURM_ARRAY_TASK_ID"

# All clusters in sorted order
ALL_CLUSTERS=(
    002_master_chef_can
    003_cracker_box
    004_sugar_box
    005_tomato_soup_can
    006_mustard_bottle
    007_tuna_fish_can
    008_pudding_box
    009_gelatin_box
    010_potted_meat_can
    011_banana
    019_pitcher_base
    021_bleach_cleanser
    024_bowl
    025_mug
    035_power_drill
    036_wood_block
    037_scissors
    040_large_marker
    052_extra_large_clamp
    061_foam_brick
)

# Round-robin assignment: pick clusters for this job
N_GPUS=8
MY_CLUSTERS=()
for i in "${!ALL_CLUSTERS[@]}"; do
    if (( i % N_GPUS == SLURM_ARRAY_TASK_ID )); then
        MY_CLUSTERS+=("${ALL_CLUSTERS[$i]}")
    fi
done

echo "Job $SLURM_ARRAY_TASK_ID processing clusters: ${MY_CLUSTERS[*]}"

for cluster in "${MY_CLUSTERS[@]}"; do
    cluster_dir="CONVERTED/dex_ycb/$cluster"
    if [ ! -d "$cluster_dir" ]; then
        echo "Skipping $cluster: directory not found"
        continue
    fi
    echo "Processing cluster: $cluster"
    python scripts/estimate_depth.py \
        --src "$cluster_dir" \
        --gpu 0
done
