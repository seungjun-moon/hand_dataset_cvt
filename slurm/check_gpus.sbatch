#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=v100
#SBATCH --output=logs/gpu_check_%a.log
#SBATCH --error=logs/gpu_check_%a.err
#SBATCH --time=00:05:00
#SBATCH --comment="GPU health check"

echo "=== GPU Check: array task $SLURM_ARRAY_TASK_ID ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

nvidia-smi
echo ""
nvidia-smi --query-gpu=index,name,memory.total,memory.free,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total,temperature.gpu --format=csv
echo ""

# Quick CUDA sanity test
python3 -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
# Allocate and compute on GPU
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
c = a @ b
print(f'Matrix multiply test: OK (result sum={c.sum().item():.2f})')
# Memory stress test
x = torch.randn(5000, 5000, device='cuda')
y = x @ x.T
print(f'Memory stress test: OK (allocated {torch.cuda.memory_allocated()/1e9:.2f} GB)')
print('GPU PASSED')
" 2>&1

echo ""
echo "=== Done ==="
