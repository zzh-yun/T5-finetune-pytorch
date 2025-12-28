#!/bin/bash
# Multi-GPU training script for T5 NMT
# Usage: bash train_distributed.sh [num_gpus]

# Get number of GPUs (default: all available GPUs)
NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}

echo "Starting distributed training with $NUM_GPUS GPUs..."

# Method 1: Using torchrun (recommended for PyTorch >= 1.9)
# torchrun automatically handles process spawning and environment variables
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    distributed=true

# Alternative Method 2: Using python -m torch.distributed.launch (for older PyTorch versions)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=29500 \
#     train.py \
#     distributed=true

echo "Training completed!"

