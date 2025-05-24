#!/bin/bash

# Complete training startup script for Vast.ai
# Usage: Run this script on the remote Vast.ai instance after setup

echo "=== FedVLA Training Startup ==="

# Activate conda environment
echo "Activating fedvla environment..."
source ~/miniconda/bin/activate fedvla

# Navigate to project directory
cd ~/FedVLA_Policy

# Check if episode data exists
if [ ! -d "./mycobot_episodes" ]; then
    echo "ERROR: Episode data not found at ./mycobot_episodes"
    echo "Please run the sync_episodes.sh script from your local machine first"
    exit 1
fi

# Create checkpoints directory
mkdir -p ./checkpoints

echo "Starting training with the following configuration:"
echo "  - Data dir: ./mycobot_episodes"
echo "  - Output dir: ./checkpoints"
echo "  - Epochs: 800"
echo "  - Batch size: 128"
echo "  - Save interval: 20"
echo "  - Eval interval: 10"
echo "  - Validation split: 10%"
echo ""

# Start training
python DP/train.py \
    --data_dir ./mycobot_episodes \
    --eval_data_dir ./mycobot_episodes \
    --output_dir ./checkpoints \
    --num_epochs 800 \
    --batch_size 128 \
    --save_interval 20 \
    --eval_interval 10 \
    --val_split_ratio 0.1 