#!/bin/bash

# Training script for Camera + LiDAR + Radar three-sensor fusion
# FUTR3D Three-Sensor Fusion Training Script

# Configuration
CONFIG="plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py"
WORK_DIR="work_dirs/lidar_cam_radar_fusion"
GPUS=${1:-1}  # Default to 1 GPU if not specified
PORT=${PORT:-29500}

echo "=========================================="
echo "FUTR3D Three-Sensor Fusion Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Work Dir: $WORK_DIR"
echo "GPUs: $GPUS"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found at $CONFIG"
    exit 1
fi

# Create work directory if it doesn't exist
mkdir -p "$WORK_DIR"

# Single GPU training
if [ $GPUS -eq 1 ]; then
    echo "Starting single GPU training..."
    python tools/train.py \
        $CONFIG \
        --work-dir $WORK_DIR \
        --seed 0 \
        --deterministic

# Multi-GPU training
else
    echo "Starting multi-GPU training with $GPUS GPUs..."
    bash tools/dist_train.sh \
        $CONFIG \
        $GPUS \
        --work-dir $WORK_DIR \
        --seed 0 \
        --deterministic
fi

echo "=========================================="
echo "Training completed!"
echo "Results saved to: $WORK_DIR"
echo "=========================================="
