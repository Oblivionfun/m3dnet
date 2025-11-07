#!/bin/bash

# Testing script for Camera + LiDAR + Radar three-sensor fusion
# FUTR3D Three-Sensor Fusion Testing Script

# Configuration
CONFIG="plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py"
CHECKPOINT=${1:-"work_dirs/lidar_cam_radar_fusion/latest.pth"}
GPUS=${2:-1}  # Default to 1 GPU if not specified
PORT=${PORT:-29500}

echo "=========================================="
echo "FUTR3D Three-Sensor Fusion Testing"
echo "=========================================="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "GPUs: $GPUS"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found at $CONFIG"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT"
    echo "Please provide a valid checkpoint path as the first argument:"
    echo "  bash test_lidar_cam_radar.sh <checkpoint_path> [num_gpus]"
    exit 1
fi

# Single GPU testing
if [ $GPUS -eq 1 ]; then
    echo "Starting single GPU testing..."
    python tools/test.py \
        $CONFIG \
        $CHECKPOINT \
        --eval bbox \
        --eval-options "jsonfile_prefix=work_dirs/lidar_cam_radar_fusion/results"

# Multi-GPU testing
else
    echo "Starting multi-GPU testing with $GPUS GPUs..."
    bash tools/dist_test.sh \
        $CONFIG \
        $CHECKPOINT \
        $GPUS \
        --eval bbox \
        --eval-options "jsonfile_prefix=work_dirs/lidar_cam_radar_fusion/results"
fi

echo "=========================================="
echo "Testing completed!"
echo "Results saved to: work_dirs/lidar_cam_radar_fusion/results"
echo "=========================================="
