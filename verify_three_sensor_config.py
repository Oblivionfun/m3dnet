#!/usr/bin/env python3
"""
Verification script for FUTR3D three-sensor fusion configuration.
This script checks if all components are correctly configured.
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}")
        print(f"   Path: {filepath}")
        print(f"   Size: {size:,} bytes")
        return True
    else:
        print(f"❌ {description}")
        print(f"   Path: {filepath} (NOT FOUND)")
        return False

def check_config_content():
    """Check if configuration file has correct settings."""
    config_path = "plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py"

    if not os.path.exists(config_path):
        return False

    with open(config_path, 'r') as f:
        content = f.read()

    checks = {
        "use_lidar=True": "LiDAR enabled",
        "use_camera=True": "Camera enabled",
        "use_radar=True": "Radar enabled",
        "radar_voxel_layer": "Radar voxel layer configured",
        "radar_voxel_encoder": "Radar encoder configured",
        "LoadRadarPointsMultiSweeps": "Radar data loading configured",
        "'radar'": "Radar in Collect3D keys",
    }

    print("\n🔍 Configuration Content Checks:")
    all_passed = True
    for key, desc in checks.items():
        if key in content:
            print(f"   ✅ {desc}")
        else:
            print(f"   ❌ {desc} (NOT FOUND)")
            all_passed = False

    return all_passed

def main():
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  FUTR3D Three-Sensor Fusion Configuration Verification".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")

    # Change to project root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"\n📁 Working Directory: {os.getcwd()}")

    all_checks_passed = True

    # Check 1: Configuration files
    print_header("1. Configuration Files")
    files_to_check = [
        ("plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py",
         "Three-sensor fusion config"),
        ("plugin/futr3d/configs/lidar_cam_radar/README.md",
         "Configuration documentation"),
        ("plugin/futr3d/configs/lidar_cam_radar/ARCHITECTURE.md",
         "Architecture documentation"),
    ]

    for filepath, desc in files_to_check:
        if not check_file(filepath, desc):
            all_checks_passed = False

    # Check 2: Training and testing scripts
    print_header("2. Training and Testing Scripts")
    scripts_to_check = [
        ("train_lidar_cam_radar.sh", "Training script"),
        ("test_lidar_cam_radar.sh", "Testing script"),
        ("THREE_SENSOR_FUSION_QUICKSTART.md", "Quick start guide"),
    ]

    for filepath, desc in scripts_to_check:
        if not check_file(filepath, desc):
            all_checks_passed = False
        else:
            # Check if executable
            if filepath.endswith('.sh'):
                if os.access(filepath, os.X_OK):
                    print(f"   ✅ Script is executable")
                else:
                    print(f"   ⚠️  Script is not executable (run: chmod +x {filepath})")

    # Check 3: Configuration content
    print_header("3. Configuration Content Verification")
    if not check_config_content():
        all_checks_passed = False

    # Check 4: Core model files
    print_header("4. Core Model Files (Should Already Exist)")
    core_files = [
        ("plugin/futr3d/models/detectors/futr3d.py",
         "FUTR3D detector"),
        ("plugin/futr3d/models/utils/futr3d_attention.py",
         "FUTR3D attention module"),
        ("plugin/futr3d/models/utils/futr3d_transformer.py",
         "FUTR3D transformer"),
    ]

    for filepath, desc in core_files:
        if not check_file(filepath, desc):
            print(f"   ⚠️  WARNING: Core file missing!")
            all_checks_passed = False

    # Check 5: Python environment
    print_header("5. Python Environment")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch not installed")
        all_checks_passed = False

    try:
        import mmcv
        print(f"✅ MMCV: {mmcv.__version__}")
    except ImportError:
        print("❌ MMCV not installed")
        all_checks_passed = False

    try:
        import mmdet
        print(f"✅ MMDetection: {mmdet.__version__}")
    except ImportError:
        print("❌ MMDetection not installed")
        all_checks_passed = False

    try:
        import mmdet3d
        print(f"✅ MMDetection3D: {mmdet3d.__version__}")
    except ImportError:
        print("❌ MMDetection3D not installed")
        all_checks_passed = False

    # Final summary
    print_header("Summary")
    if all_checks_passed:
        print("""
✅ All checks passed!

🚀 Quick Start:

1. Prepare NuScenes dataset:
   python tools/create_data.py nuscenes \\
       --root-path ./data/nuscenes \\
       --out-dir ./data/nuscenes

2. Start training:
   bash train_lidar_cam_radar.sh 4

3. Test model:
   bash test_lidar_cam_radar.sh

📖 For more details, see:
   - THREE_SENSOR_FUSION_QUICKSTART.md
   - plugin/futr3d/configs/lidar_cam_radar/README.md
        """)
    else:
        print("""
⚠️  Some checks failed!

Please review the errors above and ensure:
1. All configuration files are present
2. Scripts have correct permissions
3. Required Python packages are installed

For help, see: plugin/futr3d/configs/lidar_cam_radar/README.md
        """)
        sys.exit(1)

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
