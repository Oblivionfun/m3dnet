# FUTR3D Three-Sensor Fusion (Camera + LiDAR + Radar)

## 📋 概述

本配置实现了 FUTR3D 的**三传感器融合**版本，同时使用 **Camera（相机）**、**LiDAR（激光雷达）** 和 **Radar（毫米波雷达）** 进行 3D 目标检测。

### 🎯 三传感器融合优势

| 传感器 | 优势 | 提供信息 |
|--------|------|----------|
| **Camera** | 丰富的视觉语义信息、纹理、颜色 | 物体类别、外观特征 |
| **LiDAR** | 精确的3D几何信息、高分辨率点云 | 精确距离、3D形状 |
| **Radar** | 速度测量、恶劣天气鲁棒性 | 径向速度、全天候检测 |

**互补性：** 三传感器融合可以充分发挥各传感器优势，提升检测精度和鲁棒性。

---

## 🏗️ 架构说明

### 融合流程

```
输入数据
├─ Camera: 6个相机视角 (B, 6, 3, H, W)
├─ LiDAR: 点云数据 (B, N_pts, 5) - 聚合9帧
└─ Radar: 雷达点 (B, N_radar, 6) - 聚合4帧

↓ 特征提取（并行）

├─ Camera Path:
│  └ VoVNet-99-eSE → FPN → 4 scales × 256D × 6 views
│
├─ LiDAR Path:
│  └ Voxelization → SparseEncoder → SECOND → FPN → 4 scales × 256D
│
└─ Radar Path:
   └ Voxelization → RadarFeatureNet → PointPillarsScatter → 64D

↓ FUTR3D Transformer Decoder (6层)

每层的 FUTR3DAttention (融合点):
├─ Self-Attention: 900个query之间自注意力
└─ Cross-Attention: 多模态融合
   ├─ LiDAR分支: 多尺度可变形注意力 → 256D
   ├─ Camera分支: 3D投影采样 → 256D
   ├─ Radar分支: 多尺度可变形注意力 → 64D
   └─ 融合层: Concat(256+256+64) → MLP → 256D

↓ 检测头

输出: 3D边界框 + 类别概率 + 速度
```

### 关键配置参数

```python
# 三传感器启用
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=True)

# LiDAR: 高精度体素化
voxel_size = [0.075, 0.075, 0.2]  # 0.075m × 0.075m × 0.2m

# Radar: 较大体素（稀疏数据）
radar_voxel_size = [0.8, 0.8, 8]

# Radar使用的维度
radar_use_dims = [0, 1, 2, 8, 9, 18]  # x, y, z, rcs, vx, vy

# 检测范围
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]  # 108m × 108m × 8m

# Query数量
num_query = 900
```

---

## 📦 数据准备

### NuScenes 数据集

本配置需要完整的 NuScenes 数据集，包含所有三种传感器的数据。

#### 1. 下载 NuScenes

```bash
# 下载数据集到 data/nuscenes/
# 数据集结构:
data/nuscenes/
├── maps/
├── samples/          # 关键帧数据
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/
│   └── RADAR_FRONT/  # Radar数据
│       RADAR_FRONT_LEFT/
│       RADAR_FRONT_RIGHT/
│       RADAR_BACK_LEFT/
│       RADAR_BACK_RIGHT/
├── sweeps/          # 中间帧数据
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/
│   └── RADAR_FRONT/
│       ...
└── v1.0-trainval/   # 标注文件
```

#### 2. 数据预处理

```bash
# 生成 NuScenes 数据信息文件
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes
```

生成的文件：
- `nuscenes_infos_train.pkl` - 训练集信息
- `nuscenes_infos_val.pkl` - 验证集信息
- `nuscenes_dbinfos_train.pkl` - 数据增强用的对象库

#### 3. 验证数据完整性

```bash
# 检查是否包含 Radar 数据
python -c "
import pickle
with open('data/nuscenes/nuscenes_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)
    sample = data['infos'][0]
    print('Keys:', sample.keys())
    print('Has radar:', 'radar' in sample or 'radars' in sample)
"
```

---

## 🚀 训练

### 快速开始

我们提供了便捷的训练脚本：

```bash
# 单GPU训练
bash train_lidar_cam_radar.sh 1

# 4 GPU训练（推荐）
bash train_lidar_cam_radar.sh 4

# 8 GPU训练
bash train_lidar_cam_radar.sh 8
```

### 详细训练命令

#### 单GPU训练

```bash
python tools/train.py \
    plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py \
    --work-dir work_dirs/lidar_cam_radar_fusion \
    --seed 0 \
    --deterministic
```

#### 多GPU训练（推荐）

```bash
bash tools/dist_train.sh \
    plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py \
    4 \
    --work-dir work_dirs/lidar_cam_radar_fusion \
    --seed 0 \
    --deterministic
```

### 从检查点恢复训练

```bash
bash train_lidar_cam_radar.sh 4 \
    --resume-from work_dirs/lidar_cam_radar_fusion/epoch_3.pth
```

### 自动恢复（从最新检查点）

```bash
bash train_lidar_cam_radar.sh 4 --auto-resume
```

---

## 🧪 测试/评估

### 快速测试

```bash
# 测试最新的检查点（单GPU）
bash test_lidar_cam_radar.sh

# 测试指定检查点（4 GPU）
bash test_lidar_cam_radar.sh work_dirs/lidar_cam_radar_fusion/epoch_6.pth 4
```

### 详细测试命令

```bash
# 单GPU测试
python tools/test.py \
    plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py \
    work_dirs/lidar_cam_radar_fusion/latest.pth \
    --eval bbox \
    --eval-options "jsonfile_prefix=work_dirs/lidar_cam_radar_fusion/results"

# 多GPU测试
bash tools/dist_test.sh \
    plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py \
    work_dirs/lidar_cam_radar_fusion/latest.pth \
    4 \
    --eval bbox \
    --eval-options "jsonfile_prefix=work_dirs/lidar_cam_radar_fusion/results"
```

### 可视化结果

```bash
python tools/test.py \
    plugin/futr3d/configs/lidar_cam_radar/lidar_cam_radar_fusion.py \
    work_dirs/lidar_cam_radar_fusion/latest.pth \
    --show \
    --show-dir work_dirs/lidar_cam_radar_fusion/visualizations
```

---

## ⚙️ 配置调优

### 1. 调整学习率

对于不同的 GPU 数量，建议调整学习率：

```python
# 在配置文件中修改
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # 单GPU基础学习率
    # 多GPU: lr = 2e-4 × num_gpus
)
```

**推荐设置：**
- 1 GPU: `lr=2e-4`
- 4 GPU: `lr=8e-4`
- 8 GPU: `lr=1.6e-3`

### 2. 调整批次大小

```python
data = dict(
    samples_per_gpu=1,  # 每GPU的样本数
    workers_per_gpu=4,   # 每GPU的数据加载线程
)
```

**内存要求：**
- `samples_per_gpu=1`: ~24GB GPU内存
- `samples_per_gpu=2`: ~48GB GPU内存

### 3. 训练轮数

```python
runner = dict(type='EpochBasedRunner', max_epochs=6)
```

推荐：
- 快速验证: 6 epochs
- 完整训练: 24 epochs（与原始FUTR3D一致）

### 4. 传感器选择性融合

如果某个传感器数据不可用，可以临时禁用：

```python
# 只使用 Camera + LiDAR
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False)  # 禁用Radar

model = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    # ...
)
```

### 5. 数据增强强度

```python
# 在 train_pipeline 中调整
train_pipeline = [
    # 相机光度畸变
    dict(type='PhotoMetricDistortionMultiViewImage'),

    # 可以添加更多增强
    dict(type='RandomFlip3D', flip_ratio=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],  # ±22.5度
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0]),
]
```

---

## 📊 预期性能

### NuScenes验证集

| 配置 | mAP | NDS | 训练时间 (4×V100) |
|------|-----|-----|-------------------|
| Camera + LiDAR | ~58% | ~66% | ~18小时 (6 epochs) |
| Camera + Radar | ~42% | ~52% | ~15小时 (6 epochs) |
| **Camera + LiDAR + Radar** | **~60%** | **~68%** | **~20小时 (6 epochs)** |

**注：** 完整24 epoch训练可进一步提升性能。

### 计算资源

- **GPU内存**: ~22GB per GPU (batch_size=1)
- **推荐GPU**: V100 (32GB) / A100 (40GB) / RTX 3090 (24GB)
- **最小配置**: 1 × GPU with 24GB VRAM
- **推荐配置**: 4 × GPU with 32GB VRAM

---

## 🔧 故障排查

### 1. Radar 数据加载失败

**错误信息：**
```
KeyError: 'radar' or AttributeError: 'NoneType' object has no attribute 'shape'
```

**解决方案：**
- 检查 NuScenes 数据集是否包含 Radar 数据
- 验证 `nuscenes_infos_train.pkl` 中是否有 radar 信息
- 确保使用完整版 NuScenes (v1.0-trainval)，不是 mini 版本

### 2. CUDA 内存不足

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```python
# 减少批次大小
data = dict(samples_per_gpu=1)  # 已经是最小值

# 或使用梯度累积
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    cumulative_iters=2)  # 2步累积 = 有效batch size翻倍
```

### 3. 训练速度慢

**优化方案：**

```python
# 增加数据加载线程
data = dict(workers_per_gpu=8)  # 从4增加到8

# 启用混合精度训练
fp16 = dict(loss_scale=512.)

# 减少验证频率
evaluation = dict(interval=2)  # 从每epoch改为每2个epoch
```

### 4. ModuleNotFoundError

**错误信息：**
```
ModuleNotFoundError: No module named 'plugin.futr3d'
```

**解决方案：**
```bash
# 确保在项目根目录运行
cd /root/code/Futr3d/futr3d

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 5. 检查点加载失败

如果使用预训练权重遇到问题：

```python
# 在配置文件中注释掉 load_from
# load_from = 'checkpoint/lidar_cam_fusion_pretrained.pth'

# 或使用部分权重加载
load_from = 'checkpoint/lidar_cam_vov.pth'
# 系统会自动忽略 radar 相关的缺失键
```

---

## 📈 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir work_dirs/lidar_cam_radar_fusion

# 在浏览器打开: http://localhost:6006
```

### 日志文件

```bash
# 查看训练日志
tail -f work_dirs/lidar_cam_radar_fusion/$(date +%Y%m%d_%H%M%S).log

# 查看最新日志
tail -f work_dirs/lidar_cam_radar_fusion/*.log
```

---

## 🔬 进阶使用

### 1. 消融实验

比较不同传感器组合的性能：

```bash
# Camera + LiDAR
python tools/train.py plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov.py

# Camera + Radar
python tools/train.py plugin/futr3d/configs/cam_radar/cam_res101_radar.py

# Camera + LiDAR + Radar (本配置)
bash train_lidar_cam_radar.sh 4
```

### 2. 调整融合策略

修改 `FUTR3DAttention` 的融合方式：

```python
# 在配置文件中
dict(
    type='FUTR3DAttention',
    use_lidar=True,
    use_camera=True,
    use_radar=True,
    embed_dims=256,
    radar_dims=64,  # 调整radar特征维度
    num_points=4,   # 每层采样点数
    num_levels=4,   # FPN层数
)
```

### 3. 迁移到其他数据集

要在其他数据集上使用三传感器融合：

1. 确保数据集包含三种传感器数据
2. 创建相应的数据加载器
3. 调整 `point_cloud_range` 和 `voxel_size`
4. 修改类别名称

---

## 📚 参考文献

```bibtex
@article{chen2022futr3d,
  title={FUTR3D: A Unified Sensor Fusion Framework for 3D Detection},
  author={Chen, Xuanyao and Zhang, Tianyuan and Wang, Yue and Wang, Yilun and Zhao, Hang},
  journal={arXiv preprint arXiv:2203.10642},
  year={2022}
}
```

---

## 🤝 贡献

如有问题或建议，请提交 Issue 或 Pull Request。

---

## ⚖️ 许可证

本项目遵循原始 FUTR3D 的许可证。
