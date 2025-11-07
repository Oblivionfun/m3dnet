_base_ = [
    '../../../../configs/_base_/datasets/nus-3d.py',
    '../../../../configs/_base_/default_runtime.py'
]

plugin = 'plugin/futr3d'

# Class names for nuScenes 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Three-sensor fusion configuration
input_modality = dict(
    use_lidar=True,      # Enable LiDAR
    use_camera=True,     # Enable Camera
    use_radar=True,      # Enable Radar
    use_map=False,
    use_external=False)

# Point cloud and voxel settings
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]

# Radar-specific settings
radar_voxel_size = [0.8, 0.8, 8]
# Radar dimensions: x, y, z, rcs, vx_comp, vy_comp (indices: 0,1,2,8,9,18 from 18-dim radar data)
radar_use_dims = [0, 1, 2, 8, 9, 18]

# Image normalization config
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

# Model configuration with three sensors
model = dict(
    type='FUTR3D',
    use_lidar=True,      # Enable LiDAR
    use_camera=True,     # Enable Camera
    use_radar=True,      # Enable Radar
    use_grid_mask=True,
    freeze_backbone=True,

    # ==================== Camera Backbone and Neck ====================
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=['stage2', 'stage3', 'stage4', 'stage5']),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),

    # ==================== LiDAR Voxelization and Encoding ====================
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=False),
        in_channels=[128, 256],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        relu_before_extra_convs=True),

    # ==================== Radar Voxelization and Encoding ====================
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=radar_voxel_size,
        max_voxels=(90000, 120000),
        point_cloud_range=point_cloud_range),
    radar_voxel_encoder=dict(
        type='RadarFeatureNet',
        in_channels=6,  # x, y, z, rcs, vx, vy
        feat_channels=[32, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=radar_voxel_size,
        norm_cfg=dict(
            type='BN1d',
            eps=1.0e-3,
            momentum=0.01)),
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[128, 128]),

    # ==================== Detection Head ====================
    pts_bbox_head=dict(
        type='FUTR3DHead',
        use_dab=True,
        anchor_size=3,
        num_query=900,
        num_classes=10,
        in_channels=256,
        pc_range=point_cloud_range,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],

        # ==================== Transformer with Three-Sensor Fusion ====================
        transformer=dict(
            type='FUTR3DTransformer',
            use_dab=True,
            decoder=dict(
                type='FUTR3DTransformerDecoder',
                num_layers=6,
                use_dab=True,
                anchor_size=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        # Self-attention
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        # Cross-attention with THREE sensors fusion
                        dict(
                            type='FUTR3DAttention',
                            use_lidar=True,       # LiDAR enabled
                            use_camera=True,      # Camera enabled
                            use_radar=True,       # Radar enabled
                            pc_range=point_cloud_range,
                            embed_dims=256,
                            radar_dims=64)        # Radar feature dimension
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),

        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0)),

    # ==================== Training and Testing Settings ====================
    train_cfg=dict(pts=dict(
        point_cloud_range=point_cloud_range,
        pc_range=point_cloud_range,
        grid_size=[1440, 1440, 40],
        voxel_size=voxel_size,
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0)))),
    test_cfg=dict(pts=dict(
        pc_range=point_cloud_range[:2],
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        nms_type='circle',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
        max_num=300,
        score_threshold=0,
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0])))

# ==================== Dataset Configuration ====================
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# ==================== Data Augmentation (optional) ====================
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

# ==================== Training Pipeline with THREE Sensors ====================
train_pipeline = [
    # Load Camera images (6 views)
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),

    # Load LiDAR points
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,  # Aggregate 9 LiDAR sweeps
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),

    # Load Radar points
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,        # Full 18-dimensional radar data
        sweeps_num=4,       # Aggregate 4 radar sweeps
        use_dim=radar_use_dims,  # Use [0,1,2,8,9,18]: x,y,z,rcs,vx,vy
        max_num=1200),      # Maximum radar points

    # Data augmentation
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),

    # Filtering
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),

    # Normalization
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PointShuffle'),

    # Formatting and collection - IMPORTANT: Include all three sensors
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'radar', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# ==================== Test Pipeline with THREE Sensors ====================
test_pipeline = [
    # Load Camera images
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),

    # Load LiDAR points
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),

    # Load Radar points
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=4,
        use_dim=radar_use_dims,
        max_num=1200),

    # Normalization
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),

    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img', 'radar'])
        ])
]

# ==================== Data Loaders ====================
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'nuscenes_infos_val.pkl'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'nuscenes_infos_val.pkl'))

# ==================== Training Configuration ====================
runner = dict(type='EpochBasedRunner', max_epochs=6)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'img_neck': dict(lr_mult=0.1),
            'pts_middle_encoder': dict(lr_mult=0.1),
            'pts_backbone': dict(lr_mult=0.1),
            'pts_neck': dict(lr_mult=0.1),
            'radar_voxel_encoder': dict(lr_mult=0.1),
            'radar_middle_encoder': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# Evaluation and checkpointing
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# Pre-trained checkpoint (optional - you may need to train from scratch or use a compatible checkpoint)
# load_from = 'checkpoint/lidar_cam_fusion_pretrained.pth'

find_unused_parameters=True
