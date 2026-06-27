point_cloud_range = [-51.2, -51.2, -6.0, 51.2, 51.2, 6.0]
class_names = [
    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
    'traffic_sign', 'traffic_light'
]
dataset_type = 'PCCRDataset'
data_root = '/data/R1/'
file_client_args = dict(backend='disk')
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
pccr_class_range = dict(
    car=50,
    truck=50,
    bus=50,
    bicycle=40,
    motorcycle=40,
    adult=40,
    child=40,
    traffic_sign=30,
    traffic_light=30)
data_config = dict(
    cams=None,
    Ncams=None,
    input_size=(384, 704),
    src_size=(720, 1280),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=dict(
            cams=None,
            Ncams=None,
            input_size=(384, 704),
            src_size=(720, 1280),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0)),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light'
        ]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light'
        ]),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=dict(
            cams=None,
            Ncams=None,
            input_size=(384, 704),
            src_size=(720, 1280),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0)),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light'
        ],
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_sign', 'traffic_light'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]
share_data_config = dict(
    type='PCCRDataset',
    classes=[
        'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
        'traffic_sign', 'traffic_light'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet',
    with_velocity=False,
    discard_velocity_eval=True,
    class_range=dict(
        car=50,
        truck=50,
        bus=50,
        bicycle=40,
        motorcycle=40,
        adult=40,
        child=40,
        traffic_sign=30,
        traffic_light=30),
    ego_cam='auto')
test_data_config = dict(
    data_root='/data/R1/',
    pipeline=[
        dict(
            type='PrepareImageInputs',
            data_config=dict(
                cams=None,
                Ncams=None,
                input_size=(384, 704),
                src_size=(720, 1280),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0)),
        dict(type='LoadAnnotations'),
        dict(
            type='BEVAug',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                'child', 'traffic_sign', 'traffic_light'
            ],
            is_train=False),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'bus', 'bicycle', 'motorcycle',
                        'adult', 'child', 'traffic_sign', 'traffic_light'
                    ],
                    with_label=False),
                dict(type='Collect3D', keys=['points', 'img_inputs'])
            ])
    ],
    ann_file='/data/R1/R1_infos_val.pkl',
    test_mode=True,
    box_type_3d='LiDAR',
    type='PCCRDataset',
    classes=[
        'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
        'traffic_sign', 'traffic_light'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet',
    with_velocity=False,
    discard_velocity_eval=True,
    class_range=dict(
        car=50,
        truck=50,
        bus=50,
        bicycle=40,
        motorcycle=40,
        adult=40,
        child=40,
        traffic_sign=30,
        traffic_light=30),
    ego_cam='auto')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root='/data/R1/',
        ann_file='/data/R1/R1_infos_train.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                is_train=True,
                data_config=dict(
                    cams=None,
                    Ncams=None,
                    input_size=(384, 704),
                    src_size=(720, 1280),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0)),
            dict(type='LoadAnnotations'),
            dict(
                type='BEVAug',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_sign', 'traffic_light'
                ]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_sign', 'traffic_light'
                ]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_sign', 'traffic_light'
                ]),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light'
        ],
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        type='PCCRDataset',
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        img_info_prototype='bevdet',
        with_velocity=False,
        discard_velocity_eval=True,
        class_range=dict(
            car=50,
            truck=50,
            bus=50,
            bicycle=40,
            motorcycle=40,
            adult=40,
            child=40,
            traffic_sign=30,
            traffic_light=30),
        ego_cam='auto'),
    val=dict(
        data_root='/data/R1/',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=None,
                    Ncams=None,
                    input_size=(384, 704),
                    src_size=(720, 1280),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0)),
            dict(type='LoadAnnotations'),
            dict(
                type='BEVAug',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_sign', 'traffic_light'
                ],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'bus', 'bicycle', 'motorcycle',
                            'adult', 'child', 'traffic_sign', 'traffic_light'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img_inputs'])
                ])
        ],
        ann_file='/data/R1/R1_infos_val.pkl',
        test_mode=True,
        box_type_3d='LiDAR',
        type='PCCRDataset',
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        img_info_prototype='bevdet',
        with_velocity=False,
        discard_velocity_eval=True,
        class_range=dict(
            car=50,
            truck=50,
            bus=50,
            bicycle=40,
            motorcycle=40,
            adult=40,
            child=40,
            traffic_sign=30,
            traffic_light=30),
        ego_cam='auto'),
    test=dict(
        data_root='/data/R1/',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=None,
                    Ncams=None,
                    input_size=(384, 704),
                    src_size=(720, 1280),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0)),
            dict(type='LoadAnnotations'),
            dict(
                type='BEVAug',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_sign', 'traffic_light'
                ],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'bus', 'bicycle', 'motorcycle',
                            'adult', 'child', 'traffic_sign', 'traffic_light'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img_inputs'])
                ])
        ],
        ann_file='/data/R1/R1_infos_test.pkl',
        test_mode=True,
        box_type_3d='LiDAR',
        type='PCCRDataset',
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_sign', 'traffic_light'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        img_info_prototype='bevdet',
        with_velocity=False,
        discard_velocity_eval=True,
        class_range=dict(
            car=50,
            truck=50,
            bus=50,
            bicycle=40,
            motorcycle=40,
            adult=40,
            child=40,
            traffic_sign=30,
            traffic_light=30),
        ego_cam='auto'))
key = 'test'
evaluation = dict(interval=1)
checkpoint_config = dict(interval=4)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/petr_r50dcn_gridmask_p4/R1'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
grid_config = dict(
    x=[-51.2, 51.2, 0.8],
    y=[-51.2, 51.2, 0.8],
    z=[-6, 6, 12],
    depth=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet',
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=dict(
            x=[-51.2, 51.2, 0.8],
            y=[-51.2, 51.2, 0.8],
            z=[-6, 6, 12],
            depth=[1.0, 60.0, 1.0]),
        input_size=(384, 704),
        in_channels=256,
        out_channels=64,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet', numC_input=64, num_channels=[128, 256, 512]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(
                num_class=9,
                class_names=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_sign', 'traffic_light'
                ])
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-51.2, -51.2, -6.0, 51.2, 51.2, 6.0],
            grid_size=[1024, 1024, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=500,
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                                ])))
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=1e-07)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24])
runner = dict(type='EpochBasedRunner', max_epochs=24)
custom_hooks = [
    dict(type='MEGVIIEMAHook', init_updates=10560, priority='NORMAL')
]
gpu_ids = range(0, 2)
