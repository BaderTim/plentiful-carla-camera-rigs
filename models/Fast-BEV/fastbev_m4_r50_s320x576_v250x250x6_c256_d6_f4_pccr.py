rig_name = 'R1'
class_names = [
    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
    'traffic_light', 'traffic_sign'
]
class_range = dict(
    car=80,
    truck=80,
    bus=80,
    bicycle=80,
    motorcycle=80,
    adult=80,
    child=80,
    traffic_light=80,
    traffic_sign=80)
data_root = '/data/R1/'
model = dict(
    type='FastBEV',
    style='v1',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=[256], out_channels=[64]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=384,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        fuse=dict(in_channels=1536, out_channels=384),
        norm_cfg=dict(type='BN', requires_grad=True)),
    seg_head=None,
    bbox_head=dict(
        type='FreeAnchor3DHead',
        is_transpose=True,
        num_classes=9,
        in_channels=256,
        feat_channels=256,
        num_convs=0,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-80, -80, -1.8, 80, 80, -1.8]],
            sizes=[[4.67, 1.93, 1.53], [6.37, 2.61, 2.47], [5.92, 2.07, 2.56],
                   [1.66, 0.66, 1.62], [2.04, 0.8, 1.53], [0.38, 0.38, 1.86],
                   [0.5, 0.5, 1.1], [0.57, 0.57, 0.87], [0.8, 0.2, 0.8]],
            custom_values=[],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    multi_scale_id=[0],
    n_voxels=[[250, 250, 6]],
    voxel_size=[[0.64, 0.64, 1.33]],
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=300,
        use_scale_nms=False,
        use_tta=False,
        nms_across_levels=False,
        use_rotate_nms=True,
        nms_thr=0.2))
point_cloud_range = [-80, -80, -6, 80, 80, 6]
dataset_type = 'PCCRMultiViewDataset'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_config = dict(
    src_size=(720, 1280),
    input_size=(320, 576),
    resize=(-0.06, 0.11),
    crop=(-0.05, 0.05),
    rot=(-5.4, 5.4),
    flip=True,
    test_input_size=(320, 576),
    test_resize=0.0,
    test_rotate=0.0,
    test_flip=False,
    pad=(0, 0, 0, 0),
    pad_divisor=32,
    pad_color=(0, 0, 0))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=True,
        n_images=None,
        n_times=4,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'))
        ]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='RandomFlip3D',
        flip_2d=False,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.05, 0.05, 0.05],
        update_img2lidar=True),
    dict(
        type='RandomAugImageMultiViewImage',
        data_config=dict(
            src_size=(720, 1280),
            input_size=(320, 576),
            resize=(-0.06, 0.11),
            crop=(-0.05, 0.05),
            rot=(-5.4, 5.4),
            flip=True,
            test_input_size=(320, 576),
            test_resize=0.0,
            test_rotate=0.0,
            test_flip=False,
            pad=(0, 0, 0, 0),
            pad_divisor=32,
            pad_color=(0, 0, 0))),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[-80, -80, -6, 80, 80, 6]),
    dict(type='KittiSetOrigin', point_cloud_range=[-80, -80, -6, 80, 80, 6]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_light', 'traffic_sign'
        ]),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=True,
        n_images=None,
        n_times=4,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'))
        ]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='RandomAugImageMultiViewImage',
        data_config=dict(
            src_size=(720, 1280),
            input_size=(320, 576),
            resize=(-0.06, 0.11),
            crop=(-0.05, 0.05),
            rot=(-5.4, 5.4),
            flip=True,
            test_input_size=(320, 576),
            test_resize=0.0,
            test_rotate=0.0,
            test_flip=False,
            pad=(0, 0, 0, 0),
            pad_divisor=32,
            pad_color=(0, 0, 0)),
        is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=[-80, -80, -6, 80, 80, 6]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_light', 'traffic_sign'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='PCCRMultiViewDataset',
        data_root='/data/R1/',
        ann_file='/data/R1/R1_infos_train.pkl',
        pipeline=[
            dict(
                type='MultiViewPipeline',
                sequential=True,
                n_images=None,
                n_times=4,
                transforms=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk'))
                ]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='LoadPointsFromFile',
                dummy=True,
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(
                type='RandomFlip3D',
                flip_2d=False,
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,
                update_img2lidar=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0.05, 0.05, 0.05],
                update_img2lidar=True),
            dict(
                type='RandomAugImageMultiViewImage',
                data_config=dict(
                    src_size=(720, 1280),
                    input_size=(320, 576),
                    resize=(-0.06, 0.11),
                    crop=(-0.05, 0.05),
                    rot=(-5.4, 5.4),
                    flip=True,
                    test_input_size=(320, 576),
                    test_resize=0.0,
                    test_rotate=0.0,
                    test_flip=False,
                    pad=(0, 0, 0, 0),
                    pad_divisor=32,
                    pad_color=(0, 0, 0))),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-80, -80, -6, 80, 80, 6]),
            dict(
                type='KittiSetOrigin',
                point_cloud_range=[-80, -80, -6, 80, 80, 6]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_light', 'traffic_sign'
                ]),
            dict(
                type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_light', 'traffic_sign'
        ],
        class_range=dict(
            car=80,
            truck=80,
            bus=80,
            bicycle=80,
            motorcycle=80,
            adult=80,
            child=80,
            traffic_light=80,
            traffic_sign=80),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=1,
        with_velocity=False,
        sequential=True,
        n_times=4,
        train_adj_ids=[1, 3, 5],
        speed_mode='none',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        prev_only=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None),
    val=dict(
        type='PCCRMultiViewDataset',
        data_root='/data/R1/',
        ann_file='/data/R1/R1_infos_val.pkl',
        pipeline=[
            dict(
                type='MultiViewPipeline',
                sequential=True,
                n_images=None,
                n_times=4,
                transforms=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk'))
                ]),
            dict(
                type='LoadPointsFromFile',
                dummy=True,
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(
                type='RandomAugImageMultiViewImage',
                data_config=dict(
                    src_size=(720, 1280),
                    input_size=(320, 576),
                    resize=(-0.06, 0.11),
                    crop=(-0.05, 0.05),
                    rot=(-5.4, 5.4),
                    flip=True,
                    test_input_size=(320, 576),
                    test_resize=0.0,
                    test_rotate=0.0,
                    test_flip=False,
                    pad=(0, 0, 0, 0),
                    pad_divisor=32,
                    pad_color=(0, 0, 0)),
                is_train=False),
            dict(
                type='KittiSetOrigin',
                point_cloud_range=[-80, -80, -6, 80, 80, 6]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_light', 'traffic_sign'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ],
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_light', 'traffic_sign'
        ],
        class_range=dict(
            car=80,
            truck=80,
            bus=80,
            bicycle=80,
            motorcycle=80,
            adult=80,
            child=80,
            traffic_light=80,
            traffic_sign=80),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=1,
        with_velocity=False,
        sequential=True,
        n_times=4,
        train_adj_ids=[1, 3, 5],
        speed_mode='none',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        prev_only=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None),
    test=dict(
        type='PCCRMultiViewDataset',
        data_root='/data/R1/',
        ann_file='/data/R1/R1_infos_test.pkl',
        pipeline=[
            dict(
                type='MultiViewPipeline',
                sequential=True,
                n_images=None,
                n_times=4,
                transforms=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk'))
                ]),
            dict(
                type='LoadPointsFromFile',
                dummy=True,
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(
                type='RandomAugImageMultiViewImage',
                data_config=dict(
                    src_size=(720, 1280),
                    input_size=(320, 576),
                    resize=(-0.06, 0.11),
                    crop=(-0.05, 0.05),
                    rot=(-5.4, 5.4),
                    flip=True,
                    test_input_size=(320, 576),
                    test_resize=0.0,
                    test_rotate=0.0,
                    test_flip=False,
                    pad=(0, 0, 0, 0),
                    pad_divisor=32,
                    pad_color=(0, 0, 0)),
                is_train=False),
            dict(
                type='KittiSetOrigin',
                point_cloud_range=[-80, -80, -6, 80, 80, 6]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult',
                    'child', 'traffic_light', 'traffic_sign'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ],
        classes=[
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child',
            'traffic_light', 'traffic_sign'
        ],
        class_range=dict(
            car=80,
            truck=80,
            bus=80,
            bicycle=80,
            motorcycle=80,
            adult=80,
            child=80,
            traffic_light=80,
            traffic_sign=80),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=1,
        with_velocity=False,
        sequential=True,
        n_times=4,
        train_adj_ids=[1, 3, 5],
        speed_mode='none',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        prev_only=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None))
optimizer = dict(
    type='AdamW2',
    lr=0.0004,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0,
    by_epoch=False)
total_epochs = 40
checkpoint_config = dict(interval=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=5)
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale='dynamic')
dataset_name = 'R1'
train_ann_file = '/data/R1/R1_infos_train.pkl'
val_ann_file = '/data/R1/R1_infos_val.pkl'
test_ann_file = '/data/R1/R1_infos_test.pkl'
work_dir = './work_dirs/train_R1'
gpu_ids = range(0, 2)
