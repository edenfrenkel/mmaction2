# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian'),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=14,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        multi_class=True,
        consensus=dict(type='LSTMConsensus',
                       input_size=2048, hidden_size=512),
        loss_cls=dict(type='BinaryLogisticRegressionLoss')))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/cater/max2action/videos/'
data_root_val = 'data/cater/max2action/videos/'
ann_file_train = 'data/cater/max2action/lists/actions_present/train_subsetT.txt'
ann_file_val = 'data/cater/max2action/lists/actions_present/train_subsetV.txt'
ann_file_test = 'data/cater/max2action/lists/actions_present/val.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=5),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=5),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=5),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        multi_class=True,
        num_classes=14,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        multi_class=True,
        num_classes=14,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        multi_class=True,
        num_classes=14,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
# optimizer = dict(type='Adam', lr=0.001)
# optimizer_config = dict()
# lr_config = dict(policy='fixed')
optimizer = dict(
    type='SGD', lr=0.000625*3/8, momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[16, 36])
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1, metrics=['mean_average_precision'],
    key_indicator='mean_average_precision', rule='greater')
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/i3d_nl_r50_32x2x5_50e_cater_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1)]
