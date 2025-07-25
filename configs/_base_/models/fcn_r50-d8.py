# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    # decode_head=dict(
    #     type='CLS',
    #     # type='SegformerHead',
    #     in_channels=256,
    #     in_index=[0, 1, 2 ,3],
    #     channels=512,
    #     aff_channels=512,
    #     dropout_ratio=0.1,
    #     num_classes=150,
    #     norm_cfg=ham_norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='DiceLoss', use_sigmoid=False, loss_weight=1.0)),
    #         # type='LovaszLoss',reduction='none',loss_weight=1.0)),
    #         # type='FocalLoss')),
    #         # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # # model training and testing settings
    # train_cfg=dict(),
    # test_cfg=dict(mode='whole'))

    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
