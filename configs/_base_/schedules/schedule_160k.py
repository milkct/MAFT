# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=800)#指定迭代次数
checkpoint_config = dict(by_epoch=False, interval=800)#模型保存的次数，这里显示的是每16000次保存一次
evaluation = dict(interval=800, metric='mIoU', pre_eval=True)#训练到1000次时进行一次预评估
