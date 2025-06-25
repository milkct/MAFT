'''
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/model/backbone/__init__.py
'''

from .resnet import ResNet
# from mmseg.models.backbones.MAFT_3 import afformer_tiny,afformer_small,afformer_base
from mmseg.models.backbones.afformer_last import afformer_tiny,afformer_small,afformer_base
#
# from mmseg.models.backbones.afformer import afformer_tiny,afformer_small,afformer_base
#

__all__ = [
    'ResNet','afformer_tiny','afformer_small','afformer_base']

