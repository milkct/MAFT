'''
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/model/necks/__init__.py
'''
from .fpn import FPN
from .featurepyramid import Feature2Pyramid
__all__ = ['FPN','Feature2Pyramid']
