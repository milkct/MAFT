'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer

from mmseg.ops import resize
import sys
sys.path.append('D:/GIT/AFFormer-ww/mmseg/models/decode_heads')

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 embed_dim=[96,176,216,216],
                 rescales=[4,2,1,0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(Feature2Pyramid, self).__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    build_norm_layer(norm_cfg, embed_dim)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, inputs):
        # print(len(inputs))
        # print(len(self.rescales))
        # assert len(inputs) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))
        return tuple(outputs)

@HEADS.register_module()
class CLS(BaseDecodeHead):
    def __init__(self,
                 aff_channels=512,
                 aff_kwargs=dict(),
                 **kwargs):
        super(CLS, self).__init__(
                input_transform='multiple_select', **kwargs)
        self.aff_channels = aff_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        

        self.align = ConvModule(
            self.aff_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.neck = Feature2Pyramid(embed_dim=216)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        # print(len(inputs))
        inputs = self.neck(inputs)
        inputs = inputs[0]
        # print(len(inputs))
        # inputs = self.neck(inputs)

        x = self.squeeze(inputs)
            
        output = self.cls_seg(x)
        return output
