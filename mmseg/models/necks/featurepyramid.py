# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from ..builder import NECKS


@NECKS.register_module()
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
                 rescales=[4, 2, 1, 0.5],
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
        assert len(inputs) == len(self.rescales)
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
# class Feature2Pyramid(nn.Module):
#     def __init__(self,
#                  embed_dims=[96, 176, 216, 216],
#                  rescales=[4, 2, 1, 0.5],
#                  norm_cfg=dict(type='SyncBN', requires_grad=True)):
#         super(Feature2Pyramid, self).__init__()
#         self.rescales = rescales
#
#         # 根据 rescales 列表初始化上采样和下采样层
#         self.ops = nn.ModuleDict()
#         for i, scale in enumerate(self.rescales):
#             embed_dim = embed_dims[i]
#             if scale == 4:
#                 self.ops[f'upsample_{scale}x'] = nn.Sequential(
#                     nn.ConvTranspose2d(
#                         embed_dim, embed_dim // 2, kernel_size=2, stride=2), # 注意通道数的变化
#                     build_norm_layer(norm_cfg, embed_dim // 2)[1],
#                     nn.GELU(),
#                     nn.ConvTranspose2d(
#                         embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2), # 注意通道数的变化
#                 )
#             elif scale == 2:
#                 self.ops[f'upsample_{scale}x'] = nn.Sequential(
#                     nn.ConvTranspose2d(
#                         embed_dim, embed_dim // 2, kernel_size=2, stride=2) # 注意通道数的变化
#                 )
#             elif scale == 1:
#                 self.ops[f'identity'] = nn.Identity()
#             elif scale == 0.5:
#                 self.ops[f'downsample_{int(1/scale)}x'] = nn.MaxPool2d(kernel_size=2, stride=2)
#             elif scale == 0.25:
#                 self.ops[f'downsample_{int(1/scale)}x'] = nn.MaxPool2d(kernel_size=4, stride=4)
#             else:
#                 raise KeyError(f'invalid {scale} for feature2pyramid')
#
#     def forward(self, inputs):
#         assert len(inputs) == len(self.rescales)
#         outputs = []
#         for i, x in enumerate(inputs):
#             scale = self.rescales[i]
#             if scale == 1:
#                 op_key = 'identity'
#             else:
#                 op_key = f'upsample_{scale}x' if scale > 1 else f'downsample_{int(1 / scale)}x'
#             outputs.append(self.ops[op_key](x))
#         return tuple(outputs)

