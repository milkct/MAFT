'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''
from .aff_head import CLS
from .decode_head import BaseDecodeHead
from .aspp_head import ASPPHead
from .fcn_head import FCNHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
__all__ = ['CLS', 'BaseDecodeHead','ASPPHead','FCNHead','PSAHead','segformer_head',
           'PSPHead']