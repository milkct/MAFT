
#自己做的火星岩石数据集
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOStuffDataset(CustomDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = ('rock','background')

    PALETTE = [[0, 0, 0], [128,0,0]]

    def __init__(self, **kwargs):
        super(COCOStuffDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_labelTrainIds.png', **kwargs)
