<<<<<<< HEAD
# MAFT: Lightweight Semantic Segmentation Toolbox

This repository provides the official PyTorch implementation for MAFT and related lightweight semantic segmentation models.

## Features

- Modular and extensible codebase based on MMSegmentation.
- Supports training, evaluation, and visualization.
- Includes multiple backbone architectures and configuration files.
- Ready-to-use scripts for single/multi-GPU training and testing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/milkct/MAFT
   cd MAFT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > For more details, see `requirements/` for optional, test, and documentation dependencies.

3. (Optional) Install MMCV and MMSegmentation compatible versions if needed:
   ```bash
   pip install mmcv-full==1.5.0
   pip install mmsegmentation==0.21.1
   ```

## Dataset Preparation

- Prepare your dataset following the MMSegmentation format.
- Datasets are provided in `https://github.com/milkct/TWMARS-V2'

## Training

Single-GPU:
```bash
python tools/train.py configs/AFFormer/AFFormer_base_ade20k.py
```

Multi-GPU:
```bash
bash tools/dist_train.sh configs/AFFormer/AFFormer_base_ade20k.py <GPU_NUM>
```

## Evaluation

Single-GPU:
```bash
python tools/test.py configs/AFFormer/AFFormer_base_ade20k.py /path/to/checkpoint.pth
```

Multi-GPU:
```bash
bash tools/dist_test.sh configs/AFFormer/AFFormer_base_ade20k.py /path/to/checkpoint.pth <GPU_NUM>
```

## Visualization

Visualize a single image:
```bash
python demo/image_demo.py <IMAGE_PATH> <CONFIG_PATH> <CHECKPOINT_PATH> --device cuda:0
```

## Pretrained Weights

Pretrained weights are available in the https://pan.baidu.com/s/1aZtsEEzKkHBYh2uc7KtTtQ 提取码: p5x3