# Luzhou-Flavor Liquor Daqu Cross-section Semantic Segmentation

This repository contains code for a multiple classification image segmentation model based on [UNet](https://arxiv.org/pdf/1505.04597.pdf) [UNet++](https://arxiv.org/abs/1807.10165)


## Usage

#### Note : Use Python 3

### Dataset
make sure to put the files as the following structure:
```
data
├── images
|   ├── 0a7e06.jpg
│   ├── 0aab0a.jpg
│   ├── 0b1761.jpg
│   ├── ...
|
└── masks
    ├── 0a7e06.png
    ├── 0aab0a.png
    ├── 0b1761.png
    ├── ...
```
mask is a single-channel category index. For example, your dataset has three categories, mask should be 8-bit images with value 0,1,2 as the categorical value, this image looks black.



### Training
```bash
python train.py
```

## Tensorboard
You can visualize in real time the train and val losses, along with the model predictions with tensorboard:
```bash
tensorboard --logdir=runs
```

## Script Overview

| Script                | Function / Description                                                                 | Example Command                                                                                           |
|------------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `train.py`            | Train UNet / UNet++ segmentation model. Saves checkpoints and logs.                    | `python train.py`                                                                                         |
| `inference_color.py`  | Run inference and save **colored masks** for visualization.                            | `python inference_color.py -m ./data/checkpoints/epoch_150.pth -i ./data/test/input -o ./data/test/output` |
| `analyze_mask.py`     | Post-process masks: calculate class area ratio, fire cycle thickness, fissure length.  | `python analyze_mask.py -i ./data/test/output -o ./data/test/analysis --orig_dir ./data/test/input`       |
| `plot_training_log.py`| Plot training/validation curves (loss, mIoU, Dice, etc.) from training log CSV.        | `python plot_training_log.py --log ./training_log.csv`                                                    |

