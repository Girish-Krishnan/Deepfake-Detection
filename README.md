# Deepfake Detection Models

This repository provides a collection of baseline models for detecting deepfakes in still images. The code was developed while participating in the IEEE SPC DFWild Cup 2025 challenge.

## Repository structure

```
.deepfake_detection/      # Library containing datasets, models and training helpers
├── dataset.py
├── main.py               # Command line entry point
├── models.py
└── train.py
```

The new `deepfake_detection` package exposes reusable components so that models no longer repeat the same dataset code.

## Requirements

The project relies on PyTorch and `timm` for model definitions. The optional `wavelet_clip.py` script expects the [Wavelet-CLIP](https://github.com/lalithbharadwajbaru/wavelet-clip) repository to be cloned next to this one. A complete environment can be reproduced with the provided `Dockerfile`.

## Dataset

The DFWild Cup dataset can be downloaded from the following link:
<https://tcgcr-my.sharepoint.com/personal/md_sahidullah_tcgcrest_org/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmd%5Fsahidullah%5Ftcgcrest%5Forg%2FDocuments%2FDFWild-Cup&ga=1>

After downloading, organise the dataset into four folders:

```
train_real/   train_fake/
valid_real/   valid_fake/
```

## Usage

The recommended way to train a model is through the command line interface:

```bash
python -m deepfake_detection.main \
    --model resnet \
    --train-real path/to/train_real \
    --train-fake path/to/train_fake \
    --val-real path/to/valid_real \
    --val-fake path/to/valid_fake \
    --epochs 20 --batch-size 32 --save-model resnet.pth
```

Supported model names are `basic_cnn`, `resnet`, `inception`, `vit`, `xception` and `wavelet_clip`. Training logs will report accuracy and ROC‑AUC for every epoch. The best performing model on the validation set is saved if `--save-model` is provided.

## Citation

This repository is released under the MIT License. Feel free to use and modify the code for your own research.
