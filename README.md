# Deepfake Detection Models

This repository contains several baseline models for spotting deepfakes in still images. The code was originally developed for the IEEE SPC DFWild Cup 2025 challenge but can be applied to other datasets.

## Repository Structure

```
deepfake_detection/
├── dataset.py    # Dataset definitions
├── main.py       # Command line entry point
├── models.py     # Model architectures and transforms
└── train.py      # Training utilities
```

The `deepfake_detection` package exposes reusable components so models no longer repeat dataset code.

## Requirements

The project relies on [PyTorch](https://pytorch.org) and [timm](https://github.com/huggingface/pytorch-image-models) for model definitions. The optional `wavelet_clip.py` script expects the [Wavelet‑CLIP](https://github.com/Girish-Krishnan/wavelet-clip) repository to be cloned next to this one. You can reproduce a full environment using the provided `Dockerfile`.

You can install the Python dependencies locally with:

```bash
pip install -r requirements.txt
```

## Dataset

The DFWild Cup dataset is available here:
<https://tcgcr-my.sharepoint.com/personal/md_sahidullah_tcgcrest_org/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmd%5Fsahidullah%5Ftcgcrest%5Forg%2FDocuments%2DFWild-Cup&ga=1>

After downloading, arrange the images into four folders:

```
train_real/   train_fake/
valid_real/   valid_fake/
```

## Usage

Training is performed through the command line interface. A typical invocation looks like:

```bash
python -m deepfake_detection.main \
    --model resnet \
    --train-real path/to/train_real \
    --train-fake path/to/train_fake \
    --val-real path/to/valid_real \
    --val-fake path/to/valid_fake \
    --epochs 20 --batch-size 32 --save-model resnet.pth
```

### Command Line Arguments

| Option | Description | Default |
| ------ | ----------- | ------- |
| `--model` | Name of the model to train (`basic_cnn`, `resnet`, `inception`, `vit`, `xception`, `wavelet_clip`) | required |
| `--train-real` | Directory containing real images for training | required |
| `--train-fake` | Directory containing fake images for training | required |
| `--val-real` | Directory with real images for validation | required |
| `--val-fake` | Directory with fake images for validation | required |
| `--epochs` | Number of training epochs | `10` |
| `--batch-size` | Mini-batch size | `32` |
| `--lr` | Learning rate | `0.001` |
| `--device` | Device to train on (e.g. `cuda:0`) | automatically chosen |
| `--save-model` | Path to save the best model weights | not saved |

Supported model names are listed in the table above. Training logs report accuracy and ROC‑AUC for every epoch. If `--save-model` is specified, the best performing checkpoint on the validation set is stored at that path.

## Citation

This repository is released under the MIT License. Feel free to use and modify the code for your own research.

