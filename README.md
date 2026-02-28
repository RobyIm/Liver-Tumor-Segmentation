# Liver CT Segmentation using GAN

Automatic liver segmentation from CT scan images using a Generative Adversarial Network with a multi-layer perceptual loss. Trained on the [LiTS (Liver Tumor Segmentation) Challenge](https://competitions.codesourcearchive.com/competitions/17094) dataset.

## Overview

This project trains a GAN to predict binary liver segmentation masks from CT slices. The generator produces masks while the discriminator provides hierarchical feature-based feedback through a custom multi-layer L1 loss. A direct pixel loss is combined with the adversarial loss to stabilize training.

## Architecture

### Generator

Encoder-decoder CNN with 4 strided downsampling layers, 3 same-resolution bottleneck layers, and 4 nearest-neighbor upsampling stages.

```
Input (1×160×160)
  → Conv(1→64, stride 2) → Conv(64→128, stride 2) → Conv(128→256, stride 2) → Conv(256→512, stride 2)
  → Conv(512→256) → Upsample ×2
  → Conv(256→128) → Upsample ×2
  → Conv(128→64)  → Upsample ×2
  → Conv(64→1)    → Upsample ×2
Output (1×160×160)
```

### Discriminator

3-layer convolutional critic used both for adversarial training and as a feature extractor for the perceptual loss. Features are extracted from all layers via `forward_all_layers()` with adaptive average pooling to 8×8.

```
Input (1×160×160) → Conv(1→128)+BN+LeakyReLU → Conv(128→256)+BN+LeakyReLU → Conv(256→1)+LeakyReLU
```

### Loss Function

The generator loss combines two components:

```
g_loss = pixel_loss + 0.01 × adversarial_loss
```

- **Pixel Loss**: Direct L1 (MAE) between predicted and ground truth masks
- **Adversarial Perceptual Loss**: Multi-layer L1 distance between discriminator features of `image × predicted_mask` vs `image × ground_truth_mask`

## Project Structure

```
├── README.md
├── preprocess.py           # NIfTI to PNG conversion with parallel processing
├── dataset.py              # LiverCTDataset - loads PNG slices with cv2 resize
├── models.py               # Generator and Discriminator networks
├── loss.py                 # Multi-layer perceptual L1 loss
├── train.py                # Training loop with validation, AMP, checkpointing
├── resume_training.py      # Resume training from a saved checkpoint
└── Semantic_Segmentation.ipynb  # Full notebook (Google Colab)
```

## Dataset

### Source

Download the LiTS dataset (NIfTI format) and place it in a directory:

```
Training Batch 2/
├── volume-0.nii
├── segmentation-0.nii
├── volume-1.nii
├── segmentation-1.nii
└── ...
```

### Preprocessing

Convert NIfTI volumes to PNG slices using parallel processing:

```bash
python preprocess.py
```

This produces the following structure:

```
lits_png/
├── images/
│   ├── 0/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── 1/
│       └── ...
└── masks/
    ├── 0/
    │   ├── 0.png
    │   └── ...
    └── 1/
        └── ...
```

CT images are windowed to [-200, 300] HU and normalized to [0, 255]. Masks are saved as binary PNGs.

## Training

### From Scratch

```bash
python train.py
```

### Resume from Checkpoint

```bash
python resume_training.py
```

### Key Training Settings

| Parameter | Value |
|-----------|-------|
| Batch size | 10 |
| Learning rate (G) | 0.0002 |
| Learning rate (D) | 0.0002 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Mixed precision | FP16 (AMP) |
| Train/Val split | 85% / 15% |
| Image size | 160×160 |

### Outputs

- Best model checkpoint saved to `checkpoints/best_model.pth`
- Validation predictions saved to `checkpoints/val_predictions/epoch_XXXX/`
- Loss curves saved to `checkpoints/loss_curve.png`

## Optimizations

Several performance optimizations are applied:

- **Mixed precision training** (AMP) for ~1.5–2× GPU speedup
- **Single-pass feature extraction** via `forward_all_layers()` instead of per-layer forward passes
- **Adaptive average pooling** (8×8) before flattening discriminator features to reduce backward pass cost
- **cv2.resize** in dataset instead of `torch.nn.functional.interpolate`
- **torch.compile** for model graph optimization
- **Discriminator freezing** during generator training step
- **Persistent DataLoader workers** to avoid process respawn overhead

## Requirements

```
torch >= 2.0
torchvision
opencv-python
nibabel
numpy
matplotlib
```

## Google Colab

The notebook `Semantic_Segmentation.ipynb` is designed to run on Google Colab with GPU. It mounts Google Drive for dataset storage and checkpoint saving.

1. Upload the LiTS dataset to Google Drive
2. Open the notebook in Colab
3. Run cells in order: Preprocessing → Dataset → Models → Loss → Training

## License

This project is for educational and research purposes.
