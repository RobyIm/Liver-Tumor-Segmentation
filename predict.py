"""
predict.py - Run inference with a trained generator model.

Loads a checkpoint and generates segmentation masks for input CT images.
Supports single images, directories, and batch processing.

Usage:
    # Single image
    python predict.py --checkpoint best_model.pth --input scan.png --output pred.png

    # Directory of images
    python predict.py --checkpoint best_model.pth --input ./images/ --output ./predictions/

    # With visualization (side-by-side CT + prediction)
    python predict.py --checkpoint best_model.pth --input scan.png --output pred.png --visualize

    # With ground truth comparison
    python predict.py --checkpoint best_model.pth --input ./images/ --output ./predictions/ --masks ./masks/ --visualize
"""

import argparse
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import Generator


def load_model(checkpoint_path, device):
    """
    Load a trained generator from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pth checkpoint file.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    generator : nn.Module
        Loaded generator model in eval mode.
    epoch : int
        Epoch the checkpoint was saved at.
    """
    generator = Generator(input_channels=1).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['generator_state_dict']

    # Handle torch.compile prefixed keys
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned[new_key] = v

    generator.load_state_dict(cleaned)
    generator.eval()

    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    print(f"Loaded model from epoch {epoch} (Val Loss: {val_loss})")

    return generator, epoch


def predict_single(generator, image_path, device, output_size=160):
    """
    Run inference on a single image.

    Parameters
    ----------
    generator : nn.Module
        Trained generator model.
    image_path : str
        Path to the input grayscale CT image.
    device : torch.device
        Compute device.
    output_size : int
        Size to resize the image to before inference.

    Returns
    -------
    img_original : np.ndarray
        Original resized image (H, W), float [0, 1].
    prediction : np.ndarray
        Predicted mask (H, W), float [0, 1].
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_resized = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        if device.type == 'cuda':
            with autocast(device_type='cuda'):
                pred = generator(img_tensor)
        else:
            pred = generator(img_tensor)

    prediction = pred.squeeze().cpu().float().numpy()
    img_original = img_resized.astype(np.float32) / 255.0

    return img_original, prediction


def save_prediction(prediction, output_path):
    """
    Save a prediction mask as a PNG image.

    Parameters
    ----------
    prediction : np.ndarray
        Predicted mask (H, W).
    output_path : str
        Path to save the output image.
    """
    # Binarize at 0.5 threshold and save as uint8
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    cv2.imwrite(output_path, binary_mask)


def save_visualization(img, prediction, output_path, mask=None):
    """
    Save a side-by-side visualization of CT, prediction, and optional ground truth.

    Parameters
    ----------
    img : np.ndarray
        Input CT image (H, W).
    prediction : np.ndarray
        Predicted mask (H, W).
    output_path : str
        Path to save the visualization.
    mask : np.ndarray or None, optional
        Ground truth mask for comparison.
    """
    num_cols = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

    axes[0].set_title("CT")
    axes[0].imshow(img, cmap="gray")
    axes[0].axis("off")

    if mask is not None:
        axes[1].set_title("GT Mask")
        axes[1].imshow(mask, cmap="gray")
        axes[1].axis("off")
        axes[2].set_title("Pred Mask")
        axes[2].imshow(prediction, cmap="gray")
        axes[2].axis("off")
    else:
        axes[1].set_title("Pred Mask")
        axes[1].imshow(prediction, cmap="gray")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def process_directory(generator, input_dir, output_dir, device, mask_dir=None, visualize=False):
    """
    Run inference on all PNG images in a directory.

    Parameters
    ----------
    generator : nn.Module
        Trained generator model.
    input_dir : str
        Directory containing input CT images.
    output_dir : str
        Directory to save predictions.
    device : torch.device
        Compute device.
    mask_dir : str or None, optional
        Directory containing ground truth masks for comparison.
    visualize : bool, optional
        Whether to save side-by-side visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg"))
    )

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        img, prediction = predict_single(generator, img_path, device)

        # Save binary mask
        save_prediction(prediction, os.path.join(output_dir, f"{name}_pred.png"))

        # Save visualization if requested
        if visualize:
            mask = None
            if mask_dir:
                mask_path = os.path.join(mask_dir, filename)
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask_img, (160, 160), interpolation=cv2.INTER_NEAREST)
                    mask = (mask.astype(np.float32) / 255.0 > 0).astype(np.float32)

            save_visualization(
                img, prediction,
                os.path.join(output_dir, f"{name}_vis.png"),
                mask=mask
            )

    print(f"Saved {len(image_files)} predictions to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Liver CT Segmentation - Inference")

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input image or directory of images"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output image or directory for predictions"
    )
    parser.add_argument(
        "--masks", type=str, default=None,
        help="Path to ground truth masks directory (for comparison)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save side-by-side visualization (CT + prediction + optional GT)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    generator, _ = load_model(args.checkpoint, device)

    # Run inference
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        process_directory(
            generator, args.input, args.output, device,
            mask_dir=args.masks, visualize=args.visualize
        )
    else:
        img, prediction = predict_single(generator, args.input, device)

        if args.visualize:
            mask = None
            if args.masks and os.path.exists(args.masks):
                mask_img = cv2.imread(args.masks, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask_img, (160, 160), interpolation=cv2.INTER_NEAREST)
                mask = (mask.astype(np.float32) / 255.0 > 0).astype(np.float32)
            save_visualization(img, prediction, args.output, mask=mask)
        else:
            save_prediction(prediction, args.output)

        print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()