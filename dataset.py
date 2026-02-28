"""
dataset.py - LiverCTDataset for loading preprocessed PNG slices.

Loads grayscale CT images and binary segmentation masks from a directory
structure, resizes them to a fixed output size, and returns them as tensors.

Usage:
    from dataset import LiverCTDataset
    dataset = LiverCTDataset("/path/to/lits_png", output_size=160)
"""

import os
import cv2
import torch
from torch.utils.data import Dataset


class LiverCTDataset(Dataset):
    """
    Dataset for liver CT segmentation.

    Expects the following directory structure:
        root_dir/
        ├── images/
        │   ├── case_000/
        │   │   ├── slice_000.png
        │   │   └── ...
        │   └── ...
        └── masks/
            ├── case_000/
            │   ├── slice_000.png
            │   └── ...
            └── ...

    Parameters
    ----------
    root_dir : str
        Path to the root directory containing 'images' and 'masks' folders.
    output_size : int, optional
        Target size for resizing images and masks. Default is 160.
    """

    def __init__(self, root_dir, output_size=160):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.output_size = output_size

        self.samples = []

        for case in sorted(os.listdir(self.img_dir)):
            case_img_dir = os.path.join(self.img_dir, case)
            case_mask_dir = os.path.join(self.mask_dir, case)

            if not os.path.isdir(case_img_dir):
                continue

            for file in sorted(os.listdir(case_img_dir)):
                img_path = os.path.join(case_img_dir, file)
                mask_path = os.path.join(case_mask_dir, file)

                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

        print("Total slices found:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and preprocess a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        mask : torch.Tensor
            Binary segmentation mask of shape (1, H, W).
        img : torch.Tensor
            Normalized CT image of shape (1, H, W).
        """
        img_path, mask_path = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        img = cv2.resize(img, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        mask = (torch.from_numpy(mask).unsqueeze(0).float() / 255.0 > 0).float()

        return mask, img
