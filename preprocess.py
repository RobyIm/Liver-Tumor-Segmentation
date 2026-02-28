"""
preprocess.py - Convert NIfTI volumes to PNG slices.

Reads LiTS Challenge NIfTI files (volume-*.nii and segmentation-*.nii),
extracts 2D slices, normalizes CT values, and saves as PNG images.
Uses multiprocessing for parallel volume processing.

Usage:
    python preprocess.py
"""

import os
import numpy as np
import nibabel as nib
import cv2
from multiprocessing import Pool, cpu_count

# Configuration
DATA_DIR = "/content/lits_raw/Training Batch 2"
OUT_DIR = "/content/lits_png"
MAX_SLICES = 1000


def normalize_ct(x):
    """
    Normalize CT scan values to [0, 255] range.

    Clips HU values to [-200, 300] window (suitable for abdominal CT),
    then normalizes to uint8 range.

    Parameters
    ----------
    x : np.ndarray
        Raw CT slice in Hounsfield units.

    Returns
    -------
    np.ndarray
        Normalized uint8 image.
    """
    x = np.clip(x, -200, 300)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 255).astype(np.uint8)


def process_volume(file_number):
    """
    Process a single NIfTI volume and its segmentation into PNG slices.

    Parameters
    ----------
    file_number : int
        The file index (e.g., 33 for volume-33.nii).

    Returns
    -------
    str
        Status message.
    """
    seg_path = os.path.join(DATA_DIR, f"segmentation-{file_number}.nii")
    vol_path = os.path.join(DATA_DIR, f"volume-{file_number}.nii")

    if not (os.path.exists(seg_path) and os.path.exists(vol_path)):
        return f"Skipped {file_number}"

    seg_nii = nib.load(seg_path)
    vol_nii = nib.load(vol_path)

    seg_proxy = seg_nii.dataobj
    vol_proxy = vol_nii.dataobj

    depth = seg_nii.shape[2]
    num_slices = min(MAX_SLICES, depth)
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    img_dir = f"{OUT_DIR}/images/{file_number}"
    mask_dir = f"{OUT_DIR}/masks/{file_number}"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in slice_indices:
        img = normalize_ct(np.array(vol_proxy[:, :, i], dtype=np.float32))
        mask = np.array(seg_proxy[:, :, i], dtype=np.uint8)

        cv2.imwrite(f"{img_dir}/{i}.png", img)
        cv2.imwrite(f"{mask_dir}/{i}.png", mask)

    return f"Done {file_number}"


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Auto-detect all volume files
    file_numbers = sorted([
        int(f.split('-')[1].split('.')[0])
        for f in os.listdir(DATA_DIR)
        if f.startswith("volume-")
    ])

    print("Found volumes:", len(file_numbers))

    # Parallel processing
    num_workers = max(1, cpu_count() - 1)
    print("Using workers:", num_workers)

    with Pool(num_workers) as p:
        results = p.map(process_volume, file_numbers)

    print("\n".join(results))
    print("All volumes processed")
