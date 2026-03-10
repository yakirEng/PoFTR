import numpy as np
import torch
from pathlib import Path
import pandas as pd
import ast

import albumentations as A
import cv2



def _unpack_sample(sample, **kwargs):
        fields = [
            'image0', 'image1', 'depth0', 'depth1', 'mask0', 'mask1', 'T_0to1', 'T_1to0', 't_fpa'
        ]
        unpacked = []
        for field in fields:
            if kwargs.get(field, False):
                unpacked.append(getattr(sample, field))
        return tuple(unpacked)

def _pack_sample(sample, **kwargs):
    for key, value in kwargs.items():
        if hasattr(sample, key):
            setattr(sample, key, value)
    return sample

def load_sample_dict(df, idx):
    if len(df) == 0:
        raise RuntimeError("No records found in the dataset.")
    try:
        # Retrieve the row at index `idx` and convert it to a dict
        sample_dict = df.iloc[idx].to_dict()
    except IndexError:
        raise IndexError(f"Index {idx} is out of bounds. No document found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while fetching the record: {e}")
    # Convert the sample_dict to a dictionary with appropriate types
    sample_dict = parse_sample_dict(sample_dict)
    return sample_dict

def numpy2tensor(*numpy_arrays):
    tensors = [torch.from_numpy(array).float() for array in numpy_arrays]
    return tuple(tensors)

def expand_dims(*numpy_arrays):
    np_channelized = [np.expand_dims(array, axis=0) for array in numpy_arrays]
    return tuple(np_channelized)



def norm_01_masked(x, mask, fill_value=0.0, eps=1e-8):
    """
    Normalizes x to [0, 1] based on the min/max values in the masked areas.
    This is used for the visual images to perform contrast stretching.

    :param x: Input array
    :param mask: Boolean mask (True = valid, False = boundary)
    :param fill_value: Value to use for masked-out areas.
    :param eps: Small epsilon to avoid division by zero
    :return: Normalized array
    """
    if not mask.any():  # No valid pixels
        return np.full_like(x, fill_value=fill_value, dtype=np.float32)

    valid_x = x[mask]
    v_min, v_max = valid_x.min(), valid_x.max()

    if v_max - v_min < eps:  # Handle case where all valid pixels have same value
        normed_full = np.full_like(x, fill_value=fill_value, dtype=np.float32)
        normed_full[mask] = 0.5  # Set to a neutral middle value
        return normed_full

    normed = (valid_x - v_min) / (v_max - v_min)
    normed_full = np.full_like(x, fill_value=fill_value, dtype=np.float32)
    normed_full[mask] = normed.astype(np.float32)

    return normed_full


def norm_global_zscore_masked(x, mask, mean, std, fill_value=-3.0, eps=1e-8):
    """
    Normalizes 'x' (Z-score). Fills invalid areas with a Sentinel Value.
    CHANGED: Default fill_value from 0.0 to -10.0 for safety.
    """
    # Initialize with Sentinel (Safety against 0.0 collision)
    normed_full = np.full_like(x, fill_value=fill_value, dtype=np.float32)

    if not mask.any():
        return normed_full

    # Normalize valid pixels
    # (x - mean) / std
    normed_full[mask] = (x[mask] - mean) / (std + eps)

    return normed_full


def normalize_sample(sample, stats):
    """
    Applies global Z-score normalization.
    """
    # 1. Get Source-of-Truth Masks
    # Use pixel_mask if available (robustness), else mask0
    mask0 = sample.get('pixel_mask0', sample['mask0']).astype(bool)
    mask1 = sample.get('pixel_mask1', sample['mask1']).astype(bool)

    # 2. Define Sentinel Value
    SENTINEL = -3.0

    # 3. Normalize with explicit fill_value
    sample['phys0'] = norm_global_zscore_masked(
        sample['phys0'], mask0,
        mean=stats['phys0']['mean'],
        std=stats['phys0']['std'],
        fill_value=SENTINEL
    )

    sample['phys1'] = norm_global_zscore_masked(
        sample['phys1'], mask1,
        mean=stats['phys1']['mean'],
        std=stats['phys1']['std'],
        fill_value=SENTINEL
    )

    sample['image0'] = norm_global_zscore_masked(
        sample['image0'], mask0,
        mean=stats['image0']['mean'],
        std=stats['image0']['std'],
        fill_value=SENTINEL
    )

    sample['image1'] = norm_global_zscore_masked(
        sample['image1'], mask1,
        mean=stats['image1']['mean'],
        std=stats['image1']['std'],
        fill_value=SENTINEL
    )

    return sample


def augmentation_picker(data_type, aug_type):
    """
    Returns the requested augmentation pipeline based on the data type and intensity.

    Args:
        data_type (str): 'image' or 'physics'.
        aug_type (str):
            - For image: 'gentle' (for unstable models) or 'standard' (baseline).
            - For physics: 'easy', 'medium' (recommended), or 'hard'.

    Returns:
        list: A list of Albumentations transforms.
    """

    # ==========================
    # 1. IMAGE AUGMENTATIONS
    # ==========================
    if data_type == 'image':

        if aug_type == 'gentle':
            # Reduced noise/dropout for unstable Transformers (ASPANformer)
            return [
                A.MotionBlur(blur_limit=5, p=0.3),
                A.CoarseDropout(
                    num_holes_range=(1, 6),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill=0.0,
                    p=0.3
                ),
                A.GaussNoise(std_range=(0.005, 0.05), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
            ]

        elif aug_type == 'standard':
            # Original baseline configuration (Good for LoFTR / stable models)
            return [
                A.MotionBlur(blur_limit=8, p=0.5),
                A.CoarseDropout(
                    num_holes_range=(2, 12),
                    hole_height_range=(12, 48),
                    hole_width_range=(12, 48),
                    fill=0.0,
                    p=0.5
                ),
                A.GaussNoise(std_range=(0.01, 0.1), p=0.6),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
            ]

        else:
            raise ValueError(f"Unknown image augmentation level: {aug_type}")

    # ==========================
    # 2. PHYSICS AUGMENTATIONS
    # ==========================
    elif data_type == 'physics':

        if aug_type == 'easy':
            # Slightly tougher than baseline, good for stability check
            return [
                A.CoarseDropout(
                    num_holes_range=(4, 14),
                    hole_height_range=(16, 52),
                    hole_width_range=(16, 52),
                    fill=0.0,
                    p=0.65
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.6),
                A.GaussNoise(std_range=(0.02, 0.07), p=0.65),
            ]

        elif aug_type == 'medium':
            # The "Hardened" Winner (Best for preventing overfitting)
            return [
                A.CoarseDropout(
                    num_holes_range=(6, 18),
                    hole_height_range=(20, 60),
                    hole_width_range=(20, 60),
                    fill=0.0,
                    p=0.75
                ),
                A.GaussianBlur(blur_limit=(5, 9), p=0.7),
                A.GaussNoise(std_range=(0.03, 0.09), p=0.7),
            ]

        elif aug_type == 'hard':
            # Destructive testing
            return [
                A.CoarseDropout(
                    num_holes_range=(10, 24),
                    hole_height_range=(24, 72),
                    hole_width_range=(24, 72),
                    fill=0.0,
                    p=0.85
                ),
                A.GaussianBlur(blur_limit=(7, 13), p=0.8),
                A.GaussNoise(std_range=(0.05, 0.12), p=0.8),
            ]

        else:
            raise ValueError(f"Unknown physics augmentation level: {aug_type}")

    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'image' or 'physics'.")


def apply_augmentation(sample, config):
    """
    Robust augmentation pipeline with Asymmetric Modality Dropout.

    Changes:
    - Added 'modality_dropout': Large holes applied ONLY to images, not physics.
    - This forces the network to use the physics prior to 'inpaint' the missing visual structure.
    """

    # 1. Define image augmentations
    image_transforms = augmentation_picker('image', config['data']['image_aug_level'])

    # 2. Define physics augmentations
    phys_transforms = augmentation_picker('physics', config['data']['phys_aug_level'])

    # 3. Create Two Pipelines
    # Pipeline A: For Physics (Standard noise/blur only)
    phys_pipeline = A.Compose(phys_transforms)

    # Pipeline B:
    image_pipeline = A.Compose(image_transforms)

    def _augment_robust(img, mask=None, pipeline=None):
        # 1. Cast to Float32 for precision:
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        original_shape = img.shape
        is_channel_first = (img.ndim == 3 and img.shape[0] == 1)

        if is_channel_first:
            img = img.transpose(1, 2, 0)  # (1, H, W) -> (H, W, 1)
        elif img.ndim == 2:
            img = img[:, :, None]  # (H, W) -> (H, W, 1)

        # -----------------------------------------------------------
        # 2. Mask-Aware Normalization logic
        # -----------------------------------------------------------
        if mask is not None:
            if mask.shape != img.shape[:2]:
                mask_for_calc = cv2.resize(
                    mask.astype(np.uint8),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_for_calc = mask.astype(bool)
            valid_pixels = img[mask_for_calc]
            if valid_pixels.size > 0:
                v_min = valid_pixels.min()
                v_max = valid_pixels.max()
            else:
                v_min, v_max = img.min(), img.max()
        else:
            v_min, v_max = img.min(), img.max()

        # Handle flat range
        epsilon = 1e-6
        range_span = v_max - v_min
        if range_span < epsilon:
            range_span = 1.0
        # Project to [0, 1]
        img_norm = (img - v_min) / range_span
        img_norm = np.clip(img_norm, 0.0, 1.0)

        # -----------------------------------------------------------
        # 3. Apply Selected Augmentation Pipeline
        # -----------------------------------------------------------
        # Default to phys_pipeline if none provided (safety)
        aug_func = pipeline if pipeline else phys_pipeline
        augmented_norm = aug_func(image=img_norm)["image"]

        # -----------------------------------------------------------
        # 4. Restore Original Range
        # -----------------------------------------------------------
        img_restored = augmented_norm * range_span + v_min

        if is_channel_first:
            img_restored = img_restored.transpose(2, 0, 1)  # (H, W, 1) -> (1, H, W)
        elif len(original_shape) == 2:
            img_restored = img_restored[:, :, 0]  # (H, W, 1) -> (H, W)

        return img_restored

    # --- Apply to Sample ---

    # Apply to Images
    if 'image0' in sample:
        m0 = sample.get('mask0')
        if hasattr(m0, 'numpy'): m0 = m0.numpy()
        # Use the switched pipeline
        sample['image0'] = _augment_robust(sample['image0'], mask=m0, pipeline=image_pipeline)

    if 'image1' in sample:
        m1 = sample.get('mask1')
        if hasattr(m1, 'numpy'): m1 = m1.numpy()
        # Use the switched pipeline
        sample['image1'] = _augment_robust(sample['image1'], mask=m1, pipeline=image_pipeline)

    # Physics always uses standard pipeline (never blackout physics)
    if 'phys0' in sample:
        m0 = sample.get('mask0')
        if hasattr(m0, 'numpy'): m0 = m0.numpy()
        sample['phys0'] = _augment_robust(sample['phys0'], mask=m0, pipeline=phys_pipeline)

    if 'phys1' in sample:
        m1 = sample.get('mask1')
        if hasattr(m1, 'numpy'): m1 = m1.numpy()
        sample['phys1'] = _augment_robust(sample['phys1'], mask=m1, pipeline=phys_pipeline)

    return sample


def to_tensor(sample_dict):
    sample_dict['image0'] = np.expand_dims(sample_dict['image0'], axis=0)
    sample_dict['image1'] = np.expand_dims(sample_dict['image1'], axis=0)
    sample_dict['phys0']  = np.expand_dims(sample_dict['phys0'], axis=0)
    sample_dict['phys1']  = np.expand_dims(sample_dict['phys1'], axis=0)

    for key, value in sample_dict.items():
        if isinstance(value, np.ndarray):
            # If it's a string array
            if np.issubdtype(value.dtype, np.str_):
                # Convert string arrays to a list of strings
                sample_dict[key] = tuple(value)
                continue

            if value.dtype.kind in ('O', 'S', 'U'):
                # This is an array of paths or strings, not a tensor.
                # Keep it as a tuple (as you did for strings).
                sample_dict[key] = tuple(value)
                continue

            # Only copy if necessary
            if not value.flags.writeable:
                value = value.copy()

            sample_dict[key] = torch.from_numpy(value).float()

        elif isinstance(value, list):
            sample_dict[key] = torch.tensor(value, dtype=torch.float32)

        elif isinstance(value, (int, float)):
            sample_dict[key] = torch.tensor(value, dtype=torch.float32)

    return sample_dict





def parse_sample_dict(raw_sample_dict):
    """
    Convert certain fields in a dictionary from string representations
    into Python lists/arrays/ints, etc.
    """
    parsed = {}

    for key, value in raw_sample_dict.items():
        # If the value is a string that looks like a Python list or dict,
        # we can safely parse it using ast.literal_eval.
        if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
            try:
                parsed_value = ast.literal_eval(value)
                # If you want a NumPy array:
                if isinstance(parsed_value, list):
                    parsed_value = np.array(parsed_value)
                parsed[key] = parsed_value
            except (SyntaxError, ValueError):
                # If parsing fails, fall back to the original string
                parsed[key] = value
        else:
            # If it's not a list-like string, you might still parse ints or floats
            # For example, if you know certain keys should be integers:
            if key in ("some_int_field", "other_int_field") and isinstance(value, str):
                try:
                    parsed[key] = int(value)
                except ValueError:
                    parsed[key] = value
            else:
                parsed[key] = value
    return parsed

def load_csv(config, dataset_type):
    image_shape = config.image_shape[0]
    # Build the path to the CSV file
    csv_path = Path(config.data_path) / 'color' / f"{dataset_type}_{image_shape}.csv"
    # Check if the CSV file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        # Convert DataFrame records into a list of dictionaries
        # records = df.to_dict(orient="records")
        print(f"Successfully loaded {len(df)} records from CSV file {csv_path}.")
    except Exception as e:
        raise IOError(f"Failed to load CSV file: {csv_path}: {e}")
    return df







