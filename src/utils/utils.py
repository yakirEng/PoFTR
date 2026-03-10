import numpy as np

def center_crop(image: np.ndarray, crop_size: tuple) -> np.ndarray:
    """Crops the image around the center to the desired size (no padding)."""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Calculate the center of the image
    center_h, center_w = h // 2, w // 2

    # Calculate the cropping area
    crop_top = center_h - crop_h // 2
    crop_bottom = center_h + crop_h // 2
    crop_left = center_w - crop_w // 2
    crop_right = center_w + crop_w // 2

    # Crop the image
    return image[crop_top:crop_bottom, crop_left:crop_right]