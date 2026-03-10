import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, boundary_value=-1.0):
        super().__init__()
        self.boundary_value = boundary_value

    def _normalize_valid_pixels(self, x, valid_mask):
        """Normalize tensor using only valid pixel statistics"""
        if valid_mask.sum() == 0:
            return torch.zeros_like(x)

        # Get min/max from valid pixels only
        valid_pixels = x[valid_mask]
        x_min = valid_pixels.min()
        x_max = valid_pixels.max()

        # Normalize entire tensor using valid pixel stats
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        else:
            return torch.zeros_like(x)

    def forward(self, image, valid_mask, phys=None):
        # Normalize image
        image_norm = self._normalize_valid_pixels(image, valid_mask)
        image_norm = torch.where(~valid_mask, self.boundary_value, image_norm)

        # Normalize physics if provided
        phys_norm = None
        if phys is not None:
            phys_norm = self._normalize_valid_pixels(phys, valid_mask)
            phys_norm = torch.where(~valid_mask, self.boundary_value, phys_norm)

        return image_norm, phys_norm


