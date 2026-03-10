import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MaskAwareResize(nn.Module):
    """Resize with validity-aware interpolation."""

    def resize_with_mask(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor,
            target_size: Tuple[int, int],
            mode: str = 'bilinear'
    ) -> torch.Tensor:
        """
        FIX: Safe division (no temporary spikes before where clause).
        """
        tensor_masked = tensor * mask

        align_corners = True if mode == 'bilinear' else None
        numerator = F.interpolate(tensor_masked, size=target_size, mode=mode, align_corners=align_corners)
        denominator = F.interpolate(mask, size=target_size, mode=mode, align_corners=align_corners)

        # FIX: Initialize output first, then fill valid regions
        output = torch.zeros_like(numerator)
        valid_mask = denominator > 1e-3
        output = torch.where(valid_mask, numerator / denominator.clamp(min=1e-8), output)

        return output

    def resize_mask(
            self,
            mask: torch.Tensor,
            target_size: Tuple[int, int],
            threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        FIX: Adaptive threshold now works correctly.
        """
        mask_soft = F.interpolate(mask.float(), size=target_size, mode='bilinear', align_corners=True)

        if threshold is None:
            # FIX: Was always 0.4, now properly adaptive
            min_dim = min(target_size)
            if min_dim <= 64:
                threshold = 0.2  # Coarse: permissive
            elif min_dim <= 128:
                threshold = 0.3  # Medium
            else:
                threshold = 0.4  # Fine: strict

        return (mask_soft > threshold).float()
