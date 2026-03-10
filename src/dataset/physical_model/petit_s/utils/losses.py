import torch
from pytorch_msssim import ssim, ms_ssim
from torch import Tensor
import torch.nn.functional as F


def validate_shape(pred: torch.Tensor, target: torch.Tensor):
    if pred.shape != target.shape:
        if pred.ndim == 4 and target.ndim == 3:
            target = target.unsqueeze(1)
        elif pred.ndim == 3 and target.ndim == 4:
            pred = pred.unsqueeze(1)
        else:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    return pred, target

def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (Celsius). pred/target: [B,1,H,W] or [B,H,W]."""
    pred, target = validate_shape(pred, target)
    pred = pred.to(dtype=target.dtype, device=target.device)
    return torch.mean(torch.abs(pred - target))

def ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> torch.Tensor:
    """Structural Similarity Index Measure (SSIM) loss. pred/target: [B,1,H,W] or [B,H,W]."""
    pred, target = validate_shape(pred, target)
    pred = pred.to(dtype=target.dtype, device=target.device)
    ssim_index = ssim(pred, target, data_range=data_range, size_average=True)
    return 1.0 - ssim_index  # Convert SSIM to a loss (1 - SSIM)

def ms_ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float= 6.0) -> torch.Tensor:
    """Multi-Scale SSIM loss."""
    k = data_range / 2.0
    pred, target = validate_shape(pred, target)
    pred = pred.to(dtype=target.dtype, device=target.device)
    pred_c = torch.clamp(pred, min=-k, max=k)
    target_c = torch.clamp(target, min=-k, max=k)
    return 1.0 - ms_ssim(pred_c, target_c, data_range=data_range, size_average=True)

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes L1 loss on the Sobel gradients (edges) of the images.
    Forces the model to produce sharp boundaries.
    """
    pred, target = validate_shape(pred, target)
    pred = pred.to(dtype=target.dtype, device=target.device)

    # Standard Sobel kernels for edge detection
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    # Compute spatial gradients (padding=1 ensures output size matches input)
    pred_dx = F.conv2d(pred, kernel_x, padding=1)
    pred_dy = F.conv2d(pred, kernel_y, padding=1)
    target_dx = F.conv2d(target, kernel_x, padding=1)
    target_dy = F.conv2d(target, kernel_y, padding=1)

    # L1 loss on the gradients (penalize blurry or misplaced edges)
    loss_dx = torch.abs(pred_dx - target_dx).mean()
    loss_dy = torch.abs(pred_dy - target_dy).mean()

    return loss_dx + loss_dy

def compute_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 6, alpha: float = 0.1, beta=0.05) -> dict[
    str, Tensor]:
    """Compute loss based on the specified loss type."""
    loss_mae = mae_loss(pred, target)
    loss_ms_ssim = ms_ssim_loss(pred, target, data_range)
    loss_grad = gradient_loss(pred, target)
    loss = (1-alpha) * loss_mae + alpha * loss_ms_ssim + beta * loss_grad
    return {
        "mae_loss": loss_mae,
        "ms_ssim_loss": loss_ms_ssim,
        "gradient_loss": loss_grad,
        "loss": loss
    }
