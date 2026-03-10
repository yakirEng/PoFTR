import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio


class ThermalMetrics:
    def __init__(self, device):
        """
        Wrapper for all thermal estimation metrics.
        Lazy initialization of heavy models (LPIPS) is handled by torchmetrics moving to device.
        """
        self.device = device

        # 1. Perceptual Metric (Deep Features)
        # 'net_type=vgg' is standard for assessing perceptual quality
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

        # 2. Standard Signal Metric
        # data_range=6.0 matches your Z-score normalized data (approx [-3, 3])
        self.psnr = PeakSignalNoiseRatio(data_range=6.0).to(device)

    def compute_lpips(self, pred, target):
        # Clamp to expected physics range first to avoid outliers blowing up scaling
        pred_c = torch.clamp(pred, -3.0, 3.0)
        target_c = torch.clamp(target, -3.0, 3.0)

        # Scale to [0, 1]
        pred_01 = (pred_c + 3.0) / 6.0
        target_01 = (target_c + 3.0) / 6.0

        # Replicate 1 channel -> 3 channels
        pred_norm = pred_01.repeat(1, 3, 1, 1)
        target_norm = target_01.repeat(1, 3, 1, 1)

        return self.lpips(pred_norm, target_norm)


    def compute_gradient_mae(self, pred, target):
        """
        Computes L1 error of spatial gradients (Sobel).
        Measures how well 'edges' are preserved, which is critical for image matching.
        """
        # Define Sobel kernels
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).view(1, 1, 3, 3).float()
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).view(1, 1, 3, 3).float()

        # Compute gradients
        pred_dx = F.conv2d(pred, k_x, padding=1)
        pred_dy = F.conv2d(pred, k_y, padding=1)
        gt_dx = F.conv2d(target, k_x, padding=1)
        gt_dy = F.conv2d(target, k_y, padding=1)

        # L1 distance between gradient maps
        grad_mae = torch.abs(pred_dx - gt_dx).mean() + torch.abs(pred_dy - gt_dy).mean()
        return grad_mae

    def compute_all(self, pred, target):
        """
        Computes full suite of metrics.
        Expects inputs: [B, 1, H, W]
        """
        # Ensure correct shape
        if pred.ndim == 3: pred = pred.unsqueeze(1)
        if target.ndim == 3: target = target.unsqueeze(1)

        # 1. Standard Pixel Metrics
        mae = torch.abs(pred - target).mean()
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        psnr_val = self.psnr(pred, target)

        # 2. Edge Metric (Sharpness)
        grad_mae = self.compute_gradient_mae(pred, target)

        # 3. Perceptual Metric (LPIPS)
        lpips_val = self.compute_lpips(pred, target)

        return {
            "mae_c": mae.item(),
            "rmse_c": rmse.item(),
            "grad_mae": grad_mae.item(),
            "psnr": psnr_val.item(),
            "lpips": lpips_val.item()
        }