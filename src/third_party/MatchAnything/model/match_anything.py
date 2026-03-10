import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForKeypointMatching

class MatchAnythingHFWrapper(nn.Module):
    def __init__(self, model_id="zju-community/matchanything_eloftr", device='cuda'):
        super().__init__()
        self.device = device
        print(f"Loading MatchAnything ({model_id})...")
        self.model = AutoModelForKeypointMatching.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # CRITICAL: The model's RoPE embedding is fixed for 26x26 patches.
        # 26 * 32 (stride) = 832.
        # We must resize inputs to 832x832 to match the cached embeddings.
        self.inference_size = (832, 832)

    def filter_by_mask(self, kpts0, kpts1, scores, mask0, mask1):
        """
        Filters matches that fall into invalid (masked) regions.
        """
        if mask0 is None or mask1 is None:
            return kpts0, kpts1, scores

        # Ensure masks are boolean and 2D
        mask0 = mask0.bool().squeeze()
        mask1 = mask1.bool().squeeze()

        # Handle case where batch dim might be squeezed out incorrectly
        if mask0.ndim > 2: mask0 = mask0[0]
        if mask1.ndim > 2: mask1 = mask1[0]

        h, w = mask0.shape[-2:]

        # Clamp coordinates to avoid index errors
        x0 = kpts0[:, 0].long().clamp(0, w - 1)
        y0 = kpts0[:, 1].long().clamp(0, h - 1)
        x1 = kpts1[:, 0].long().clamp(0, w - 1)
        y1 = kpts1[:, 1].long().clamp(0, h - 1)

        # Check validity in both masks
        valid_0 = mask0[y0, x0]
        valid_1 = mask1[y1, x1]
        keep = valid_0 & valid_1

        return kpts0[keep], kpts1[keep], scores[keep]

    def forward(self, batch):
        # 1. Get Pre-processed Images from DataLoader
        #    (Already Robust Normalized + ImageNet Normalized)
        img0 = batch['image0'].to(self.device)
        img1 = batch['image1'].to(self.device)

        # 2. Capture Original Dimensions for Rescaling
        B, C, H_orig, W_orig = img0.shape

        # 3. Resize to Model Input Size (832x832)
        #    We must resize because the model has fixed positional embeddings.
        img0_resized = F.interpolate(img0, size=self.inference_size, mode='bilinear', align_corners=False)
        img1_resized = F.interpolate(img1, size=self.inference_size, mode='bilinear', align_corners=False)

        # 4. Stack for Model Input (B, 2, 3, H, W)
        pixel_values = torch.stack([img0_resized, img1_resized], dim=1)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        # 5. Parse Output (Batch Size 1 assumption)
        matches0 = outputs.matches[0, 0]
        scores0 = outputs.matching_scores[0, 0]
        grid0 = outputs.keypoints[0, 0]
        grid1 = outputs.keypoints[0, 1]

        # 6. Filter Valid Matches (matches0 > -1 indicates a match found)
        valid_mask = matches0 > -1

        kpts0_norm = grid0[valid_mask]
        scores = scores0[valid_mask]

        valid_indices_in_img1 = matches0[valid_mask].long()
        kpts1_norm = grid1[valid_indices_in_img1]

        # 7. Denormalize to Original Resolution
        #    The model outputs normalized coordinates [0, 1].
        #    We scale them back to the original image dimensions (W_orig, H_orig).
        kpts0 = torch.zeros_like(kpts0_norm)
        kpts1 = torch.zeros_like(kpts1_norm)

        kpts0[:, 0] = kpts0_norm[:, 0] * W_orig
        kpts0[:, 1] = kpts0_norm[:, 1] * H_orig

        kpts1[:, 0] = kpts1_norm[:, 0] * W_orig
        kpts1[:, 1] = kpts1_norm[:, 1] * H_orig

        # 8. Filter using High-Res Masks (Optional but Recommended)
        if 'pixel_mask0' in batch:
            kpts0, kpts1, scores = self.filter_by_mask(
                kpts0, kpts1, scores,
                batch['pixel_mask0'].to(self.device),
                batch['pixel_mask1'].to(self.device)
            )

        return {
            'mkpts0_f': kpts0.unsqueeze(0),
            'mkpts1_f': kpts1.unsqueeze(0),
            'mconf': scores.unsqueeze(0)
        }