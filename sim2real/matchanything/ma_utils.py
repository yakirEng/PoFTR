import torch
import numpy as np
import torchvision.transforms as T

from sim2real.general_helpers import plot_images

def safe_center_crop_numpy(img, crop_size=256):
    """Safely pads the image if it's too small, then crops."""
    h, w = img.shape
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)

    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='reflect')

    h, w = img.shape
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

def robust_normalize_tau2(img_tensor):
    val_min = torch.quantile(img_tensor, 0.01)
    val_max = torch.quantile(img_tensor, 0.99)
    img_tensor = torch.clamp(img_tensor, min=val_min, max=val_max)
    denom = val_max - val_min
    if denom < 1e-6:
        denom = 1.0
    return (img_tensor - val_min) / denom

def prepare_matchanything_batch(img0_np, img1_np, crop_size=256, coarse_scale=8):
    # FIX: Removed redundant safe_center_crop_numpy calls here.
    # Cropping is already done in the main loop before calling this function.

    # 1. Tensor & Normalize
    img0_t = robust_normalize_tau2(torch.from_numpy(img0_np).float().unsqueeze(0))
    img1_t = robust_normalize_tau2(torch.from_numpy(img1_np).float().unsqueeze(0))

    # 2. ImageNet format (1, 3, H, W)
    img0_t = img0_t.repeat(3, 1, 1).unsqueeze(0)
    img1_t = img1_t.repeat(3, 1, 1).unsqueeze(0)

    # 3. Strict Mask Shapes
    h, w = crop_size, crop_size
    pixel_mask0 = torch.ones((1, h, w), dtype=torch.bool)
    pixel_mask1 = torch.ones((1, h, w), dtype=torch.bool)
    mask0 = torch.ones((1, h // coarse_scale, w // coarse_scale), dtype=torch.bool)
    mask1 = torch.ones((1, h // coarse_scale, w // coarse_scale), dtype=torch.bool)

    batch = {
        'image0': img0_t,
        'image1': img1_t,
        'mask0': mask0,
        'mask1': mask1,
        'pixel_mask0': pixel_mask0,
        'pixel_mask1': pixel_mask1
    }
    return batch

def run_matchanything(batch, device, wl0, wl1, model):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(batch)

    kpts0 = outputs['mkpts0_f'].view(-1, 2).cpu().numpy()
    kpts1 = outputs['mkpts1_f'].view(-1, 2).cpu().numpy()
    conf  = outputs['mconf'].view(-1).cpu().numpy()

    print(f"MatchAnything found {len(kpts0)} putative matches.")
    return kpts0, kpts1, conf