import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def format_image_for_plot(img):
    """Safely scales a 2D numpy array to 0-255 uint8 for visualization."""
    # Handle torch tensors if passed by mistake
    if hasattr(img, 'cpu'):
        img = img.cpu().numpy()

    img = np.squeeze(img)  # Remove any batch/channel dims (1, 256, 256) -> (256, 256)
    v_min, v_max = img.min(), img.max()
    denom = v_max - v_min if v_max - v_min > 1e-6 else 1.0
    img_norm = (img - v_min) / denom
    return (np.clip(img_norm, 0.0, 1.0) * 255).astype(np.uint8)

def plot_sim2real_matches(img0, img1, kpts0, kpts1, model_name, save_path):
    """
    Computes RANSAC inliers/outliers and saves two versions to disk:
    1. Clean: Just the images and match lines (OpenCV).
    2. Labeled: Includes title with inlier stats (Matplotlib).
    """
    if save_path is None:
        return

    # 1. Format images to uint8 grayscale
    img0_vis = format_image_for_plot(img0)
    img1_vis = format_image_for_plot(img1)

    # 2. Compute RANSAC inliers/outliers
    if len(kpts0) >= 4:
        # 3.0 pixel threshold matches the AeroSync dataset evaluation protocol
        H, inlier_mask = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, 3.0)
        if inlier_mask is not None:
            inlier_mask = inlier_mask.ravel().astype(bool)
        else:
            inlier_mask = np.zeros(len(kpts0), dtype=bool)
    else:
        inlier_mask = np.zeros(len(kpts0), dtype=bool)

    # 3. Create Canvas
    h0, w0 = img0_vis.shape
    h1, w1 = img1_vis.shape
    height = max(h0, h1)
    width  = w0 + w1
    output_img = np.zeros((height, width), dtype=np.uint8)
    output_img[:h0, :w0]       = img0_vis
    output_img[:h1, w0:w0+w1]  = img1_vis
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)

    # 4. Shift keypoints for target image
    kpts1_shifted = kpts1.copy()
    kpts1_shifted[:, 0] += w0

    # 5. Draw Matches (outliers Red, inliers Green)
    for is_inlier in [False, True]:
        for i in range(len(kpts0)):
            if inlier_mask[i] == is_inlier:
                pt1   = tuple(kpts0[i].astype(int))
                pt2   = tuple(kpts1_shifted[i].astype(int))
                color = (0, 255, 0) if is_inlier else (0, 0, 255)
                cv2.line(output_img, pt1, pt2, color, 1, cv2.LINE_AA)
                cv2.circle(output_img, pt1, 2, color, -1, cv2.LINE_AA)
                cv2.circle(output_img, pt2, 2, color, -1, cv2.LINE_AA)

    # 6. Save Version 1: Clean image (OpenCV)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path = save_path.parent / (save_path.stem + "_clean" + save_path.suffix)
    cv2.imwrite(str(clean_path), output_img)

    # 7. Save Version 2: With metadata title (Matplotlib)
    # Using a local figure to avoid polluting global state
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(output_rgb)

    inlier_count = inlier_mask.sum()
    total_count = len(kpts0)
    percentage = 100 * inlier_count / max(total_count, 1)

    ax.set_title(
        f"{model_name} | Inliers: {inlier_count} / {total_count} ({percentage:.1f}%)",
        fontsize=12
    )
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)  # Crucial to free memory during large batch runs


def center_crop(img_tensor, crop_size=256):
    """Expects a tensor of shape (..., H, W)"""
    h, w = img_tensor.shape[-2:]
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    return img_tensor[..., start_y:start_y+crop_size, start_x:start_x+crop_size]

def plot_images(image0, image1, wl0, wl1):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axs[0].imshow(image0, cmap='gray')
    axs[0].set_title(f'{wl0}')
    fig.colorbar(im0, ax=axs[0], orientation='vertical')

    im1 = axs[1].imshow(image1, cmap='gray')
    axs[1].set_title(f'{wl1}')
    fig.colorbar(im1, ax=axs[1], orientation='vertical')

    plt.tight_layout()
    plt.show()

def compute_ransac_metrics(kpts0, kpts1, ransac_thresh=3.0, min_matches=8):
    """Computes RANSAC inlier count and inlier ratio."""
    if len(kpts0) < min_matches:
        return 0, 0.0
    try:
        _, mask = cv2.findFundamentalMat(
            kpts0.astype(np.float32),
            kpts1.astype(np.float32),
            cv2.RANSAC,
            ransac_thresh
        )
        if mask is None:
            return 0, 0.0
        n_inliers = int(mask.sum())
        inlier_ratio = n_inliers / len(kpts0)
        return n_inliers, inlier_ratio
    except Exception:
        return 0, 0.0


def load_coefficients(coeff_path, wl0, wl1):
    """Loads precomputed coefficients for the given wavelength pair."""
    coeff_path = Path(coeff_path)
    coeff_filename = f"coefficients_{wl0}_{wl1}.npz"
    if not coeff_path.exists():
        raise FileNotFoundError(f"Coefficient file not found: {coeff_path}")
    coeffs = np.load(coeff_path / coeff_filename)
    return coeffs

def generate_prior_maps(img0, img1, wl0, wl1, phys_model, fpa0, fpa1):
    prior0 = phys_model.predict(img0, t_fpa=fpa0, direction='rad_to_temp', band=wl0)
    prior1 = phys_model.predict(img1, t_fpa=fpa1, direction='rad_to_temp', band=wl1)
    return prior0, prior1




def build_inference_sample(img0, img1, prior0=None, prior1=None, use_phys=True, coarse_scale=8):
    """
    img0: 9um image tensor (1, 256, 256)
    img1: pan image tensor (1, 256, 256)
    prior0/1: the generated prior maps (1, 256, 256)
    """
    # 1. Real images don't have warp artifacts, so the valid pixel mask is all 1s.
    b, h, w = img0.shape
    pixel_mask0 = torch.ones((1, h, w), dtype=torch.bool)
    pixel_mask1 = torch.ones((1, h, w), dtype=torch.bool)

    # 2. Downsample masks for the transformer coarse level
    mask0 = F.interpolate(pixel_mask0.unsqueeze(0).float(), scale_factor=1/coarse_scale, mode='nearest')[0].bool()
    mask1 = F.interpolate(pixel_mask1.unsqueeze(0).float(), scale_factor=1/coarse_scale, mode='nearest')[0].bool()

    sample = {
        'mask0': mask0,
        'mask1': mask1,
        'pixel_mask0': pixel_mask0,
        'pixel_mask1': pixel_mask1
    }

    # 3. Format the inputs based on the model type
    if use_phys:
        # For PoFTR: Concat [Image, Prior, Mask]
        sample['image0'] = torch.cat([img0, prior0, pixel_mask0.float()], dim=0).unsqueeze(0) # Add batch dim
        sample['image1'] = torch.cat([img1, prior1, pixel_mask1.float()], dim=0).unsqueeze(0)
    else:
        # For XoFTR / MatchAnything: Just the image
        sample['image0'] = img0.unsqueeze(0)
        sample['image1'] = img1.unsqueeze(0)

    return sample

import json
from pathlib import Path

STATS_BASE = Path("data/stats")

def load_stats(wl0: str, wl1: str, stats_base: Path = STATS_BASE) -> dict:
    """Load global dataset stats for a given wl pair.

    Looks for:
        {stats_base}/{wl0}_{wl1}/stats.json
    e.g.: truncnorm/9um_pan/stats.json

    Args:
        wl0:        first wavelength string  (e.g. '9um', '11um')
        wl1:        second wavelength string (e.g. 'pan', '11um')
        stats_base: root folder containing per-pair subdirectories

    Returns:
        dict with keys: 'image0', 'image1', 'phys0', 'phys1'
        each containing 'mean', 'std', 'count'
    """
    stats_path = stats_base / f"{wl0}_{wl1}" / "stats.json"

    if not stats_path.exists():
        # Try reversed order (wl1_wl0) as fallback
        stats_path_rev = stats_base / f"{wl1}_{wl0}" / "stats.json"
        if stats_path_rev.exists():
            print(f"Warning: found stats under reversed pair '{wl1}_{wl0}', loading that.")
            stats_path = stats_path_rev
        else:
            raise FileNotFoundError(
                f"Stats file not found for pair '{wl0}_{wl1}'.\n"
                f"Tried:\n  {stats_base / f'{wl0}_{wl1}' / 'stats.json'}"
                f"\n  {stats_base / f'{wl1}_{wl0}' / 'stats.json'}"
            )

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    print(f"Loaded stats from: {stats_path}")
    return stats