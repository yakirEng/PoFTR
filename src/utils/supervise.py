import torch
from einops import repeat
from loguru import logger
from kornia.utils import create_meshgrid
import matplotlib.pyplot as plt
import numpy as np

from src.utils.geometry import warp_kpts


##############  ↓  Coarse-Level supervision  ↓  ##############
@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt

@torch.no_grad()
def supervise_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['data']['coarse_scale']
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)  # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    # loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    loop_back = torch.gather(nearest_index0, 1, nearest_index1)
    correct_0to1 = loop_back == torch.arange(h0 * w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })


##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def supervise_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['data']['fine_scale']
    radius = config['data']['fine_window_size'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})





def visualize_coarse_supervision(data, config, save_path=None):
    """
    Visualize the coarse supervision results including:
    - Grid mapping from image0 to image1
    - Valid matches after mask filtering
    - Confidence matrix heatmap

    Args:
        data: Dictionary containing supervision results from supervise_coarse()
        config: Configuration dictionary
        save_path: Optional path to save the figure
    """
    # Extract data
    N = data['image0'].shape[0]
    batch_idx = 0  # Visualize first item in batch

    img0 = data['image0'][batch_idx].cpu().permute(1, 2, 0).numpy()
    img1 = data['image1'][batch_idx].cpu().permute(1, 2, 0).numpy()

    # Normalize images for display
    img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 1e-8)
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)

    # Get coarse-level dimensions
    scale = config['data']['coarse_scale']
    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]
    h0, w0 = H0 // scale, W0 // scale
    h1, w1 = H1 // scale, W1 // scale

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ========== 1. Original Images with Grid Overlay ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img0)
    ax1.set_title('Image 0 with Coarse Grid', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Draw coarse grid on image0
    for i in range(0, H0, scale):
        ax1.axhline(y=i, color='cyan', alpha=0.3, linewidth=0.5)
    for j in range(0, W0, scale):
        ax1.axvline(x=j, color='cyan', alpha=0.3, linewidth=0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img1)
    ax2.set_title('Image 1 with Coarse Grid', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Draw coarse grid on image1
    for i in range(0, H1, scale):
        ax2.axhline(y=i, color='cyan', alpha=0.3, linewidth=0.5)
    for j in range(0, W1, scale):
        ax2.axvline(x=j, color='cyan', alpha=0.3, linewidth=0.5)

    # ========== 2. Warped Points Visualization ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img1)
    ax3.set_title('Warped Points (img0→img1)', fontsize=12, fontweight='bold')

    # Get warped points for this batch item
    w_pt0_i = data['spv_w_pt0_i'][batch_idx].cpu().numpy()

    # Sample points to avoid cluttering
    sample_rate = max(1, len(w_pt0_i) // 500)
    sampled_pts = w_pt0_i[::sample_rate]

    ax3.scatter(sampled_pts[:, 0], sampled_pts[:, 1],
                c='red', s=5, alpha=0.5, marker='x')
    ax3.axis('off')

    # ========== 3. Valid Matches (after mask filtering) ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img0)
    ax4.set_title('Valid Match Points in Image 0', fontsize=12, fontweight='bold')

    # Get valid matches for this batch
    mask = data['spv_b_ids'] == batch_idx
    i_ids = data['spv_i_ids'][mask].cpu().numpy()

    # Convert linear indices to 2D coordinates
    grid_pt0_i = data['spv_w_pt0_i'][batch_idx].cpu().numpy()
    valid_pt0 = grid_pt0_i[i_ids]

    # Reconstruct grid points for image0
    scale0 = scale * data['scale0'][batch_idx].cpu().item() if 'scale0' in data else scale
    y_coords = (i_ids // w0) * scale0
    x_coords = (i_ids % w0) * scale0

    ax4.scatter(x_coords, y_coords, c='lime', s=20, alpha=0.7, edgecolors='darkgreen')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(img1)
    ax5.set_title('Valid Match Points in Image 1', fontsize=12, fontweight='bold')

    # Get corresponding points in image1
    j_ids = data['spv_j_ids'][mask].cpu().numpy()
    scale1 = scale * data['scale1'][batch_idx].cpu().item() if 'scale1' in data else scale
    y_coords_1 = (j_ids // w1) * scale1
    x_coords_1 = (j_ids % w1) * scale1

    ax5.scatter(x_coords_1, y_coords_1, c='lime', s=20, alpha=0.7, edgecolors='darkgreen')
    ax5.axis('off')

    # ========== 4. Match Lines ==========
    ax6 = fig.add_subplot(gs[1, 2])

    # Create side-by-side image
    combined = np.zeros((max(H0, H1), W0 + W1, 3))
    combined[:H0, :W0] = img0
    combined[:H1, W0:] = img1
    ax6.imshow(combined)
    ax6.set_title(f'Match Correspondences (Total: {len(i_ids)})',
                  fontsize=12, fontweight='bold')

    # Draw match lines (sample to avoid clutter)
    n_samples = min(100, len(i_ids))
    sample_indices = np.random.choice(len(i_ids), n_samples, replace=False)

    for idx in sample_indices:
        x0, y0 = x_coords[idx], y_coords[idx]
        x1, y1 = x_coords_1[idx] + W0, y_coords_1[idx]
        ax6.plot([x0, x1], [y0, y1], 'y-', linewidth=0.5, alpha=0.6)

    ax6.axis('off')

    # ========== 5. Confidence Matrix Heatmap ==========
    ax7 = fig.add_subplot(gs[2, :])

    conf_matrix = data['conf_matrix_gt'][batch_idx].cpu().numpy()
    conf_matrix_2d = conf_matrix.reshape(h0 * w0, h1 * w1)

    # Subsample for visualization if too large
    max_size = 200
    if h0 * w0 > max_size or h1 * w1 > max_size:
        stride0 = max(1, (h0 * w0) // max_size)
        stride1 = max(1, (h1 * w1) // max_size)
        conf_matrix_2d = conf_matrix_2d[::stride0, ::stride1]

    im = ax7.imshow(conf_matrix_2d, cmap='hot', aspect='auto', interpolation='nearest')
    ax7.set_title('Ground Truth Confidence Matrix', fontsize=12, fontweight='bold')
    ax7.set_xlabel(f'Image 1 Grid Points (coarse: {h1}x{w1}={h1 * w1})')
    ax7.set_ylabel(f'Image 0 Grid Points (coarse: {h0}x{w0}={h0 * w0})')
    plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)

    # ========== 6. Statistics ==========
    plt.figtext(0.5, 0.02,
                f'Statistics: Valid Matches={len(i_ids)} | '
                f'Grid Size: {h0}x{w0} ({h0 * w0} pts) × {h1}x{w1} ({h1 * w1} pts) | '
                f'Match Ratio: {100 * len(i_ids) / (h0 * w0):.2f}%',
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()

    # Print additional statistics
    print("\n" + "=" * 60)
    print("COARSE SUPERVISION STATISTICS")
    print("=" * 60)
    print(f"Batch size: {N}")
    print(f"Coarse scale: {scale}")
    print(f"Image 0: {H0}x{W0} → Coarse: {h0}x{w0}")
    print(f"Image 1: {H1}x{W1} → Coarse: {h1}x{w1}")
    print(f"Total valid matches: {len(data['spv_i_ids'])}")
    print(f"Matches in batch[{batch_idx}]: {mask.sum().item()}")
    if 'mask0' in data:
        print(f"Mask filtering: ENABLED")
    else:
        print(f"Mask filtering: DISABLED")
    print("=" * 60 + "\n")
