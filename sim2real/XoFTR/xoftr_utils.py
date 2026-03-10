import torch
import numpy as np


from src.configs.poftr_configs import get_config
from src.utils.misc import lower_config
from src.third_party import LoFTR, ASpanFormer
from src.PoFTR.backbone import build_phys_backbone
from src.third_party.XoFTR.src.xoftr import XoFTR

# ==========================================
# 1. MODEL WRAPPER
# ==========================================
import torch.nn as nn

class PoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        poftr_config = self.config['poftr']
        base_model = poftr_config['proj']['base_model']
        phys_cfg = poftr_config['phys']

        # pick model
        if base_model == 'loftr':
            model = LoFTR(self.config['method']['loftr'])
        elif base_model == 'aspanformer':
            model = ASpanFormer(self.config['method']['aspan'])
        elif base_model == 'xoftr':
            model = XoFTR(self.config['method']['xoftr'])
        else:
            raise ValueError(f"Unknown base_model: {base_model}")

        # add physics (same backbone for both models)
        if phys_cfg['use_phys']:
            phys_backbone = build_phys_backbone(poftr_config)
            model.backbone = phys_backbone

        return model

    def forward(self, data):
        return self.model(data)

# ==========================================
# 2. DATA PREPARATION HELPERS
# ==========================================
def safe_center_crop_numpy(img, crop_size=256):
    """Safely pads the image if it's too small, then crops."""
    h, w = img.shape
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)

    # Pad symmetrically using reflection to maintain edge continuity
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='reflect')

    h, w = img.shape
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

def normalize_with_stats(tensor, mean, std):
    """Applies your dataset's specific global z-score normalization."""
    return (tensor - mean) / std

def prepare_xoftr_batch(img0_np, img1_np, global_stats, crop_size=256, coarse_scale=8):
    """Prepares the concatenated [Image, Prior, Mask] input for PoFTR."""
    # 1. Center Crop Images and Priors identically
    i0 = safe_center_crop_numpy(img0_np, crop_size)
    i1 = safe_center_crop_numpy(img1_np, crop_size)


    # 2. Convert to Tensors (1, H, W)
    i0_t = torch.from_numpy(i0).float().unsqueeze(0).unsqueeze(0)
    i1_t = torch.from_numpy(i1).float().unsqueeze(0).unsqueeze(0)


    # 3. Apply Global Normalization (Using nested keys from stats.json)
    i0_t = normalize_with_stats(i0_t, global_stats['image0']['mean'], global_stats['image0']['std'])
    i1_t = normalize_with_stats(i1_t, global_stats['image1']['mean'], global_stats['image1']['std'])


    # 4. Create Masks
    pixel_mask0 = torch.ones((1, crop_size, crop_size), dtype=torch.bool)
    pixel_mask1 = torch.ones((1, crop_size, crop_size), dtype=torch.bool)

    mask0 = torch.ones((1, crop_size // coarse_scale, crop_size // coarse_scale), dtype=torch.bool)
    mask1 = torch.ones((1, crop_size // coarse_scale, crop_size // coarse_scale), dtype=torch.bool)


    # 6. Assemble Batch
    batch = {
        'image0': i0_t,
        'image1': i1_t,
        'mask0': mask0,
        'mask1': mask1,
        'pixel_mask0': pixel_mask0,
        'pixel_mask1': pixel_mask1
    }
    return batch


def load_xoftr_model(config, checkpoint_path, device='cuda'):
    """Load and return the XoFTR model once. Pass the returned model to run_xoftr."""
    print(f"Loading PoFTR wrapper with base model: {config['poftr']['proj']['base_model']}...")
    config['poftr']['phys']['use_phys'] = False  # XoFTR has no physics backbone

    model = PoFTR(config).to(device)

    print(f"Loading fine-tuned weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['state_dict']
    clean_state_dict = {
        (k.replace("model.model.", "model.") if k.startswith("model.model.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    print("XoFTR model ready.")
    return model


def run_xoftr(batch, model, device='cuda'):
    """Run inference with a pre-loaded XoFTR model."""
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        model(batch)

    kpts0 = batch['mkpts0_f'].cpu().numpy()
    kpts1 = batch['mkpts1_f'].cpu().numpy()
    conf  = batch['mconf'].cpu().numpy()

    print(f"XoFTR found {len(kpts0)} putative matches.")
    return kpts0, kpts1, conf

