import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from src.PoFTR.backbone.backbone_utils import MaskAwareResize
from src.PoFTR.backbone.basic_block import BasicBlock
from src.PoFTR.modules.sft import SFT


class ResNetFPN_8_2_SFT(nn.Module):
    """
    ResNet+FPN with Physics-Informed SFT injection at three stages:
    1. Input Level (Raw signal modulation) - The "Sensor Fusion"
    2. Coarse Level (Stride 8) - Global semantic structure
    3. Fine Level (Stride 2) - Local geometric refinement
    """

    def __init__(self, config: dict):
        super().__init__()

        # Validate config
        backbone_cfg = config.get('backbone', {})
        phys_cfg = config.get('phys', {})
        sft_cfg = config.get('sft', {})

        if 'initial_dim' not in backbone_cfg or 'block_dims' not in backbone_cfg:
            raise ValueError("Config must have backbone.initial_dim and backbone.block_dims")

        initial_dim = backbone_cfg['initial_dim']
        block_dims = backbone_cfg['block_dims']  # e.g., [128, 196, 256]

        if len(block_dims) != 3:
            raise ValueError(f"block_dims must have 3 elements, got {len(block_dims)}")

        self.use_phys = phys_cfg["use_phys"]

        # Injection flags
        self.inject_input_sft = phys_cfg.get('inject_input_sft', True) and self.use_phys
        self.inject_coarse_sft = phys_cfg.get('inject_coarse_sft', True) and self.use_phys
        self.inject_fine_sft = phys_cfg.get('inject_fine_sft', True) and self.use_phys

        # Backbone
        self.in_planes = initial_dim
        self.conv1 = nn.Conv2d(1, initial_dim, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_dims[0], stride=1)
        self.layer2 = self._make_layer(block_dims[1], stride=2)
        self.layer3 = self._make_layer(block_dims[2], stride=2)

        # FPN Lateral Connections
        self.layer3_outconv = self._conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = self._conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            self._conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            self._conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = self._conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            self._conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            self._conv3x3(block_dims[1], block_dims[0]),
        )

        # Physics modules
        if self.use_phys:
            self.mask_resizer = MaskAwareResize()

            bottleneck_dim = sft_cfg.get('bottleneck_dim', 128)
            dropout_p = sft_cfg.get('dropout_p', 0.1)
            learnable_scale = sft_cfg.get('learnable_scale', True)
            initial_scale = sft_cfg.get('initial_scale', 0.1)

            # 1. Input SFT: Operates on 1-channel image
            if self.inject_input_sft:
                self.sft_input = SFT(c_feat=1, bottleneck_dim=bottleneck_dim, dropout_p=dropout_p,
                                     learnable_scale=learnable_scale, initial_scale=initial_scale)

            # 2. Coarse SFT: Operates on block_dims[2] (e.g., 256)
            if self.inject_coarse_sft:
                self.sft_coarse = SFT(c_feat=block_dims[2], bottleneck_dim=bottleneck_dim, dropout_p=dropout_p,
                                      learnable_scale=learnable_scale, initial_scale=initial_scale)

            # 3. Fine SFT: Operates on Lateral Output (block_dims[1])
            # FIX: Matches the channel count AFTER layer1_outconv projection (e.g. 196)
            if self.inject_fine_sft:
                self.sft_fine = SFT(c_feat=block_dims[0], bottleneck_dim=bottleneck_dim, dropout_p=dropout_p,
                                    learnable_scale=learnable_scale, initial_scale=initial_scale)

        self._initialize_weights()

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        layer1 = BasicBlock(self.in_planes, dim, stride)
        layer2 = BasicBlock(dim, dim, 1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    @staticmethod
    def _conv1x1(in_c, out_c):
        return nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

    @staticmethod
    def _conv3x3(in_c, out_c):
        return nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _split_input(self, x):
        if not self.use_phys:
            if x.shape[1] != 1:
                raise ValueError(f"Baseline mode expects 1 channel, got {x.shape[1]}")
            return x, None, None
        else:
            if x.shape[1] != 3:
                raise ValueError(f"Physics mode expects 3 channels [img,phys,mask], got {x.shape[1]}")
            return x[:, 0:1], x[:, 1:2], x[:, 2:3]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Input: [B, 3, H, W] = [img, phys, mask]
        Output: [coarse_feat (1/8), fine_feat (1/2)]
        """
        img, phys, mask = self._split_input(x)

        # --- 1. Input SFT (Pre-Backbone Injection) ---
        # "Imprints" thermal edges onto the visible image before processing
        if self.inject_input_sft and phys is not None:
            # Resize just in case, though usually input is full res
            p0 = self.mask_resizer.resize_with_mask(phys, mask, img.shape[-2:], 'bilinear')
            m0 = self.mask_resizer.resize_mask(mask, img.shape[-2:])
            img = self.sft_input(img, p0, m0)

        # Backbone
        x0 = self.relu(self.bn1(self.conv1(img)))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # --- 2. Coarse SFT (Stride 8) ---
        # Injected AFTER backbone, BEFORE FPN fusion
        if self.inject_coarse_sft and phys is not None:
            p3 = self.mask_resizer.resize_with_mask(phys, mask, x3.shape[-2:], 'bilinear')
            m3 = self.mask_resizer.resize_mask(mask, x3.shape[-2:])
            x3 = self.sft_coarse(x3, p3, m3)

        # FPN coarse path
        x3_out = self.layer3_outconv(x3)
        x3_out_2x = F.interpolate(x3_out, scale_factor=2.0, mode='bilinear', align_corners=True)

        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        # FPN fine path
        x2_out_2x = F.interpolate(x2_out, scale_factor=2.0, mode='bilinear', align_corners=True)

        # Lateral Projection (Stride 2)
        x1_out = self.layer1_outconv(x1)

        # Fusion + refinement (contains BN inside layer1_outconv2)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        # --- 3. Fine SFT (Stride 2) ---
        # Injected AFTER fusion/refinement so BN doesn’t damp it immediately
        if self.inject_fine_sft and phys is not None:
            p1 = self.mask_resizer.resize_with_mask(phys, mask, x1_out.shape[-2:], 'bilinear')
            m1 = self.mask_resizer.resize_mask(mask, x1_out.shape[-2:])
            x1_out = self.sft_fine(x1_out, p1, m1)

        return [x3_out, x1_out]

    def get_sft_stats(self) -> dict:
        stats = {}
        if self.inject_input_sft:
            stats['input_scale'] = self.sft_input.get_modulation_strength()
        if self.inject_coarse_sft:
            stats['coarse_scale'] = self.sft_coarse.get_modulation_strength()
        if self.inject_fine_sft:
            stats['fine_scale'] = self.sft_fine.get_modulation_strength()
        return stats



class ResNet_8_2_SFT_XoFTR(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        # --- Config Extraction ---
        backbone_cfg = config['backbone']
        phys_cfg = config['phys']
        sft_cfg = config.get('sft', {})

        initial_dim = backbone_cfg['initial_dim']
        block_dims = backbone_cfg['block_dims']
        self.use_phys = phys_cfg.get("use_phys", True)

        # Injection Toggles
        self.inject_input = phys_cfg.get('inject_input_sft', True) and self.use_phys
        self.inject_coarse = phys_cfg.get('inject_coarse_sft', True) and self.use_phys
        self.inject_fine = phys_cfg.get('inject_fine_sft', True) and self.use_phys

        # --- Standard ResNet Components ---
        self.in_planes = initial_dim
        # XoFTR starts with a stride-2 conv, reducing resolution to 1/2 immediately
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(BasicBlock, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(BasicBlock, block_dims[2], stride=2)  # 1/8

        self.layer3_outconv = nn.Conv2d(block_dims[2], block_dims[2], kernel_size=1, bias=False)

        # --- SFT Modules ---
        if self.use_phys:
            self.mask_resizer = MaskAwareResize()

            # Shared SFT settings
            sft_kwargs = {
                'bottleneck_dim': sft_cfg.get('bottleneck_dim', 128),
                'dropout_p': sft_cfg.get('dropout_p', 0.1),
                'learnable_scale': sft_cfg.get('learnable_scale', True),
                'initial_scale': sft_cfg.get('initial_scale', 0.1)
            }

            # INPUT INJECTION: Modulates the 1-channel raw image
            if self.inject_input:
                self.sft_input = SFT(c_feat=1, **sft_kwargs)

            if self.inject_coarse:
                self.sft_coarse = SFT(c_feat=block_dims[2], **sft_kwargs)

            if self.inject_fine:
                self.sft_fine = SFT(c_feat=block_dims[0], **sft_kwargs)

        self._initialize_weights()

    def _make_layer(self, block, dim, stride=1):
        l1 = block(self.in_planes, dim, stride=stride)
        l2 = block(dim, dim, stride=1)
        self.in_planes = dim
        return nn.Sequential(l1, l2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, 3, H, W] -> [Grayscale, Physics, Mask]
        Returns: x3 (1/8), x2 (1/4), x1 (1/2)
        """
        img = x[:, 0:1]

        # --- Stage 0: Input Injection ---
        if self.inject_input:
            phys = x[:, 1:2]
            mask = x[:, 2:3]
            # No resizing needed for input-level; usually matches image resolution
            img = self.sft_input(img, phys, mask)

        # --- Backbone ---
        # Note: self.conv1 has stride 2, so x0 is 1/2 resolution
        x0 = self.relu(self.bn1(self.conv1(img)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # --- Stage 1: Coarse Injection (1/8) ---
        if self.inject_coarse:
            phys = x[:, 1:2]
            mask = x[:, 2:3]
            p3 = self.mask_resizer.resize_with_mask(phys, mask, x3.shape[-2:], 'bilinear')
            m3 = self.mask_resizer.resize_mask(mask, x3.shape[-2:])
            x3 = self.sft_coarse(x3, p3, m3)

        x3_out = self.layer3_outconv(x3)

        # --- Stage 2: Fine Injection (1/2) ---
        if self.inject_fine:
            phys = x[:, 1:2]
            mask = x[:, 2:3]
            p1 = self.mask_resizer.resize_with_mask(phys, mask, x1.shape[-2:], 'bilinear')
            m1 = self.mask_resizer.resize_mask(mask, x1.shape[-2:])
            x1 = self.sft_fine(x1, p1, m1)

        return x3_out, x2, x1





