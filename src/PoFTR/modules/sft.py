import torch
import torch.nn as nn
import warnings



class SFT(nn.Module):
    """
    Spatial Feature Transform: out = x * (1 + scale * gamma) + scale * beta
    """

    def __init__(
            self,
            c_feat: int,
            bottleneck_dim: int = 128,
            dropout_p: float = 0.1,
            learnable_scale: bool = True,
            initial_scale: float = 0.1
    ):
        super().__init__()
        self.c_feat = c_feat

        # MLP: [phys, mask] -> [gamma, beta]
        self.mlp = nn.Sequential(
            nn.Conv2d(2, bottleneck_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(bottleneck_dim, 2 * c_feat, 1, 1, 0)
        )

        # Learnable gating
        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(initial_scale))
        else:
            self.register_buffer('output_scale', torch.tensor(initial_scale))

        self._initialize_weights()

    def _initialize_weights(self):
        for i, m in enumerate(self.mlp):
            if isinstance(m, nn.Conv2d):
                # Check if this is the very last layer in the MLP sequence
                if i == len(self.mlp) - 1:
                    # Final layer: EXACT ZERO.
                    # Combined with initial_scale=1.0, this ensures the model starts
                    # at the baseline performance but has maximum 'plasticity'
                    # to learn immediately.
                    nn.init.constant_(m.weight, 0.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                else:
                    # Intermediate layers: Kaiming Normal.
                    # Essential to keep these random so gradients can flow
                    # through the "Zero Wall" of the final layer.
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, phys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Validate shapes
        if phys.dim() == 3:
            phys = phys.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        if phys.shape[1] != 1:
            raise ValueError(f"Physics must have 1 channel, got {phys.shape[1]}")
        if mask.shape[1] != 1:
            raise ValueError(f"Mask must have 1 channel, got {mask.shape[1]}")
        if phys.shape[-2:] != x.shape[-2:]:
            raise ValueError(f"Size mismatch: x={x.shape[-2:]}, phys={phys.shape[-2:]}")

        # Ensure binary mask
        mask = mask.float()
        if not torch.all((mask == 0) | (mask == 1)):
            warnings.warn("Non-binary mask, binarizing at 0.5")
            mask = (mask > 0.5).float()

        # Forward
        phys_masked = phys * mask
        mlp_input = torch.cat([phys_masked, mask], dim=1)
        modulation = self.mlp(mlp_input)
        gamma, beta = modulation.chunk(2, dim=1)

        # Mask modulation
        gamma = gamma * mask
        beta = beta * mask

        # Apply
        scale = self.output_scale
        out = x * (1.0 + scale * gamma) + scale * beta

        return out

    def get_modulation_strength(self) -> float:
        return self.output_scale.item()



