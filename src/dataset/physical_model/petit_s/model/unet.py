import torch
import torch.nn as nn
import torch.nn.functional as F

def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.InstanceNorm2d(cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.InstanceNorm2d(cout), nn.SiLU(inplace=True),
    )

class PetitS_UNet(nn.Module):
    def __init__(self, c_in=2, chans=(256, 512, 1024)):
        super().__init__()
        c1, c2, c3 = chans
        self.e1 = _block(c_in, c1)
        self.e2 = _block(c1, c2)
        self.e3 = _block(c2, c3)
        self.pool = nn.AvgPool2d(2)
        self.d2  = _block(c3 + c2, c2)
        self.d1  = _block(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):                      # x: [B,2,H,W] = [simulator, t_fpa_z_tiled]
        s1 = self.e1(x)                        # H
        s2 = self.e2(self.pool(s1))            # H/2
        b  = self.e3(self.pool(s2))            # H/4
        u2 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.d2(torch.cat([u2, s2], 1))
        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.d1(torch.cat([u1, s1], 1))
        return self.out(u1)                    # [B,1,H,W], Celsius logits
