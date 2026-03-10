import torch.nn as nn

from src.third_party import LoFTR
from src.third_party import ASpanFormer
from src.PoFTR.backbone import build_phys_backbone
from src.third_party.XoFTR.src.xoftr import XoFTR


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
            phys_backbone = build_phys_backbone(self.config['poftr'])
            model.backbone = phys_backbone

        return model

    def forward(self, data):
        return self.model(data)


