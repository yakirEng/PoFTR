from src.losses.loftr_loss import LoFTRLoss
from src.losses.aspan_loss import ASpanLoss
from src.losses.xoftr_loss import XoFTRLoss


def define_loss(base_model, config):
    if base_model == 'loftr':
        loss = LoFTRLoss(config['method'])
    elif base_model == 'aspanformer':
        loss = ASpanLoss(config['method'])
    elif base_model == 'xoftr':
        loss = XoFTRLoss(config['method'])
    else:
        raise ValueError(f"Unknown base_model: {config['proj']['base_model']}")
    return loss