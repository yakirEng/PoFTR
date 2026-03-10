from src.PoFTR.backbone.phys_backbone import  ResNetFPN_8_2_SFT, ResNet_8_2_SFT_XoFTR

def build_phys_backbone(config):
    base_model = config['proj']['base_model']
    if base_model == 'loftr' or base_model == 'aspanformer':
        return ResNetFPN_8_2_SFT(config)
    elif base_model == 'xoftr':
        return ResNet_8_2_SFT_XoFTR(config)