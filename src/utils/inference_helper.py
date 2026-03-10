import os
import torch
from src.LoFTR.pl_loftr import PL_LoFTR
from src.AspanFormer.pl_aspanformer import PL_AspanFormer
from src.utils.configs import Config, InferenceConfig


def _get_checkpoint_path(config: InferenceConfig, base_model=None):
    base_model = base_model or config.base_model
    checkpoint_path = config.inference_path/ base_model/ config.sub_model
    model_filename = os.listdir(checkpoint_path)[0]
    return  checkpoint_path / model_filename


def _get_pl_model(config: Config, data_module, base_model=None):
    base_model = base_model or config.inference.base_model
    if base_model == 'loftr':
        return PL_LoFTR(config, data_module)
    elif base_model == 'aspanformer':
        return PL_AspanFormer(config, data_module)
    else:
        raise ValueError(f"Unsupported base model: {base_model}")

def _load_model_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_inference_model(config: Config, data_module, base_model='loftr'):
    checkpoint_path = _get_checkpoint_path(config.inference, base_model)
    model = _get_pl_model(config, data_module)
    model = _load_model_weights(model, checkpoint_path)
    return model


def get_inference_args():
    """
    Get inference arguments based on the configuration and base model.
    """
    inference_args = dict(
        accelerator="gpu",
        deterministic=True,
        inference_mode=True,
        enable_progress_bar=True,
    )
    return inference_args

def set_inference_environment(config):
    """
    Set the environment for inference based on the configuration.
    """
    os.chdir(config.cwd)