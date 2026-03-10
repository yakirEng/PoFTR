import os
import platform

import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl

from datetime import datetime
import random
import numpy as np
from loguru import logger

from lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.model_summary import summarize
from src.utils.profiler import build_profiler

from src.PoFTR.lightning.pl_poftr import PL_PoFTR
from src.PoFTR.lightning.data_module import SATDataModule


@rank_zero_only
def summarize_model(model):
    """
    Summarizes the model architecture and parameters.
    """
    summarize(model)


def get_callbacks(config):
    phys_str = "phys" if config['phys']['use_phys'] else "no_phys"
    dataset_version = config['data']['dataset_version']
    model_name = config['proj']['base_model']
    checkpoint_dir = f"./checkpoints/best/{dataset_version}/{model_name}/{phys_str}/{datetime.now().strftime('%d_%m__%H_%M')}"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    metric_key = "val/Pose_Success_10px"

    if local_rank != 0:
        return []

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=metric_key,
        verbose=True,
        save_top_k=1,
        mode="max",
        filename='best',
        save_on_train_epoch_end=False,
        save_last=True
    )
    callbacks = [checkpoint_callback]
    return callbacks

def get_trainer_args(config, mlflow_logger, callbacks):

    # profiler= build_profiler('pytorch', config)
    trainer_args = dict(
        max_epochs=config['train']['max_epochs'],
        accelerator=config['run']['accelerator'],
        num_nodes=config['run']['num_nodes'],
        devices=torch.cuda.device_count(),
        log_every_n_steps=config['run']['log_every_n_steps'],
        logger=mlflow_logger,
        deterministic=False,
        callbacks=callbacks,
        inference_mode=False,
        enable_progress_bar=True,
    )

    strategy = strategy_picker(config)

    if strategy:
        trainer_args["strategy"] = strategy
    return trainer_args


def strategy_picker(config):
    num_gpus = config['run']['num_gpus']
    is_windows = platform.system() == "Windows"
    print("Number of Nodes:", config['run']['num_nodes'],
          "Number of GPUs detected:", num_gpus,
          "OS:", platform.system())
    # If there are no GPUs, no distributed training is needed.
    if num_gpus <= 1:
        if num_gpus == 0:
            logger.info("No GPUs detected, no distributed strategy needed.")
        else:
            logger.info("Single GPU detected, no distributed strategy needed.")
        return None

    # For single-node setups with multiple GPUs:
    elif config['run']['num_nodes'] == 1:
        if is_windows:
            logger.info("Single node, multiple GPUs detected on Windows. Using 'ddp_spawn' strategy.")
            return "ddp_spawn"
        else:
            logger.info("Single node, multiple GPUs detected on Linux. Using 'ddp' strategy.")
            return "ddp"

    # For multi-node setups (this is for distributed training across multiple machines):
    else:
        logger.info("Multi-node setup detected. Using 'DDPStrategy' for distributed training.")
        return DDPStrategy(find_unused_parameters=True)


def set_running_environment(config):
    # 1. Precision: Good for Ampere (RTX 30xx/40xx) cards.
    # Use 'high' if you see numerical instability.
    torch.set_float32_matmul_precision('medium')

    seed = config['data']['master_seed']

    # 2. Lightning Seeding: This REPLACES the manual random/np/torch calls.
    # It sets Python, NumPy, and Torch seeds + handles DataLoader workers.
    pl.seed_everything(seed, workers=True)

    # 3. Determinism: pl.seed_everything doesn't strictly enforce these,
    # so keeping them manual is good for your specific requirements.
    torch.backends.cudnn.benchmark = False  # False = reproducible, True = faster
    # torch.backends.cudnn.deterministic = True # Uncomment for strict reproducibility (might crash some ops)

    # 4. Matplotlib settings
    import matplotlib.pyplot as plt
    plt.rcParams['figure.max_open_warning'] = 100

    # 5. uncomment this if debugging a crash/NaN error.
    # torch.autograd.set_detect_anomaly(True)

    # Logging
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    logger.info(f"Running environment set. Seed: {seed}, World Size: {world_size}, Rank: {rank}")


def load_pretrained(pl_model: PL_PoFTR):
    checkpoint_state_dict = torch.load(pl_model.pretrained_ckpt, map_location='cpu', weights_only=True)['state_dict']
    model_sd = pl_model.model.state_dict()
    for name, tensor in model_sd.items():
        if pl_model.base_model == 'loftr':
            converted_name = name.removeprefix('model.')
        elif pl_model.base_model == 'aspanformer':
            converted_name = name.replace('model.', 'matcher.')
        elif pl_model.base_model == 'xoftr':
            converted_name = name.replace('model.', 'matcher.')
        else:
            raise ValueError(f"Unknown base_model: {pl_model.base_model}")
        if converted_name in checkpoint_state_dict and tensor.shape == checkpoint_state_dict[converted_name].shape:
            model_sd[name] = checkpoint_state_dict[converted_name].clone()
        else:
            logger.info(f"Pretrained weights not found or shape mismatch for layer: {converted_name}")

    # update the lightning model:
    pl_model.model.load_state_dict(model_sd)
    return pl_model



def restore_sample(k, batch):
    """
    for debugging purposes:
    Restores all keypoints for the k-th sample from the batch.

    Args:
        k: Index of the keypoint to restore.
        batch: Batch dictionary from the collate_fn (includes images, keypoints, indices).

    Returns:
        restored_sample: The original sample corresponding to index k (all keypoints in that batch).
    """

    # Extract necessary data from the batch
    images0_concat = batch['image0']  # Shape: [B, C, H, W]
    images1_concat = batch['image1']  # Shape: [B, C, H, W]
    kp0_concat = batch['kp0']  # Shape: [total_keypoints_0, 2]
    kp1_concat = batch['kp1']  # Shape: [total_keypoints_1, 2]
    b_ids = batch['b_ids']  # Shape: [total_keypoints]
    i_ids = batch['i_ids']  # Shape: [total_keypoints]
    j_ids = batch['j_ids']  # Shape: [total_keypoints]

    # 1. Get the batch index (which image in the batch the k-th keypoint belongs to)
    batch_idx = k  # Get the batch index of the k-th keypoint

    # 2. Get all keypoints that belong to the same image (same batch_idx)
    batch_mask = (b_ids == batch_idx)  # Boolean mask to select all keypoints from the same image

    # Filter keypoints from image0 and image1 based on the mask
    kp0_sample = kp0_concat[batch_mask]  # All keypoints for image0 in this batch
    kp1_sample = kp1_concat[batch_mask]  # All keypoints for image1 in this batch

    i_ids_sample = i_ids[batch_mask]  # Indices for image0 keypoints
    j_ids_sample = j_ids[batch_mask]  # Indices for image1 keypoints

    # 3. Restore the image for this batch
    image0_sample = images0_concat[batch_idx]  # Shape: [C, H, W]
    image1_sample = images1_concat[batch_idx]  # Shape: [C, H, W]

    # 4. Restore the FPA temperature (if necessary)
    fpa_sample = batch['t_fpa'][batch_idx]  # FPA for this image pair

    # 5. Return the restored sample as a dictionary
    restored_sample = {
        'image0': image0_sample,
        'image1': image1_sample,
        'kp0_gt': kp0_sample,  # All keypoints for image0
        'kp1_gt': kp1_sample,  # All keypoints for image1
        't_fpa': fpa_sample

    }

    return restored_sample, i_ids_sample, j_ids_sample

def restore_and_show(k, batch):
    """
    for debugging purposes:
    Restores the k-th sample from the batch and prints its details.

    Args:
        k: Index of the sample to restore.
        batch: Batch dictionary from the collate_fn (includes images, keypoints, indices).
    """
    restored_sample, i_ids_sample, j_ids_sample = restore_sample(k, batch)

    print(f"Restored Sample {k}:")
    print("Image0 shape:", restored_sample['image0'].shape)
    print("Image1 shape:", restored_sample['image1'].shape)
    print("Keypoints for Image0:", restored_sample['kp0_gt'].shape)
    print("Keypoints for Image1:", restored_sample['kp1_gt'].shape)
    print("FPA Temperature:", restored_sample['fpa'])
    print("i_ids:", i_ids_sample)
    print("j_ids:", j_ids_sample)

    fig, ax = plt.subplots(1, 2)
    plt.suptitle(f"Sample {k}")
    ax[0].imshow(restored_sample['image0'].detach().cpu().numpy().squeeze(), cmap='gray')
    ax[0].scatter(restored_sample['kp0_gt'][:, 0].detach().cpu().numpy(),
                  restored_sample['kp0_gt'][:, 1].detach().cpu().numpy(),
                  s=5, c='blue', alpha=0.2, edgecolors='none')
    ax[0].set_title("Image0")

    ax[1].imshow(restored_sample['image1'].detach().cpu().numpy().squeeze(), cmap='gray')
    ax[1].scatter(restored_sample['kp1_gt'][:, 0].detach().cpu().numpy(),
                  restored_sample['kp1_gt'][:, 1].detach().cpu().numpy(),
                  s=5, c='blue', alpha=0.2, edgecolors='none')
    ax[1].set_title("Image1")
    plt.show()