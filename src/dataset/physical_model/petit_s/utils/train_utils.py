import numpy as np
import os
import random
import torch
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
from loguru import logger
from pytorch_lightning.strategies import DDPStrategy
import platform
from pathlib import Path

from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def set_running_environment(config):
    torch.set_float32_matmul_precision('medium')  # or 'high'
    # 1) Python built-in
    seed = config['train']['seed']
    random.seed(seed)
    # 2) NumPy
    np.random.seed(seed)
    # 3) Torch on CPU & GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 4) Make CuDNN deterministic (may slow you down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # 5) Python hash seed (for any hashing order)
    os.environ['PYTHONHASHSEED'] = str(seed)
    plt.rcParams['figure.max_open_warning'] = 100
    # 6) Lightning: seeds workers & does extra book-keeping
    #    Setting workers=True ensures each DataLoader worker gets a unique but deterministic seed.
    seed_everything(seed, workers=True)

    # scale lr and warmup-step automatically
    # config.TRAIN.world_size = config.RUN.num_gpus * config.RUN.num_nodes
    # config.TRAIN.true_batch_size = config.TRAIN.world_size * config.TRAIN.batch_size
    torch.autograd.set_detect_anomaly(True)
    logger.info(f"Running environment set. Seed: {seed}, World Size: {int(os.environ.get('WORLD_SIZE', 1))}, rank: {int(os.environ.get('RANK', 0))}, ")


def get_callbacks(config, fold_idx):
    """
    Returns a list of callbacks for the PyTorch Lightning Trainer.
    """
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_path = Path(config['proj']['checkpoints_path']) / config['proj']['base_model'] / config['data']['wl'] / f"fold_{str(fold_idx)}" / "second"
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = f"f{fold_idx}_e{{epoch:02d}}_lpips{{val/lpips:.4f}}"
    # Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=filename,
        monitor="val/lpips",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks = [checkpoint_callback]

    return callbacks

def get_trainer_args(config, mlflow_logger, callbacks=None):
    trainer_args = dict(
        max_epochs=config["train"]["max_epochs"],
        accelerator="gpu",
        num_nodes=config["run"]["num_nodes"],
        devices=config["run"]["num_gpus"],
        callbacks=callbacks,
        log_every_n_steps=config["run"]["log_every_n_steps"],
        logger=mlflow_logger,
        deterministic=True,
        inference_mode=False,
        enable_progress_bar=True,
    )
    #         profiler=build_profiler('pytorch', config),
    strategy = strategy_picker(config)

    if strategy:
        trainer_args["strategy"] = strategy
    return trainer_args


def strategy_picker(config):
    num_gpus = config["run"]["num_gpus"]
    is_windows = platform.system() == "Windows"
    print("Number of Nodes:", config["run"]["num_nodes"],
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
    elif config["run"]["num_nodes"] == 1:
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


@rank_zero_only
def summarize_model(model):
    """
    Summarizes the model architecture and parameters.
    """
    summarize(model)