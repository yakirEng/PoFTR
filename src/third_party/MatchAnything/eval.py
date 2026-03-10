import os
import torch
import logging
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import MLFlowLogger

# --- Import your project modules ---
from src.third_party.MatchAnything.dataset.ma_pl_datamodule import MatchAnythingDataModule
from src.third_party.MatchAnything.model.pl_matchanything import PL_MatchAnything
from src.third_party.MatchAnything.utils.ma_config import lower_config, get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# USER SETTINGS
# ==========================================
# Choose the specific dataset version you want to evaluate here
TARGET_SCENARIO = "9um_11um"  # Options: "9um_pan", "11um_pan", "9um_11um"


# ==========================================

def main():
    # 1. Load Configuration
    config = lower_config(get_config())

    # 2. Setup Global Seed
    pl.seed_everything(config["poftr"]["data"]["master_seed"])

    # 3. Setup Device
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    # --- A. CRITICAL FIX: Update Config for TARGET_SCENARIO ---
    physat_cfg = config['poftr']

    # 1. Update the version tag
    physat_cfg['data']['dataset_version'] = TARGET_SCENARIO

    # 2. Construct the absolute 'root_dir' path
    #    Logic derived from your previous MonochromDs structure:
    #    Path = cwd / data / simulated / datasets / distribution / version / test

    cwd = physat_cfg['proj']['cwd']
    dist_type = physat_cfg['sim']['distribution_type']

    # Construct path: .../data/simulated/datasets/truncnorm/9um_pan/test
    root_dir = os.path.join(
        cwd,
        "data", "simulated", "datasets",
        dist_type,
        TARGET_SCENARIO,
        "test"
    )

    # 3. Inject into config (This prevents the ValueError)
    physat_cfg['data']['root_dir'] = root_dir

    # Verify path exists
    if not os.path.exists(root_dir):
        logger.error(f"CRITICAL: The constructed dataset path does not exist:\n{root_dir}")
        return

    logger.info(f"Running evaluation on: {TARGET_SCENARIO}")
    logger.info(f"Root Directory: {physat_cfg['data']['root_dir']}")

    # --- B. Initialize Data Module ---
    dm = MatchAnythingDataModule(
        config=physat_cfg,
        splits=['test'],
        is_analysis_mode=False
    )
    dm.setup()

    # --- C. Initialize Model ---
    model = PL_MatchAnything(config, dm)

    # --- D. Initialize Logger ---
    mlf_logger = None
    train_logger = physat_cfg['train']['logger']
    tracking_uri = physat_cfg['run']['tracking_uri']

    if train_logger == "mlflow":
        mlf_logger = MLFlowLogger(
            experiment_name="MatchAnything_ZeroShot",
            run_name=f"eval_{TARGET_SCENARIO}",
            tracking_uri=tracking_uri
        )

    # --- E. Initialize Trainer ---
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=mlf_logger,
        enable_checkpointing=False,
        inference_mode=True,
        log_every_n_steps=10
    )

    # --- F. Run Test ---
    logger.info(f"Starting test cycle...")
    trainer.test(model, datamodule=dm)
    logger.info(f"Finished evaluation for {TARGET_SCENARIO}.")


if __name__ == "__main__":
    main()