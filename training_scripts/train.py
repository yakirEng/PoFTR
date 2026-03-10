import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import matplotlib
matplotlib.use("Agg")

from src.utils.train_helper import (get_callbacks,
                                    get_trainer_args, summarize_model,
                                    set_running_environment, load_pretrained)

from src.configs.poftr_configs import get_config
from src.PoFTR.lightning.pl_poftr import PL_PoFTR
from src.PoFTR.lightning.data_module import SATDataModule
from src.utils.misc import lower_config
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

def main():

    config = lower_config(get_config())
    set_running_environment(config['poftr'])

    # ---------------------------
    # Datasets and Data Module
    # ---------------------------
    data_module = SATDataModule(config=config['poftr'])

    # ---------------------------
    # Callbacks
    # ---------------------------
    callbacks = get_callbacks(config=config['poftr'])

    tags = {}
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        tags["job_id"] = job_id

    # ---------------------------
    # Logger
    # ---------------------------
    mlflow_logger = MLFlowLogger(
        experiment_name=config['poftr']['run']['experiment_name'],
        run_name=config['poftr']['run']['run_name'],
        tracking_uri=config['poftr']['run']['tracking_uri'],
        tags=tags,
    )
    mlflow_logger.finalize = lambda status=None: None

    # ---------------------------
    # Trainer args
    # ---------------------------
    trainer_args = get_trainer_args(config['poftr'], mlflow_logger, callbacks)

    model = PL_PoFTR(config=config, data_module=data_module)

    summarize_model(model)

    # ---------------------------
    # Fine tune the model.
    # ---------------------------
    pl_model = load_pretrained(model)

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(pl_model, data_module)

    # ---------------------------
    # Test the model.
    # ---------------------------
    trainer.test(pl_model, data_module, ckpt_path='best')


if __name__ == "__main__":
    main()