import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import matplotlib
matplotlib.use("Agg")

from src.dataset.physical_model.petit_s.utils.train_utils import (
    summarize_model,
    set_running_environment,
    get_trainer_args, get_callbacks)

from src.dataset.physical_model.petit_s.utils.petits_configs import get_config, lower_config

from src.dataset.physical_model.petit_s.utils.petits_data_module import PetitSDataModule
from src.dataset.physical_model.petit_s.model.pl_petit_s import PL_PetitS
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


def main():
    config = lower_config(get_config())
    fold_idx = config['run']['fold_idx']
    set_running_environment(config)
    os.chdir(config['proj']['cwd'])

    # ---------------------------
    # Datasets and Data Module
    # ---------------------------
    data_module = PetitSDataModule(config=config)

    # ---------------------------
    # Logger
    # ---------------------------
    mlflow_logger = MLFlowLogger(
        experiment_name=config['run']['experiment_name'],
        run_name=config['run']['run_name'],
        tracking_uri=config['run']['tracking_uri'],
    )
    mlflow_logger.finalize = lambda status=None: None

    callbacks = get_callbacks(config, fold_idx)

    # ---------------------------
    # Trainer args
    # ---------------------------
    trainer_args = get_trainer_args(config, mlflow_logger, callbacks)

    model = PL_PetitS(config=config, data_module=data_module)

    summarize_model(model)

    # ---------------------------
    # Train the model.
    # ---------------------------
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, data_module)

    # ---------------------------
    # Test the model.
    # ---------------------------
    trainer.test(model, data_module)

    mlflow_logger.experiment.log_artifacts(
        mlflow_logger.run_id,
        local_dir=f"./profiler_logs/{config['run']['experiment_name']}/{config['run']['run_name']}",
        artifact_path="../profiler_logs"
    )
    mlflow_logger.experiment.set_terminated(mlflow_logger.run_id, status="FINISHED")


if __name__ == "__main__":
    main()