from pytorch_lightning.callbacks import Callback
import GPUtil
from src.utils.callback_helper import *




class ConfMapsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Check if conf_maps has been populated
        if hasattr(pl_module, 'log_data') and pl_module.log_data:
            log_data = pl_module.log_data
            # Use the global_step or current_epoch for logging
            epoch = trainer.current_epoch
            log_confidence_maps_mlflow(
                conf_gt=log_data['conf_gt'],
                conf_pred=log_data['conf_pred'],
                conf_spvs=log_data['conf_spvs'],
                spvs_ratio=log_data['spvs_ratio'],
                epoch=epoch,
                trainer=trainer,
            )

class GPUMetricsCallback(Callback):
    """
    A PyTorch Lightning callback that logs GPU statistics to MLflow at the end of each training epoch.
    It logs GPU load, memory utilization, free memory, used memory, and temperature for every available GPU.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Retrieve information for all available GPUs
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            mlflow_logger = trainer.logger
            # gpu.load and gpu.memoryUtil are fractions (0 to 1). Multiplying by 100 for percentage.
            mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f'system/gpu_{gpu.id}_load_percent', gpu.load * 100, step=epoch)
            mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f'system/gpu_{gpu.id}_memory_util_percent', gpu.memoryUtil * 100, step=epoch)
            mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f'system/gpu_{gpu.id}_free_memory_mb', gpu.memoryFree, step=epoch)
            mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f'system/gpu_{gpu.id}_used_memory_mb', gpu.memoryUsed, step=epoch)
            mlflow_logger.experiment.log_metric(mlflow_logger.run_id, f'system/gpu_{gpu.id}_temperature_celsius', gpu.temperature, step=epoch)


class ConfidenceMaps2(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Logs confidence maps (ground truth & prediction) to MLflow at the end of each epoch.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: The model (LightningModule).
        """
        # Ensure `conf_maps` exists in the model
        if hasattr(pl_module, 'log_data') and pl_module.log_data:
            log_data = pl_module.log_data
            epoch = trainer.current_epoch

            # Call the function to log only GT and prediction
            log_confidence_maps2(
                conf_gt=log_data['conf_c_gt'],
                conf_pred=log_data['conf_c_pred'],
                epoch=epoch,
                type='coarse',
                trainer=trainer,
            )

class CoarsePredCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, 'log_data') and pl_module.log_data:
            log_data = pl_module.log_data
            epoch = trainer.current_epoch
            log_coarse_pred(log_data, trainer, epoch, self.config)

class LossPlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        # 1) record losses if available
        train_loss = metrics.get("train/avg_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
        val_loss = metrics.get("val/avg_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())

        # 2) plot losses
        log_train_val_losses(
            trainer,
            self.train_losses,
            self.val_losses,
        )

class ProfilerLogCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.mlflow_client.log_artifact(
            run_id=trainer.logger.run_id,
            local_path = "profiler_logs/pytorch_profile.trace.json",
            artifact_path="traces",
        )


