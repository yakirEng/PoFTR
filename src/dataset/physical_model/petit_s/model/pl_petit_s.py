from turtledemo.nim import computerzug
from typing import Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
import numpy as np


from src.dataset.physical_model.petit_s.optimizers import build_optimizer, build_scheduler
from src.dataset.physical_model.petit_s.model.unet import PetitS_UNet
from src.dataset.physical_model.petit_s.utils.losses import compute_loss
from src.dataset.physical_model.petit_s.utils.metrics import ThermalMetrics

class PL_PetitS(pl.LightningModule):
    """
    Lightning Module for PETIT-S (simulator-only temperature student).
    Expects batches with:
      - 'simulator'      : [B,1,H,W]  normalized simulator (GAN) image
      - 't_fpa'     : [B]        sensor intrinsic temp in °C
      - 'T_teacher' : [B,1,H,W] or [B,H,W] teacher temperature in °C
    """
    def __init__(self, config, data_module=None, net: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.data_module = data_module

        # network
        self.net = net if net is not None else PetitS_UNet(c_in=2)

        # plotting
        self.plot_counter = 0
        self.val_plot_interval = config['train']['val_plot_interval']
        self.n_val_pairs_to_plot = config['train']["n_val_pairs_to_plot"]

        # buffers for epoch-end aggregation (if needed later)
        self._val_outputs = []

        # where to dump artifacts
        self.dump_dir = getattr(config, "results_path", "./results")

        self.metrics_engine = None

    # ---------- MLflow helpers ----------
    @property
    def mlflow_client(self):
        if isinstance(self.logger, MLFlowLogger):
            return self.logger.experiment
        return None

    @property
    def mlflow_run_id(self):
        if isinstance(self.logger, MLFlowLogger):
            return self.logger.run_id
        return None

    # ---------- lifecycle ----------
    @rank_zero_only
    def on_train_start(self):
        to_log = {
            "machine":       self.config["run"]["run_platform"],
            "fold_idx":      self.config["run"]["fold_idx"],
            "initial_lr":    self.config["train"]["initial_lr"],
            "batch_size":    self.config["train"]["batch_size"],
            "weight_decay":  self.config["train"]["weight_decay"],
            "scheduler":     self.config["train"]["scheduler"],
        }
        if self.logger is not None:
            self.logger.log_hyperparams(to_log)

    def on_validation_epoch_start(self):
        self.plot_counter = 0

    def on_test_epoch_start(self):
        self.plot_counter = 0

    # ---------- core modules ----------
    def _tile_t_fpa(self, t_fpa: torch.Tensor, like_img: torch.Tensor) -> torch.Tensor:
        """z-score t_fpa (°C) and tile to [B,1,H,W] matching like_img."""
        #t = (t_fpa.squeeze(1) - self.mu_fpa_c) / (self.sigma_fpa_c + 1e-8)  # [B]
        return t_fpa.view(-1, 1, 1, 1).expand(like_img.shape)

    def forward(self, mono: torch.Tensor, t_fpa: torch.Tensor) -> torch.Tensor:
        """simulator: [B,1,H,W] (normalized image); t_fpa: [B] in °C. Returns T_pred °C [B,1,H,W]."""
        t_ch = self._tile_t_fpa(t_fpa, mono)
        x = torch.cat([mono, t_ch], dim=1)
        return self.net(x)

    # ---------- train/val/test ----------
    def training_step(self, batch, batch_idx):
        mono   = batch["mono_image"].float()           # [B,1,H,W]
        t_fpa  = batch["t_fpa"].float()          # [B]
        T_gt   = batch["pan_temp_map"].float()      # [B,1,H,W] or [B,H,W]

        T_pred = self(mono, t_fpa)               # [B,1,H,W], °C
        losses = compute_loss(T_pred, T_gt)

        # log
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, prog_bar=(name=="loss"), on_step=False, on_epoch=True, batch_size=mono.size(0))
        return {"loss": losses['loss']}

    def validation_step(self, batch, batch_idx):
        if self.metrics_engine is None:
            self.metrics_engine = ThermalMetrics(self.device)

        mono   = batch["mono_image"].float()
        t_fpa  = batch["t_fpa"].float()
        T_gt   = batch["pan_temp_map"].float()

        T_pred = self(mono, t_fpa)
        losses = compute_loss(T_pred, T_gt)

        for name, loss in losses.items():
            self.log(f"val/{name}", loss, prog_bar=(name=="loss"), on_epoch=True, sync_dist=True, batch_size=mono.size(0))

        # Metrics calculation and logging
        metrics = self.metrics_engine.compute_all(T_pred, T_gt)
        for k, v in metrics.items():
            self.log(f"val/{k}",
                     v,
                     prog_bar=(k == 'mae_c'),
                     on_epoch=True,
                     sync_dist=True,
                     batch_size=mono.size(0)
                     )
        # simple figures every so often on rank 0
        if (self.trainer.is_global_zero and self.mlflow_client and
                self.trainer.current_epoch % self.val_plot_interval == 0
                and self.plot_counter < self.n_val_pairs_to_plot):
            figs = self._make_temp_figs(T_pred.detach(), T_gt.detach(), mono)
            for name, fig in figs.items():
                self.mlflow_client.log_figure(self.mlflow_run_id, fig, f"figures/{name}/val/epoch_{self.current_epoch}/batch_{batch_idx}.png")
                plt.close(fig)
            self.plot_counter += 1
            plt.close('all')
        return metrics


    def test_step(self, batch, batch_idx):
        mono   = batch["mono_image"].float()
        t_fpa  = batch["t_fpa"].float()
        T_gt   = batch["pan_temp_map"].float()

        T_pred = self(mono, t_fpa)
        losses = compute_loss(T_pred, T_gt)

        for name, loss in losses.items():
            self.log(f"test/{name}", loss, prog_bar=(name=="loss"), on_epoch=True, sync_dist=True, batch_size=mono.size(0))

        # Metrics calculation and logging
        metrics = self.metrics_engine.compute_all(T_pred, T_gt)
        for k, v in metrics.items():
            self.log(f"test/{k}",
                     v,
                     prog_bar=(k == 'mae_c'),
                     on_epoch=True,
                     sync_dist=True,
                     batch_size=mono.size(0)
                     )

        # optional figs
        if (self.trainer.is_global_zero and self.mlflow_client and
                self.plot_counter < self.n_val_pairs_to_plot):
            figs = self._make_temp_figs(T_pred.detach(), T_gt.detach(), mono.detach())
            for name, fig in figs.items():
                self.mlflow_client.log_figure(self.mlflow_run_id, fig, f"figures/{name}/test/epoch_{self.current_epoch}/batch_{batch_idx}.png")
                plt.close(fig)
            self.plot_counter += 1
            plt.close('all')

        return metrics

    # ---------- optimizer ----------
    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def _plot_temp_comparison_maps(self, T_gt_np, T_pred_np, mono_np, err,
                                   temp_min, temp_max, err_abs_max,
                                   err_mae, err_std):
        """Generates the 1x3 temperature comparison plot (GT, Pred, Error)."""

        # Made figsize taller (18, 6) to make room for titles
        fig, axs = plt.subplots(1, 4, figsize=(18, 6))

        # T_gt
        im0 = axs[0].imshow(T_gt_np, cmap="magma", vmin=temp_min, vmax=temp_max)
        axs[0].set_title(f"T_gt (°C)\n(min: {T_gt_np.min():.2f}, max: {T_gt_np.max():.2f})")
        axs[0].axis("off")
        fig.colorbar(im0, ax=axs[0], shrink=0.8)

        # T_pred
        im1 = axs[1].imshow(T_pred_np, cmap="magma", vmin=temp_min, vmax=temp_max)
        axs[1].set_title(f"T̂_pred (°C)\n(min: {T_pred_np.min():.2f}, max: {T_pred_np.max():.2f})")
        axs[1].axis("off")
        fig.colorbar(im1, ax=axs[1], shrink=0.8)

        # mono image
        im1 = axs[2].imshow(mono_np, cmap="magma")
        axs[2].set_title(f"Mono Image (Synthetic)")
        axs[2].axis("off")
        fig.colorbar(im1, ax=axs[2], shrink=0.8)

        # Error (Delta T)
        im2 = axs[3].imshow(err, cmap="coolwarm", vmin=-err_abs_max, vmax=err_abs_max)
        axs[3].set_title(f"ΔT (Error) (°C)\n(MAE: {err_mae:.2f}, std: {err_std:.2f})")
        axs[3].axis("off")
        fig.colorbar(im2, ax=axs[3], shrink=0.8)

        fig.tight_layout()
        return fig

    def _plot_error_histogram(self, err, err_abs_max, err_mean, err_std, err_mae):
        """Generates the error histogram plot with statistics."""

        # Made figsize larger (7, 5) to make room for title and text box
        fig_h, ax_h = plt.subplots(figsize=(7, 5))
        ax_h.hist(err.flatten(), bins=100, range=(-err_abs_max, err_abs_max))

        ax_h.axvline(0, color='r', linestyle='--', linewidth=2)
        ax_h.set_title("ΔT Histogram (°C)")
        ax_h.set_xlabel("T̂_pred − T_gt (°C)")
        ax_h.set_ylabel("Pixel Count")

        stats_text = f"Mean: {err_mean:.2f}\nStd: {err_std:.2f}\nMAE: {err_mae:.2f}"
        ax_h.text(0.05, 0.95, stats_text, transform=ax_h.transAxes,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig_h.tight_layout()
        return fig_h

    def _make_temp_figs(self, T_pred: torch.Tensor, T_gt: torch.Tensor, mono: torch.Tensor):
        """
        Prepares data and calls dedicated plotting functions for validation figures.
        """
        # --- 1. Data Prep ---
        T_pred_np = T_pred[0, 0].detach().cpu().numpy()
        T_gt_np = (T_gt[0, 0] if T_gt.ndim == 4 else T_gt[0]).detach().cpu().numpy()
        mono_np = mono[0, 0].detach().cpu().numpy()
        err = T_pred_np - T_gt_np

        # --- 2. Calculate Statistics ---
        err_mae = np.abs(err).mean()
        err_std = err.std()
        err_mean = err.mean()

        # Use try/except for robustness if an image is all NaNs, etc.
        try:
            temp_min = min(np.nanmin(T_gt_np), np.nanmin(T_pred_np))
            temp_max = max(np.nanmax(T_gt_np), np.nanmax(T_pred_np))
            err_abs_max = np.nanmax(np.abs(err))

            # Handle the case where err_abs_max is 0 (perfect prediction)
            if err_abs_max == 0:
                err_abs_max = 1.0

        except Exception as e:
            print(f"Warning: Could not compute stats for plotting. Error: {e}")
            # Return empty if stats calculation fails
            return {}

            # --- 3. Generate Figures ---
        figs = {}
        try:
            figs["temps_comparison"] = self._plot_temp_comparison_maps(
                T_gt_np, T_pred_np, mono_np, err,
                temp_min, temp_max, err_abs_max,
                err_mae, err_std
            )

            figs["delta_hist"] = self._plot_error_histogram(
                err, err_abs_max, err_mean, err_std, err_mae
            )
        except Exception as e:
            print(f"Warning: Plot generation failed. Error: {e}")
            # Close any partially created figures to prevent memory leaks
            plt.close('all')
            return {}

        return figs