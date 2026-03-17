import random

import torch
from pathlib import Path
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from loguru import logger
from pytorch_lightning.loggers import MLFlowLogger
import torch.profiler

from src.PoFTR.poftr import PoFTR

from src.utils.supervise import supervise_coarse, supervise_fine
from src.utils.supervise_xoftr import supervise_coarse_xoftr, supervise_fine_xoftr

from src.utils.planar_metrics import compute_planar_metrics, compute_planar_metrics_raw
from src.utils.misc import flattenList
from src.utils.plotting import make_matching_figures
from src.optimizers import build_optimizer, build_scheduler
from src.utils.comm import all_gather, gather

from src.losses import define_loss



class PL_PoFTR(pl.LightningModule):
    def __init__(self, config, data_module):
        """Lightning Module for PoFTR."""
        super().__init__()
        self.config = config
        self.model = PoFTR(config=config)
        self.use_phys = config['poftr']['phys']['use_phys']
        self.data_module = data_module
        self.base_model = config['poftr']['proj']['base_model']
        self.loss = define_loss(self.base_model, config)
        self.n_val_plots = min(config['poftr']['train']['n_val_pairs_to_plot'], config['poftr']['train']['batch_size'])
        self.n_test_plots = min(config['poftr']['test']['n_test_pairs_to_plot'], config['poftr']['train']['batch_size'])
        self.pretrained_ckpt = Path(config['poftr']['pretrained_ckpt'])
        self.prior_ablation = config['poftr']['test']['prior_ablation']
        self.val_plotted_pairs = 0

        self.spvs_map = {
            'xoftr': {
                'coarse': supervise_coarse_xoftr,
                'fine': supervise_fine_xoftr
            },
            'loftr': {
                'coarse': supervise_coarse,
                'fine': supervise_fine
            },
            'aspanformer': {
                'coarse': supervise_coarse,
                'fine': supervise_fine
            }
        }

        self._train_outputs = []
        self._val_outputs = []
        self._test_outputs = []


    @property
    def mlflow_client(self):
        """Returns the MlflowClient if using MLFlowLogger, else None."""
        if isinstance(self.logger, MLFlowLogger):
            return self.logger.experiment
        return None

    @property
    def mlflow_run_id(self):
        """Returns the current run_id if using MLFlowLogger, else None."""
        if isinstance(self.logger, MLFlowLogger):
            return self.logger.run_id
        return None

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = build_optimizer(self, self.config['poftr'])
        scheduler = build_scheduler(self.config['poftr'], optimizer)
        return [optimizer], [scheduler]

    def _set_frozen_state(self, freeze_backbone: bool):
        # 1. Set global backbone state
        for param in self.model.model.backbone.parameters():
            param.requires_grad = not freeze_backbone

        # 2. Rescue SFT modules (keep them trainable)
        if freeze_backbone:
            for name, module in self.model.model.backbone.named_modules():
                # We identify SFT modules by the string 'sft' in their name
                if 'sft' in name:
                    for param in module.parameters():
                        param.requires_grad = True

    @rank_zero_only
    def on_train_start(self):

        # Freeze backbone for first 5 epochs
        if self.use_phys:
            if self.current_epoch < 5:
                self._set_frozen_state(freeze_backbone=True)
                if self.trainer.is_global_zero:
                    logger.info(f"STARTUP: Backbone Frozen until epoch 5")

        to_log = {
            'machine': self.config['poftr']['run']['run_platform'],
            'dataset': self.config['poftr']['data']['dataset_version'],
            'ablation': self.config['poftr']['data']['ablation_version'],
            'distribution': self.config['poftr']['sim']['distribution_type'],
            'train_ds_size': self.data_module.train_dataset.__len__(),
            'val_ds_size': self.data_module.val_dataset.__len__(),
            'base_model': self.config['poftr']['proj']['base_model'],
            'batch_size': self.config['poftr']['train']['batch_size'],
            'initial_lr': self.config['poftr']['train']['initial_lr'],
            'seed': self.config['poftr']['data']['master_seed'],
            'weight_decay': self.config['poftr']['train']['weight_decay'],
            'image_aug': self.config['poftr']['data']['image_aug_level'],
            'phys_aug': self.config['poftr']['data']['phys_aug_level'],
            'scheduler': self.config['poftr']['train']['scheduler'],
            'fusion_type': self.config['poftr']['phys']['fusion_type'],
            "phys_config": "TFF",
            'use_phys': self.config['poftr']['phys']['use_phys'],
            'use_phys_input': self.config['poftr']['phys']['inject_input_sft'],
            'use_phys_coarse': self.config['poftr']['phys']['inject_coarse_sft'],
            'use_phys_fine': self.config['poftr']['phys']['inject_fine_sft'],
            'learnable_sft': self.config['poftr']['sft']['learnable_scale'],
            'initial_sft_scale': self.config['poftr']['sft']['initial_scale'],
            'bottleneck_sft': self.config['poftr']['sft']['bottleneck_dim'],
            'dropout_sft': self.config['poftr']['sft']['dropout_p'],
        }
        self.logger.log_hyperparams(to_log)

    def _compute_metrics(self, batch):
        """
        Computes planar metrics, assigns to batch (for plotting),
        and returns them for aggregation.
        """
        with torch.no_grad():
            # Use detailed thresholds for Val/Test analysis
            metrics = compute_planar_metrics(
                batch,
                thresholds=[1, 2, 3, 5]
            )

        # Assign to batch so 'make_matching_figures' can access them
        batch['metrics'] = metrics

        # Return the metrics dict (scalars)
        return metrics

    def _compute_metrics_raw(self, batch):
        """
        Computes planar metrics, per image pair, assigns to batch (for model testing),
        """
        with torch.no_grad():
            # Use detailed thresholds for Val/Test analysis
            metrics = compute_planar_metrics_raw(
                batch,
                thresholds=[1, 2, 3, 5]
            )

        # Assign to batch so 'make_matching_figures' can access them
        batch['metrics'] = metrics

        # Return the metrics dict (scalars)
        return metrics

    def _trainval_inference(self, batch):
        # Get the specific strategies for the current base_model
        strategy = self.spvs_map.get(self.base_model)
        spvs_config = self.config['method'] if self.base_model == 'xoftr' else self.config['poftr']
        if not strategy:
            raise ValueError(f"Invalid base_model: {self.base_model}")

        # Execution
        strategy['coarse'](batch, spvs_config)

        self.model(batch)

        strategy['fine'](batch, spvs_config)

        self.loss(batch)

    def on_train_epoch_start(self):
        """
        Handles the transition at Epoch 5.
        """
        if self.use_phys:
            if self.current_epoch == 5:
                self._set_frozen_state(freeze_backbone=False)
                if self.trainer.is_global_zero:
                    logger.info(f"CURRICULUM: Unfreezing Backbone at Epoch 5!")

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        with torch.no_grad():
            metrics = compute_planar_metrics(batch, thresholds=[3])
        batch['metrics'] = metrics

        # Logging: Guard against NaNs in the progress bar
        mma_3 = metrics['MMA@3']
        if np.isnan(mma_3): mma_3 = 0.0

        self.log("train_step/mma_3", mma_3,
                 on_step=True, on_epoch=False, prog_bar=True, logger=True)

        for k, v in batch['loss_scalars'].items():

            self.log(f"train_step/{k}",
                     v,
                     on_step=True,
                     on_epoch=False,
                     logger=True
                     )

            self.log(f"train_epoch/{k}",
                     v,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=(k == 'loss'),
                     sync_dist=True
                     )

        # Plotting (Safe version)
        if batch_idx == 0 and self.config['poftr']['train']['enable_plotting'] and self.trainer.is_global_zero:
            figures = make_matching_figures(batch, self.config['poftr'], self.n_val_plots,
                                            self.config['poftr']['train']['plot_mode'])
            for k, v in figures.items():
                if self.mlflow_client is not None:
                    for plot_idx, fig in enumerate(v):
                        self.mlflow_client.log_figure(self.mlflow_run_id, fig,
                                                      f"figures/{k}/train/epoch_{self.current_epoch}/{plot_idx}.png")
                        plt.close(fig)  # Close specific figure

        return {'loss': batch['loss']}


    def on_train_epoch_end(self):
        if self.trainer.optimizers:
            lr = torch.tensor(self.trainer.optimizers[0].param_groups[0]["lr"], dtype=torch.float32, device=self.device)
            self.log('lr', lr, prog_bar=False, sync_dist=False)

    def on_validation_epoch_start(self) -> None:
        self.val_plotted_pairs = 0

    def validation_step(self, batch, batch_idx):
        # 1) Forward pass, supervision
        self._trainval_inference(batch)

        # 2) Compute and retrieve metrics
        metrics = self._compute_metrics(batch)

        # 3) Plotting Logic
        # handle case where num_batches might be 0 (sanity checks)
        num_batches = self.trainer.num_val_batches[0] if self.trainer.num_val_batches else 0

        if num_batches > 0:
            plot_prob = min(1.0, (self.n_val_plots * 1.2) / num_batches)
            should_plot = (self.global_rank == 0
                           and self.mlflow_client is not None
                           and self.config['poftr']['train']['enable_plotting']
                           and self.val_plotted_pairs < self.n_val_plots
                           and random.random() < plot_prob)

            if should_plot:
                figure = make_matching_figures(batch, self.config['poftr'], num_plots=1,
                                               mode=self.config['poftr']['train']['plot_mode'])

                for k, v in figure.items():
                    for plot_idx, fig in enumerate(v):
                        self.mlflow_client.log_figure(
                            self.mlflow_run_id,
                            fig,
                            f"figures/{k}/val/epoch_{self.current_epoch}/{self.val_plotted_pairs}.png"
                        )
                        plt.close(fig)  # Explicitly close specific fig

                plt.close('all')  # Catch-all close to be safe
                self.val_plotted_pairs += 1

        self._val_outputs.append({
            'metrics': metrics,
            'loss_scalars': batch['loss_scalars'],
        })

    def on_validation_epoch_end(self):
        # 1) Drain buffer
        outputs, self._val_outputs = self._val_outputs, []
        if not outputs: return

        # 2) Aggregate Metrics (FIX 2: Convert Tensors to floats)
        agg_metrics = defaultdict(list)
        for o in outputs:
            for k, v in o['metrics'].items():
                # Safety check: detach from GPU immediately
                val = v.item() if isinstance(v, torch.Tensor) else v
                agg_metrics[k].append(val)

        val_metrics = {}
        for k, v_list in agg_metrics.items():
            # DDP Sync
            global_list = flattenList(all_gather(v_list))

            # Double safety: ensure list contents are floats before Numpy
            clean_list = [x.item() if isinstance(x, torch.Tensor) else x for x in global_list]

            # Unified NaN handling
            if len(clean_list) > 0:
                val_metrics[k] = np.nanmean(clean_list)
                if np.isnan(val_metrics[k]): val_metrics[k] = 0.0
            else:
                val_metrics[k] = 0.0

        # 3) Aggregate Loss
        _loss_scalars = [o['loss_scalars'] for o in outputs]
        loss_scalars = {}
        if _loss_scalars:
            for k in _loss_scalars[0].keys():
                # Collect local (convert to float/item for safety)
                vals = [ls[k].item() if isinstance(ls[k], torch.Tensor) else ls[k] for ls in _loss_scalars]
                # Sync global
                loss_scalars[k] = flattenList(all_gather(vals))

        # Log Loss
        for k, v in loss_scalars.items():
            # v is now a list of floats, safe to make a new tensor
            mean_v = torch.tensor(v, dtype=torch.float32).mean().to(self.device)
            self.log(f'val_epoch/{k}', mean_v, prog_bar=(k == 'loss'), sync_dist=True)

        # 4) Log Metrics
        for k, v in val_metrics.items():
            k_log = k.replace('@', '_')
            # Add Pose Success to progress bar so you can see it in real-time
            is_prog = k in ['MMA@3', 'MRE', 'Pose_Success_10px']
            self.log(f'val/{k_log}', v, prog_bar=is_prog, sync_dist=True)

        # 5) Log Physical Metrics
        phys_metrics = self._calc_phys_metrics()
        for k, v in phys_metrics.items():
            self.log(k, v, prog_bar=False, sync_dist=True)

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            self._log_summary(val_metrics, type='Validation')

    def on_test_epoch_start(self) -> None:
        self.test_plotted_pairs = 0
        self._test_outputs = []

    def _extract_phys(self, batch):
        """Extracts the physics channel from the 3-channel input [img, phys, mask]."""
        phys0 = batch['image0'][:, 1:2, :, :]
        phys1 = batch['image1'][:, 1:2, :, :]
        return phys0, phys1

    def _apply_and_stitch(self, batch, phys0, phys1):
        """Overwrites the physics channel and re-stitches the input tensor."""
        batch['image0'][:, 1:2, :, :] = phys0
        batch['image1'][:, 1:2, :, :] = phys1

    def _zero_priors(self, batch):
        phys0, phys1 = self._extract_phys(batch)
        self._apply_and_stitch(batch, torch.zeros_like(phys0), torch.zeros_like(phys1))

    def _noisy_priors(self, batch, std=1.5):
        phys0, phys1 = self._extract_phys(batch)
        p0 = phys0 + torch.randn_like(phys0) * std
        p1 = phys1 + torch.randn_like(phys1) * std
        self._apply_and_stitch(batch, p0, p1)

    def _shuffled_priors(self, batch):
        phys0, phys1 = self._extract_phys(batch)
        def shuffle_tensor(p):
            b, c, h, w = p.shape
            bs = 16  # 16x16 blocks as discussed for ECCV
            p = p.view(b, c, h // bs, bs, w // bs, bs).permute(0, 1, 2, 4, 3, 5).reshape(b, c, -1, bs, bs)
            for i in range(b):
                idx = torch.randperm(p.size(2))
                p[i] = p[i, :, idx]
            return p.view(b, c, h // bs, w // bs, bs, bs).permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)

        self._apply_and_stitch(batch, shuffle_tensor(phys0), shuffle_tensor(phys1))

    def _mismatched_priors(self, batch):
        phys0, phys1 = self._extract_phys(batch)
        # Roll the batch so Image i gets the Prior from Image i+1
        self._apply_and_stitch(batch, torch.roll(phys0, 1, 0), torch.roll(phys1, 1, 0))

    def _prior_ablation_step(self, batch):
        if self.prior_ablation == 'zeroed':self._zero_priors(batch)
        elif self.prior_ablation == 'noised': self._noisy_priors(batch)
        elif self.prior_ablation == 'shuffled': self._shuffled_priors(batch)
        elif self.prior_ablation == 'mismatched': self._mismatched_priors(batch)
        else: raise ValueError(f"Invalid prior_ablation: {self.prior_ablation}")


    def test_step(self, batch, batch_idx):
        # 0) prior ablation:
        if self.prior_ablation:
            self._prior_ablation_step(batch)

        # 1) Inference
        self._trainval_inference(batch)

        # 2) Compute Metrics
        # Returns dict of SCALARS
        metrics = self._compute_metrics(batch)

        # 3) Log Loss (Accumulate for epoch)
        for k, v in batch['loss_scalars'].items():
            self.log(
                f"test/{k}", v,
                on_step=False, on_epoch=True, prog_bar=(k == 'loss'),
                logger=True, sync_dist=True
            )

        # 4) Plotting Logic (Standard)
        nb = self.trainer.num_test_batches
        num_batches = nb[0] if isinstance(nb, list) and len(nb) > 0 else (nb if isinstance(nb, int) else 0)

        if num_batches > 0:
            plot_prob = min(1.0, (self.n_test_plots * 1.2) / num_batches)
            should_plot = (self.global_rank == 0
                           and self.mlflow_client is not None
                           and self.config['poftr']['test']['enable_plotting']
                           and self.test_plotted_pairs < self.n_test_plots
                           and random.random() < plot_prob)

            if should_plot:
                figure = make_matching_figures(batch, self.config['poftr'], num_plots=1,
                                               mode=self.config['poftr']['test']['plot_mode'])
                for k, v in figure.items():
                    for plot_idx, fig in enumerate(v):
                        self.mlflow_client.log_figure(
                            self.mlflow_run_id,
                            fig,
                            f"figures/{k}/test/epoch_{self.current_epoch}/{self.test_plotted_pairs}.png"
                        )
                        plt.close(fig)
                plt.close('all')
                self.test_plotted_pairs += 1

        # 5) Buffer Results
        # We append the scalar dict to the list
        self._test_outputs.append({'metrics': metrics})

    def on_test_epoch_end(self):
        outputs, self._test_outputs = self._test_outputs, []
        if not outputs: return

        agg_metrics = defaultdict(list)
        for o in outputs:
            for k, v in o['metrics'].items():
                agg_metrics[k].append(v)

        test_metrics = {}
        for k, v_list in agg_metrics.items():
            global_list = flattenList(all_gather(v_list))
            # Consistent NaN handling
            if len(global_list) > 0:
                test_metrics[k] = np.nanmean(global_list)
                if np.isnan(test_metrics[k]): test_metrics[k] = 0.0
            else:
                test_metrics[k] = 0.0

        # MLflow & Lightning Logging
        if self.trainer.is_global_zero and self.mlflow_client is not None:
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.ckpt_path and self.trainer.sanity_checking: cur_epoch = -1
            for k, v in test_metrics.items():
                k_log = k.replace('@', '_')
                val_to_log = v.item() if isinstance(v, torch.Tensor) else v
                self.mlflow_client.log_metric(self.mlflow_run_id, f"test/{k_log}", val_to_log, step=cur_epoch)

        for k, v in test_metrics.items():
            k_log = k.replace('@', '_')
            self.log(f'test_{k_log}', v, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            self._log_summary(test_metrics, type='Test')

    def _calc_phys_metrics(self):
        """
        Returns:
            - lambda: 0-1 (Natural). Permission to use physics.
            - activity: 0-1 (Normalized). Effort in processing physics.
        """
        metrics = {}
        if not self.config['poftr']['phys']['use_phys']:
            return metrics

        # FIX: Added .model.model to reach the backbone
        backbone = self.model.model.backbone

        stages = ['input', 'coarse', 'fine']

        # HEURISTIC: A 'healthy' Conv layer usually has weight std between 0.02 and 0.1.
        MAX_EXPECTED_STD = 0.2

        for stage in stages:
            sft_name = f'sft_{stage}'

            if hasattr(backbone, sft_name):
                sft_module = getattr(backbone, sft_name)

                # --- 1. Lambda (The Valve) ---
                if hasattr(sft_module, 'get_modulation_strength'):
                    val = sft_module.get_modulation_strength()
                    if isinstance(val, torch.Tensor): val = val.item()
                    metrics[f'phys_params/{stage}_lambda'] = val

                # --- 2. Activity (The Engine) ---
                weight_stds = []
                # FIX: Access .mlp inside the SFT module
                for name, param in sft_module.mlp.named_parameters():
                    if 'weight' in name and param.requires_grad:
                        weight_stds.append(param.data.std())

                if weight_stds:
                    raw_std = torch.stack(weight_stds).mean().item()
                    # Normalize
                    norm_activity = min(max(raw_std / MAX_EXPECTED_STD, 0.0), 1.0)
                    metrics[f'phys_params/{stage}_activity_norm'] = norm_activity

        return metrics

    def _log_summary(self, metrics, type='Validation'):
        logger.info(f"\n-----------{type} summary-----------\n")
        if type == 'Validation':
            logger.info(f"Epoch: {self.current_epoch}")

        # 1. Define the keys we actually compute now
        # We check if they exist in the dict to be safe
        thresholds = [1, 3, 5, 10]
        mma_keys = [f"MMA@{t}" for t in thresholds if f"MMA@{t}" in metrics]
        geo_keys = ["MRE", "Corner_Error", "Num_Matches", "Num_Inliers"]

        # 2. Helper to build tables
        def build_table(keys, metrics_dict):
            # Filter out keys that might be missing
            valid_keys = [k for k in keys if k in metrics_dict]
            if not valid_keys:
                return " [No metrics available]"

            # Dynamic column width based on key length
            col_widths = [max(len(k), 8) for k in valid_keys]

            # Create Separator
            sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"

            # Create Header
            header = "|" + "|".join(f" {k:^{w}} " for k, w in zip(valid_keys, col_widths)) + "|"

            # Create Values
            vals = []
            for k, w in zip(valid_keys, col_widths):
                val = metrics_dict[k]
                if isinstance(val, (float, np.float32, np.float64)):
                    vals.append(f" {val:^{w}.4f} ")
                else:
                    vals.append(f" {val:^{w}} ")
            val_str = "|" + "|".join(vals) + "|"

            return f"{sep}\n{header}\n{sep}\n{val_str}\n{sep}"

        # 3. Log the tables
        logger.info(
            "\n------ Mean Matching Accuracy (MMA) ------\n"
            + build_table(mma_keys, metrics)
        )

        logger.info(
            "\n------ Geometric Errors & Stats ------\n"
            + build_table(geo_keys, metrics)
        )