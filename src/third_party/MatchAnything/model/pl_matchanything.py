import random
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from pytorch_lightning.loggers import MLFlowLogger
from loguru import logger

# --- Imports from your project structure ---
from src.utils.planar_metrics import compute_planar_metrics
from src.utils.misc import flattenList
from src.utils.plotting import make_matching_figures
from src.utils.comm import all_gather
from src.third_party.MatchAnything.model.match_anything import  MatchAnythingHFWrapper

# --- NEW IMPORT: Restore Supervision for GT generation ---
from src.utils.supervise import supervise_coarse


class PL_MatchAnything(pl.LightningModule):
    def __init__(self, config, data_module):
        super().__init__()
        self.config = config
        self.data_module = data_module

        # 1. Initialize the Hugging Face Wrapper
        model_id = config.get('matchanything', {}).get('model_id', "zju-community/matchanything_eloftr")
        self.model = MatchAnythingHFWrapper(model_id=model_id)

        # 2. Plotting / Logging Config
        self.n_val_plots = min(config['poftr']['train']['n_val_pairs_to_plot'], 4)
        self.n_test_plots = min(config['poftr']['test']['n_test_pairs_to_plot'], 4)
        self.val_plotted_pairs = 0
        self.test_plotted_pairs = 0

        self._val_outputs = []
        self._test_outputs = []

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

    def configure_optimizers(self):
        return None

    def _inference(self, batch):
        """
        Runs inference and prepares data for metrics/plotting.
        """
        # 1. Generate Ground Truth for Metrics
        #    'supervise_coarse' requires low-res masks (32x32).
        #    We generate them dynamically from your high-res 'pixel_mask' (256x256).

        with torch.no_grad():

            # Run Supervision (Now receives correct 32x32 masks)
            # This generates 'conf_matrix_gt' needed for "Feature_Recall" and plotting
            if 'mask0' in batch and 'mask1' in batch:
                supervise_coarse(batch, self.config['poftr'])
            else:
                raise ValueError("Missing 'mask0' or 'mask1' in batch. Please check your datamodule.")

        # 2. Forward Pass (Wrapper uses 'pixel_mask0' for fine-level filtering)
        outputs = self.model(batch)

        # 3. Extract and Flatten (Remove Batch Dim 1 -> N)
        kpts0 = outputs['mkpts0_f'].squeeze(0)  # (N, 2)
        kpts1 = outputs['mkpts1_f'].squeeze(0)  # (N, 2)
        conf = outputs['mconf'].squeeze(0)  # (N,)

        # 4. Generate 'm_bids' (Batch IDs)
        #    Since batch_size=1, all N matches belong to batch 0.
        num_matches = kpts0.shape[0]
        m_bids = torch.zeros(num_matches, dtype=torch.long, device=kpts0.device)

        # 5. Update Batch with everything
        batch['mkpts0_f'] = kpts0
        batch['mkpts1_f'] = kpts1
        batch['mconf'] = conf
        batch['m_bids'] = m_bids
        # Note: We don't update(outputs) here to avoid overwriting the manual keys above
        # if the wrapper outputs them differently.

    def _compute_metrics(self, batch):
        with torch.no_grad():
            metrics = compute_planar_metrics(
                batch,
                thresholds=[1, 2, 3, 5, 10]
            )
        batch['metrics'] = metrics
        return metrics

    # ... [Rest of Validation/Test Loops remain exactly the same] ...

    # Copying standard validation/test logic for completeness:
    def on_validation_epoch_start(self):
        self.val_plotted_pairs = 0
        self._val_outputs = []

    def validation_step(self, batch, batch_idx):
        self._inference(batch)
        metrics = self._compute_metrics(batch)

        # Plotting
        num_batches = self.trainer.num_val_batches[0] if self.trainer.num_val_batches else 0
        if num_batches > 0:
            should_plot = (self.val_plotted_pairs < self.n_val_plots)
            if should_plot and self.config['poftr']['train']['enable_plotting']:
                batch['loss_scalars'] = {}
                figures = make_matching_figures(batch, self.config['poftr'], num_plots=1,
                                                mode=self.config['poftr']['train']['plot_mode'])
                if self.mlflow_client:
                    for k, v in figures.items():
                        for plot_idx, fig in enumerate(v):
                            self.mlflow_client.log_figure(self.mlflow_run_id, fig,
                                                          f"figures/{k}/val/zero_shot/{self.val_plotted_pairs}.png")
                            plt.close(fig)
                plt.close('all')
                self.val_plotted_pairs += 1
        self._val_outputs.append({'metrics': metrics})

    def on_validation_epoch_end(self):
        self._aggregate_and_log(self._val_outputs, prefix='val')
        self._val_outputs = []

    def on_test_epoch_start(self):
        self.test_plotted_pairs = 0
        self._test_outputs = []

    def test_step(self, batch, batch_idx):
        self._inference(batch)
        metrics = self._compute_metrics(batch)

        # Plotting
        num_batches = self.trainer.num_test_batches
        num_batches = num_batches[0] if isinstance(num_batches, list) else (
            num_batches if isinstance(num_batches, int) else 0)

        if num_batches > 0:
            should_plot = (self.test_plotted_pairs < self.n_test_plots)
            if should_plot and self.config['poftr']['test']['enable_plotting']:
                batch['loss_scalars'] = {}
                figures = make_matching_figures(batch, self.config['poftr'], num_plots=1,
                                                mode=self.config['poftr']['test']['plot_mode'])
                if self.mlflow_client:
                    for k, v in figures.items():
                        for plot_idx, fig in enumerate(v):
                            self.mlflow_client.log_figure(self.mlflow_run_id, fig,
                                                          f"figures/{k}/test/zero_shot/{self.test_plotted_pairs}.png")
                            plt.close(fig)
                plt.close('all')
                self.test_plotted_pairs += 1
        self._test_outputs.append({'metrics': metrics})

    def on_test_epoch_end(self):
        self._aggregate_and_log(self._test_outputs, prefix='test')
        self._test_outputs = []

    def _aggregate_and_log(self, outputs, prefix):
        if not outputs: return
        agg_metrics = defaultdict(list)
        for o in outputs:
            for k, v in o['metrics'].items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                agg_metrics[k].append(val)

        final_metrics = {}
        for k, v_list in agg_metrics.items():
            global_list = flattenList(all_gather(v_list))
            clean_list = [x.item() if isinstance(x, torch.Tensor) else x for x in global_list]
            if len(clean_list) > 0:
                final_metrics[k] = np.nanmean(clean_list)
                if np.isnan(final_metrics[k]): final_metrics[k] = 0.0
            else:
                final_metrics[k] = 0.0

        if self.trainer.is_global_zero:
            self._log_summary(final_metrics, type=prefix.capitalize())
            if self.mlflow_client:
                for k, v in final_metrics.items():
                    k_log = k.replace('@', '_')
                    self.mlflow_client.log_metric(self.mlflow_run_id, f"{prefix}/{k_log}", v)

    def _log_summary(self, metrics, type='Test'):
        logger.info(f"\n----------- MatchAnything (Zero-Shot) {type} Summary -----------\n")
        thresholds = [1, 3, 5, 10]
        mma_keys = [f"MMA@{t}" for t in thresholds if f"MMA@{t}" in metrics]
        geo_keys = ["MRE", "Corner_Error_Mean", "Pose_Success_10px", "Num_Matches"]

        def log_section(title, keys):
            valid_keys = [k for k in keys if k in metrics]
            if valid_keys:
                logger.info(f"\n--- {title} ---")
                for k in valid_keys:
                    logger.info(f"{k}: {metrics[k]:.4f}")

        log_section("Mean Matching Accuracy", mma_keys)
        log_section("Geometric Stats", geo_keys)