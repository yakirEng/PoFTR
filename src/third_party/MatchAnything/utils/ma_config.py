from yacs.config import CfgNode as CN
import numpy as np
import os
from datetime import datetime
import sys

# We don't need LoFTR/ASpan defaults for MatchAnything,
# but we keep imports if your codebase checks them globally.
# from src.configs.loftr_configs import get_loftr_defaults
# from src.configs.aspan_configs import get_aspan_defaults

_CN = CN()

##############  ↓  MatchAnything Project  ↓  ##############
_CN.PROJ = CN()
_CN.PROJ.base_model = 'matchanything'
_CN.PROJ.pretrained_ckpt = ""  # Handled by HF Wrapper
_CN.PROJ.cwd = ""
_CN.PROJ.data_path = ""

##############  ↓  MatchAnything Specifics  ↓  ##############
_CN.MATCHANYTHING = CN()
# Use "zju-community/matchanything_eloftr" or "zju-community/matchanything_roma"
_CN.MATCHANYTHING.model_id = "zju-community/matchanything_eloftr"

##############  ↓  Sensor / Intrinsics (Keep for Metrics)  ↓  ##############
_CN.SENSOR = CN()
_CN.SENSOR.fx = 1117.65
_CN.SENSOR.fy = 1117.65
_CN.SENSOR.cx = 320.0
_CN.SENSOR.cy = 256.0
_CN.SENSOR.skew = 0.0
_CN.SENSOR.width = 640
_CN.SENSOR.height = 512
_CN.SENSOR.focal_length_mm = 19.0
_CN.SENSOR.pixel_pitch_mm = 0.017
_CN.SENSOR.pixel_tolerance = 1.0


def get_K_from_cfg(cfg) -> np.ndarray:
    """Build the 3x3 intrinsics matrix K."""
    S = cfg.SENSOR
    return np.array([
        [S.fx, S.skew, S.cx],
        [0.0, S.fy, S.cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


##############  ↓  Simulation Params (Keep for Dataset)  ↓  ##############
_CN.SIM = CN()
_CN.SIM.dataset_mode = True
_CN.SIM.sim_level = 2
_CN.SIM.plane2sat = (500 * 10 ** 3) / 2500
_CN.SIM.scale = 1.0
_CN.SIM.distribution_type = "truncnorm"
_CN.SIM.truncated_mu = 0.5
_CN.SIM.truncated_sigma = 0.2
_CN.SIM.mixture_weights = [0.25, 0.50, 0.25]
_CN.SIM.mixture_mus = [0.1, 0.5, 0.9]
_CN.SIM.mixture_sigmas = [0.05, 0.1, 0.05]
_CN.SIM.num_calibration_samples = 1000

# Derived stds
_CN.SIM.pitch_std = 10.0
_CN.SIM.roll_std = 10.0
_CN.SIM.yaw_std = 10.0
_CN.SIM.tx_std = 10.0
_CN.SIM.ty_std = 10.0
_CN.SIM.tz_std = 10.0

_CN.SIM.LEVELS = CN()
_CN.SIM.LEVELS.L1 = CN({"pitch": 0.0039, "roll": 0.0039, "yaw": 0.0039, "tx": 0.1, "ty": 0.1, "tz": 0.1})
_CN.SIM.LEVELS.L2 = CN({"pitch": 2.0, "roll": 2.0, "yaw": 2.0, "tx": 10.0, "ty": 10.0, "tz": 10.0})
_CN.SIM.LEVELS.L3 = CN({"pitch": 7.0, "roll": 7.0, "yaw": 7.0, "tx": 60.0, "ty": 60.0, "tz": 30.0})

##############  ↓  Physical Model Config (Disabled)  ↓  ##############
_CN.PHYS = CN()
_CN.PHYS.use_phys = False  # CRITICAL: MatchAnything is Zero-Shot
_CN.PHYS.inject_input_sft = False
_CN.PHYS.inject_coarse_sft = False
_CN.PHYS.inject_fine_sft = False
_CN.PHYS.fusion_type = "sft"
_CN.PHYS.sft_lambda = 1.0
_CN.PHYS.dtype = "numpy"
_CN.PHYS.coeff_path = ""

##############  ↓  Backbone Config (Placeholder)  ↓  ##############
_CN.BACKBONE = CN()
_CN.BACKBONE.INITIAL_DIM = 128
_CN.BACKBONE.BLOCK_DIMS = [128, 196, 256]

##############  ↓  SFT Config (Placeholder)  ↓  ##############
_CN.SFT = CN()
_CN.SFT.bottleneck_dim = 64
_CN.SFT.dropout_p = 0.0
_CN.SFT.learnable_scale = True
_CN.SFT.initial_scale = 0.05

##############  ↓  Data Config  ↓  ##############
_CN.DATA = CN()
_CN.DATA.wls = ["9um", "11um", "pan"]
_CN.DATA.train_size = 26090
_CN.DATA.val_size = 4605
_CN.DATA.test_size = 5417
_CN.DATA.sample_shape = [128, 128]
_CN.DATA.width = 256
_CN.DATA.height = 256
_CN.DATA.image_shape = [256, 256]
_CN.DATA.fine_scale = 2
_CN.DATA.coarse_scale = 8
_CN.DATA.fine_window_size = 8
_CN.DATA.mono_or_color = "simulator"
_CN.DATA.min_clip = 0
_CN.DATA.max_clip = 65535
_CN.DATA.dynamic_range = 65535
_CN.DATA.subs_num = 1
_CN.DATA.overscan_scale = 3
_CN.DATA.image_aug_level = 'gentle'
_CN.DATA.phys_aug_level = 'easy'

# splits / loader
_CN.DATA.restart_dataset = True
_CN.DATA.train_ratio = 0.85
_CN.DATA.val_ratio = 0.15
_CN.DATA.test_ratio = 0.15

# sampling / geometry
_CN.DATA.min_iou = 0.1
_CN.DATA.max_iou = 0.3
_CN.DATA.samples_num_per_image = 1
_CN.DATA.debug_ratio = 1.0
_CN.DATA.covisibility_tolerance = 0.1

# folds / seeds
_CN.DATA.k_folds = 5
_CN.DATA.master_seed = 42
_CN.DATA.debug_ratio = 1.0

# dataset version tag
_CN.DATA.dataset_version = "9um_11um"
_CN.DATA.ablation_version = "standard"

##############  ↓  Train Config (Mostly Placeholders for Eval)  ↓  ##############
_CN.TRAIN = CN()
_CN.TRAIN.max_epochs = 1
_CN.TRAIN.logger = "mlflow"
_CN.TRAIN.weight_decay = 0.0
_CN.TRAIN.batch_size = 1  # CRITICAL for Eval
_CN.TRAIN.val_interval = 1
_CN.TRAIN.log_dir = "logs"
_CN.TRAIN.world_size = int(os.environ.get("WORLD_SIZE", 1))
_CN.TRAIN.profiler_name = "pytorch"
_CN.TRAIN.optimizer = "adamw"
_CN.TRAIN.initial_lr = 0.0
_CN.TRAIN.warmup_ratio = 0.0
_CN.TRAIN.warmup_epochs = 0
_CN.TRAIN.scheduler = "CosineAnnealing"
_CN.TRAIN.scheduler_interval = "epoch"
_CN.TRAIN.eta_min = 0.0

# plotting
_CN.TRAIN.enable_plotting = True
_CN.TRAIN.n_val_pairs_to_plot = 8
_CN.TRAIN.plot_mode = "evaluation"
_CN.TRAIN.plot_matches_alpha = "dynamic"
_CN.TRAIN.plot_every_n_epochs = 1

# geometry / pose
_CN.TRAIN.epi_err_thr = [1e-5, 1e-4, 3e-4]
_CN.TRAIN.pose_geo_model = "E"
_CN.TRAIN.pose_estimation_method = "RANSAC"
_CN.TRAIN.ransac_pixel_thr = 0.5
_CN.TRAIN.ransac_conf = 0.99999
_CN.TRAIN.ransac_max_iters = 10000
_CN.TRAIN.use_magsacpp = False

# sampling
_CN.TRAIN.data_sampler = "scene_balance"
_CN.TRAIN.n_samples_per_subset = 200
_CN.TRAIN.sb_subset_sample_replacement = True
_CN.TRAIN.sb_subset_shuffle = True
_CN.TRAIN.sb_repeat = 1
_CN.TRAIN.rdm_replacement = True
_CN.TRAIN.rdm_num_samples = None
_CN.TRAIN.gradient_clipping = 0.5

# ────────── Test ──────────
_CN.TEST = CN()
_CN.TEST.enable_plotting = True
_CN.TEST.n_test_pairs_to_plot = 8
_CN.TEST.plot_mode = "evaluation"

# ────────── Run/System ──────────
_CN.RUN = CN()
_CN.RUN.run_platform = ""
_CN.RUN.sanity_check = None
_CN.RUN.sanity_dataset_size = None
_CN.RUN.num_gpus = 1
_CN.RUN.num_nodes = 0
_CN.RUN.multi_nodes = None
_CN.RUN.accelerator = "gpu"
_CN.RUN.tracking_uri = ""
_CN.RUN.experiment_name = ""
_CN.RUN.run_name = ""
_CN.RUN.num_workers = 1
_CN.RUN.prefetch_factor = 4
_CN.RUN.log_every_n_steps = 1


def finalize_cfg(cfg):
    """Populate derived/run-time fields."""
    finalize_sim(cfg)
    cfg.defrost()

    # 1) Run platform
    cfg.RUN.run_platform = 'hpc' if os.environ.get('SLURM_JOB_ID') else 'local'
    is_hpc = (cfg.RUN.run_platform == 'hpc')

    # 2) Paths
    cfg.PROJ.cwd = ("/home/ARO.local/yakirh/Projects/yakirs_thesis/thesis"
                    if is_hpc else r"F:/yakirs_thesis/thesis_code")
    cfg.PROJ.data_path = f"{cfg.PROJ.cwd}/data"

    # MatchAnything doesn't use local checkpoints for the model itself,
    # but we define a path to keep logic consistent.
    cfg.ckpt_path = f"{cfg.PROJ.cwd}/checkpoints/MatchAnything_ZeroShot"

    cfg.RUN.tracking_uri = (
        "http://10.26.36.96:5000" if is_hpc else "http://127.0.0.1:5000"
    )

    # Not used, but kept to prevent crashing if dataset checks it
    cfg.PHYS.coeff_path = (
        f"{cfg.PROJ.cwd}/src/dataset/physical_model/coeff/coefficients_9um.npz"
    )

    cfg.RUN.num_workers = 8 if is_hpc else 1
    cfg.RUN.num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1)) if is_hpc else 1
    cfg.RUN.multi_nodes = is_hpc

    # Experiment Name: Explicitly state ZeroShot comparison
    def _build_experiment_name(config):
        return f"MatchAnything_{config.DATA.dataset_version}_ZeroShot"

    cfg.RUN.experiment_name = _build_experiment_name(cfg)
    cfg.RUN.run_name = datetime.now().strftime("%d_%m__%H_%M")

    cfg.freeze()
    return cfg


def finalize_sim(cfg):
    L = cfg.SIM.LEVELS
    sel = L.L3 if cfg.SIM.dataset_mode else {1: L.L1, 2: L.L2, 3: L.L3}[cfg.SIM.sim_level]

    cfg.defrost()
    cfg.SIM.pitch_std = float(sel.pitch)
    cfg.SIM.roll_std = float(sel.roll)
    cfg.SIM.yaw_std = float(sel.yaw)
    p2s = float(cfg.SIM.plane2sat)
    cfg.SIM.tx_std = float(sel.tx) / p2s
    cfg.SIM.ty_std = float(sel.ty) / p2s
    cfg.SIM.tz_std = float(sel.tz) / p2s
    cfg.freeze()


def get_matchanything_defaults():
    return _CN.clone()


def tune_hparams(cfg):
    """
    Force Zero-Shot Hyperparameters for MatchAnything
    """
    cfg.defrost()

    # Run Mode
    cfg.RUN.sanity_check = False
    cfg.RUN.run_type = 'full'

    # Model
    cfg.PROJ.base_model = "matchanything"

    # Data Defaults
    cfg.DATA.dataset_version = "9um_11um"
    cfg.DATA.ablation_version = "standard"
    cfg.SIM.distribution_type = "truncnorm"

    # Zero-Shot: Disable Augmentations
    cfg.DATA.image_aug_level = 'gentle'
    cfg.DATA.phys_aug_level = 'easy'

    # Physics: HARD DISABLE
    cfg.PHYS.use_phys = False
    cfg.PHYS.inject_input_sft = False
    cfg.PHYS.inject_coarse_sft = False
    cfg.PHYS.inject_fine_sft = False

    # Seed
    cfg.DATA.master_seed = 42

    # Zero-Shot: Batch Size 1 is safest for eval
    cfg.TRAIN.batch_size = 1

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def get_config():
    """
    Returns the final configuration for MatchAnything Evaluation.
    Structure matches your codebase: run_cfg.PHYSAT, run_cfg.METHOD (empty)
    """
    physat_config = get_matchanything_defaults()
    tune_hparams(physat_config)
    finalize_cfg(physat_config)

    run_cfg = CN()
    run_cfg.PHYSAT = physat_config.clone()

    # Empty METHOD config since MatchAnything wrapper handles its own internals
    run_cfg.METHOD = CN()
    run_cfg.MODEL_NAME = "matchanything"

    run_cfg.freeze()
    return run_cfg