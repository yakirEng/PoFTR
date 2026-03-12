from yacs.config import CfgNode as CN
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import sys

from src.configs.loftr_configs import get_loftr_defaults
from src.configs.aspan_configs import get_aspan_defaults
from src.configs.xoftr_configs import get_xoftr_defaults


_CN = CN()

##############  ↓  PoFTR Pipeline  ↓  ##############
_CN.PROJ = CN()
_CN.PROJ.base_model= 'aspanformer' # options: ['loftr', 'aspanformer']
_CN.PROJ.pretrained_ckpt = ""
_CN.PROJ.cwd = ""  # fill in finalize step
_CN.PROJ.data_path = ""  # fill in finalize step


##############  ↓  Sensor / Intrinsics  ↓  ##############
_CN.SENSOR = CN()
_CN.SENSOR.fx = 1117.65            # focal length (px) horizontal
_CN.SENSOR.fy = 1117.65            # focal length (px) vertical
_CN.SENSOR.cx = 320.0              # principal point x
_CN.SENSOR.cy = 256.0              # principal point y
_CN.SENSOR.skew = 0.0              # skew
_CN.SENSOR.width  = 640            # image width (px)
_CN.SENSOR.height = 512            # image height (px)

# more properties
_CN.SENSOR.focal_length_mm = 19.0      # focal length in mm
_CN.SENSOR.pixel_pitch_mm  = 0.017     # pixel pitch in mm (17 µm)
_CN.SENSOR.pixel_tolerance = 1.0       # pixel tolerance

def get_K_from_cfg(cfg) -> np.ndarray:
    """Build the 3x3 intrinsics matrix K from a CN that has SENSOR.* fields."""
    S = cfg.SENSOR if hasattr(cfg, "SENSOR") else cfg["sensor"]
    fx = S.fx if hasattr(S, "fx") else S["fx"]
    fy = S.fy if hasattr(S, "fy") else S["fy"]
    skew = S.skew if hasattr(S, "skew") else S["skew"]
    cx = S.cx if hasattr(S, "cx") else S["cx"]
    cy = S.cy if hasattr(S, "cy") else S["cy"]

    return np.array([
        [fx,  skew , cx],
        [0.0,   fy,   cy],
        [0.0,   0.0,    1.0]
    ], dtype=np.float32)

##############  ↓  Simulation Params  ↓  ##############
_CN.SIM = CN()
_CN.SIM.dataset_mode = True
_CN.SIM.sim_level = 2                     # 1=mild, 2=moderate, 3=strong

_CN.SIM.plane2sat = (500 * 10**3) / 2500  # 500km / 2.5km
_CN.SIM.scale = 1.0

# simulation distribution params
_CN.SIM.distribution_type = "truncnorm"  # options: "truncnorm", "mixture_norm"
_CN.SIM.truncated_mu = 0.5
_CN.SIM.truncated_sigma = 0.2
_CN.SIM.mixture_weights = [0.25, 0.50, 0.25]
_CN.SIM.mixture_mus   = [0.1, 0.5, 0.9]
_CN.SIM.mixture_sigmas    = [0.05, 0.1, 0.05]
_CN.SIM.num_calibration_samples = 1000

# Derived stds (fill during finalize step)
_CN.SIM.pitch_std = 10.0
_CN.SIM.roll_std  = 10.0
_CN.SIM.yaw_std   = 10.0
_CN.SIM.tx_std    = 10.0
_CN.SIM.ty_std    = 10.0
_CN.SIM.tz_std    = 10.0

# Reference levels (optional, handy for finalize code)
_CN.SIM.LEVELS = CN()
_CN.SIM.LEVELS.L1 = CN({"pitch": 0.0039, "roll": 0.0039, "yaw": 0.0039,
                        "tx": 0.1, "ty": 0.1, "tz": 0.1})
_CN.SIM.LEVELS.L2 = CN({"pitch": 2.0,    "roll": 2.0,    "yaw": 2.0,
                        "tx": 10.0, "ty": 10.0, "tz": 10.0})
_CN.SIM.LEVELS.L3 = CN({"pitch": 7.0,   "roll": 7.0,   "yaw": 7.0,
                        "tx": 60.0, "ty": 60.0, "tz": 30.0})



##############  ↓  Physical Model Config  ↓  ##############
_CN.PHYS = CN()
_CN.PHYS.use_phys = False
_CN.PHYS.inject_input_sft = False if not _CN.PHYS.use_phys else True
_CN.PHYS.inject_coarse_sft = False if not _CN.PHYS.use_phys else True
_CN.PHYS.inject_fine_sft = False if not _CN.PHYS.use_phys else True
_CN.PHYS.fusion_type      = "sft"   # options: "sft" | "concat"
_CN.PHYS.sft_lambda     = 1.0       # only for sft fusion
_CN.PHYS.dtype        = "numpy"   # options: "numpy" | "torch"


_CN.PHYS.coeff_path   = ""        # fill in finalize, e.g. f"{cfg.project.cwd}/src/physical_model/coeff/coefficients_9um.npz"

##############  ↓  Backbone Config  ↓  ##############
_CN.BACKBONE = CN()
_CN.BACKBONE.INITIAL_DIM = 128
_CN.BACKBONE.BLOCK_DIMS = [128, 196, 256]

##############  ↓  SFT Config  ↓  ##############

_CN.SFT = CN()
_CN.SFT.bottleneck_dim = 64
_CN.SFT.dropout_p = 0.0
_CN.SFT.learnable_scale = True
_CN.SFT.initial_scale = 0.05

##############  ↓  Data Config  ↓  ##############
_CN.DATA = CN()
_CN.DATA.wls = ["9um", "11um", "pan"]
_CN.DATA.train_size = 26090
_CN.DATA.val_size   = 4605
_CN.DATA.test_size  = 5417
_CN.DATA.sample_shape = [128, 128]
_CN.DATA.width  = 256
_CN.DATA.height = 256
_CN.DATA.image_shape = [256, 256]
_CN.DATA.fine_scale = 2
_CN.DATA.coarse_scale = 8
_CN.DATA.fine_window_size = 8
_CN.DATA.mono_or_color = "simulator"           # "simulator" | "color"
_CN.DATA.min_clip = 0
_CN.DATA.max_clip = 65535                 # simulator default; override if color
_CN.DATA.dynamic_range = 65535            # usually set to max_clip in finalize
_CN.DATA.subs_num = 1
_CN.DATA.overscan_scale = 3
_CN.DATA.image_aug_level = 'gentle'  # gentle, standard
_CN.DATA.phys_aug_level = 'easy'  # easy, medium, strong

# splits / loader
_CN.DATA.restart_dataset = True
_CN.DATA.train_ratio = 0.85
_CN.DATA.val_ratio   = 0.15
_CN.DATA.test_ratio  = 0.15

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
_CN.DATA.dataset_version = "11um_pan"   # options: "9um_11um", "11um_pan", "9um_pan"
_CN.DATA.ablation_version = "upper_bound"   # options: "upper_bound", "zeroed_priors", "noised_priors", "None"

##############  ↓  Train Config  ↓  ##############
_CN.TRAIN = CN()
_CN.TRAIN.max_epochs = 50

_CN.TRAIN.logger = "mlflow"                 # ["mlflow","tensorboard"]
_CN.TRAIN.weight_decay = 1e-2
_CN.TRAIN.batch_size = 8
_CN.TRAIN.val_interval = 1
_CN.TRAIN.log_dir = "logs"

# distributed
_CN.TRAIN.world_size = int(os.environ.get("WORLD_SIZE", 1))

# profiler
_CN.TRAIN.profiler_name = "pytorch"

# optimizer
_CN.TRAIN.optimizer = "adamw"               # ["adam","adamw"]
_CN.TRAIN.initial_lr = 1e-4

# warmup
_CN.TRAIN.warmup_ratio = 0.1
_CN.TRAIN.warmup_epochs = 5

# scheduler
_CN.TRAIN.scheduler = "CosineAnnealing"     # ["MultiStepLR","CosineAnnealing","ExponentialLR"]
_CN.TRAIN.scheduler_interval = "epoch"      # ["epoch","step"]
_CN.TRAIN.eta_min = 1e-6

# plotting
_CN.TRAIN.enable_plotting = True
_CN.TRAIN.n_val_pairs_to_plot = 8
_CN.TRAIN.plot_mode = "evaluation"          # ["evaluation","confidence"]
_CN.TRAIN.plot_matches_alpha = "dynamic"
_CN.TRAIN.plot_every_n_epochs = 1

# geometry / pose
_CN.TRAIN.epi_err_thr = [1e-5, 1e-4, 3e-4]
_CN.TRAIN.pose_geo_model = "E"              # ["E","F","H"]
_CN.TRAIN.pose_estimation_method = "RANSAC" # ["RANSAC","DEGENSAC","MAGSAC"]
_CN.TRAIN.ransac_pixel_thr = 0.5
_CN.TRAIN.ransac_conf = 0.99999
_CN.TRAIN.ransac_max_iters = 10000
_CN.TRAIN.use_magsacpp = False

# sampling
_CN.TRAIN.data_sampler = "scene_balance"    # ["scene_balance","random","normal"]
_CN.TRAIN.n_samples_per_subset = 200
_CN.TRAIN.sb_subset_sample_replacement = True
_CN.TRAIN.sb_subset_shuffle = True
_CN.TRAIN.sb_repeat = 1
_CN.TRAIN.rdm_replacement = True
_CN.TRAIN.rdm_num_samples = None

# optimization safety
_CN.TRAIN.gradient_clipping = 0.5

# ────────── Test ──────────
_CN.TEST = CN()
_CN.TEST.enable_plotting = True
_CN.TEST.n_test_pairs_to_plot = 8
_CN.TEST.plot_mode = "evaluation"          # ["evaluation","confidence"]
_CN.TEST.prior_ablation = None # options: 'zeroed', 'noised', shuffled', 'mismatched', None

# ────────── Run/System ──────────
_CN.RUN = CN()
_CN.RUN.run_platform = ""            # ["local","hpc"]
_CN.RUN.sanity_check = None
_CN.RUN.sanity_dataset_size = None
_CN.RUN.num_gpus = 1                 # fill at runtime if desired
_CN.RUN.num_nodes = 0
_CN.RUN.multi_nodes = None

_CN.RUN.accelerator = "gpu"
_CN.RUN.tracking_uri = ""
_CN.RUN.experiment_name = ""
_CN.RUN.run_name = ""
_CN.RUN.num_workers = 1           # set in finalize per platform
_CN.RUN.prefetch_factor = 4
_CN.RUN.log_every_n_steps = 1

def _get_ckpt_path(cfg):

    base_model = cfg.PROJ.base_model
    if base_model == "loftr":
        return f"{cfg.PROJ.cwd}/checkpoints/weights_loftr/loftr_outdoor.ckpt"
    elif cfg.PROJ.base_model == "aspanformer":
        return f"{cfg.PROJ.cwd}/checkpoints/weights_aspanformer/outdoor.ckpt"
    elif cfg.PROJ.base_model == "xoftr":
        return f"{cfg.PROJ.cwd}/checkpoints/weights_xoftr/weights_xoftr_640.ckpt"
    else:
        raise ValueError(f"Unknown base_model: {base_model}")

def finalize_cfg(cfg):
    """Populate derived/run-time fields (CN version of your __post_init__)."""
    finalize_sim(cfg)
    cfg.defrost()

    # 1) Run platform & sanity toggle
    cfg.RUN.run_platform = 'hpc' if os.environ.get('SLURM_JOB_ID') else 'local'
    is_hpc = (cfg.RUN.run_platform == 'hpc')

    # 2) Repo root (auto-detected from this file's location: src/configs/poftr_configs.py)
    cfg.PROJ.cwd = str(Path(__file__).resolve().parents[2])

    cfg.PROJ.data_path = f"{cfg.PROJ.cwd}/data"

    cfg.pretrained_ckpt = _get_ckpt_path(cfg)
    dataset_version = cfg.DATA.dataset_version
    phys_str = "phys" if cfg.PHYS.use_phys else "no_phys"
    base_model = cfg.PROJ.base_model
    cfg.ckpt_path = f"{cfg.PROJ.cwd}/checkpoints/best/{dataset_version}/{base_model}/{phys_str}"

    cfg.RUN.tracking_uri = (
        "http://10.26.36.96:5000" if is_hpc else "http://127.0.0.1:5000"
    )
    cfg.PHYS.coeff_path = (
        f"{cfg.PROJ.cwd}/src/dataset/physical_model/coeff/coefficients_9um.npz"
    )

    cfg.RUN.num_workers = 8 if is_hpc else 0
    cfg.RUN.prefetch_factor = 4 if is_hpc else None
    cfg.RUN.num_nodes   = int(os.environ.get("SLURM_JOB_NUM_NODES", 1)) if is_hpc else 1
    cfg.RUN.multi_nodes = is_hpc

    # If you already have a builder, use it; else inline the simple one:
    def _build_experiment_name(config):
        model_name = config.PROJ.base_model
        if config.DATA.ablation_version != "standard":
            dataset_name = f"{config.DATA.dataset_version}_{config.DATA.ablation_version}"
        else:
            dataset_name = config.DATA.dataset_version
        return f"{model_name}_{dataset_name}_comparison_fixed"

    cfg.RUN.experiment_name = _build_experiment_name(cfg)
    cfg.RUN.run_name = datetime.now().strftime("%d_%m__%H_%M")

    cfg.freeze()
    return cfg


# finalize the SIM section based on dataset_mode / sim_level
def finalize_sim(cfg):
    # pick level: force L2 when dataset_mode=True (like your dataclass)
    L = cfg.SIM.LEVELS
    sel = L.L3 if cfg.SIM.dataset_mode else {1: L.L1, 2: L.L2, 3: L.L3}[cfg.SIM.sim_level]

    cfg.defrost()
    cfg.SIM.pitch_std = float(sel.pitch)
    cfg.SIM.roll_std  = float(sel.roll)
    cfg.SIM.yaw_std   = float(sel.yaw)
    # translate position stds from meters to plane2sat scale
    p2s = float(cfg.SIM.plane2sat)
    cfg.SIM.tx_std = float(sel.tx) / p2s
    cfg.SIM.ty_std = float(sel.ty) / p2s
    cfg.SIM.tz_std = float(sel.tz) / p2s
    cfg.freeze()


def scale_sim(sim_config, scale: float):
    """
    Scales simulation parameters in a config object, whether it's
    a dictionary or a yacs-style config node.
    """

    params_to_scale = [
        'pitch_std', 'roll_std', 'yaw_std',
        'tx_std', 'ty_std', 'tz_std'
    ]

    # Modify parameters based on the type of sim_config
    if isinstance(sim_config, dict):
        # Handle dictionary-style access
        for param in params_to_scale:
            if param in sim_config:
                sim_config[param] *= scale
            else:
                print(f"Warning: '{param}' not found in sim_config dict.", file=sys.stderr)
    else:
        # Handle attribute-style access
        for param in params_to_scale:
            if hasattr(sim_config, param):
                current_value = getattr(sim_config, param)
                setattr(sim_config, param, current_value * scale)
            else:
                print(f"Warning: '{param}' not found in sim_config object.", file=sys.stderr)

    return sim_config


def get_poftr_defaults():
    return _CN.clone()

def tune_hparams(cfg):
     """
     tunning hyperparameters
     dataset_version, batch size, learning rate, weight decay, seed,
     :return:
     """
     cfg.defrost()

     # run:
     cfg.RUN.sanity_check = False
     cfg.RUN.sanity_dataset_size = 5000
     cfg.RUN.run_type = 'sanity' if cfg.RUN.sanity_check else 'full'

     # model:
     cfg.PROJ.base_model = "xoftr"  # options: ['loftr', 'aspanformer', 'xoftr']

     # data:
     cfg.DATA.dataset_version = "9um_pan"   # options: "9um_11um", "11um_pan", "9um_pan"
     cfg.DATA.ablation_version = "standard"
     cfg.SIM.distribution_type = "truncnorm"

     cfg.DATA.image_aug_level = 'standard' # gentle, standard
     cfg.DATA.phys_aug_level = 'medium' # easy, medium, strong

     # phys:
     cfg.PHYS.use_phys = True
     cfg.PHYS.inject_input_sft = True
     cfg.PHYS.inject_coarse_sft = True
     cfg.PHYS.inject_fine_sft = True

     # seed:
     cfg.DATA.master_seed = 999
     cfg.SFT.bottleneck_dim = 128 if cfg.PHYS.use_phys else 0
     cfg.SFT.dropout_p = 0.2 if cfg.PHYS.use_phys else 0.0
     cfg.SFT.learnable_scale = True
     cfg.SFT.initial_scale = 1.0

     # testing:
     cfg.TEST.prior_ablation = None # options: 'zeroed', 'noised', shuffled', 'mismatched', None

     # training:
     cfg.TRAIN.max_epochs = 20
     cfg.TRAIN.initial_lr = 1e-4 # 1e-4
     cfg.TRAIN.weight_decay = 1e-2
     cfg.TRAIN.batch_size = 16 if cfg.RUN.run_platform == 'hpc' else 2


def get_method_config(base_model: str) -> CN:
    if base_model == "loftr":
        method = get_loftr_defaults()
    elif base_model == "aspanformer":
        method = get_aspan_defaults()
    elif base_model == "xoftr":
        method = get_xoftr_defaults()
    else:
        raise ValueError(f"Unknown base_model: {base_model}")
    return method

def get_config():
    poftr_config = get_poftr_defaults()
    tune_hparams(poftr_config) # hyperparameter tuning
    finalize_cfg(poftr_config) # populate derived fields

    base_model = poftr_config.PROJ.base_model

    method = get_method_config(base_model)

    # 3) build one run config with namespaces
    run_cfg = CN()
    run_cfg.POFTR = poftr_config.clone()
    run_cfg.METHOD = method.clone()  # e.g., contains .LOFTR or .ASPAN subtree

    run_cfg.defrost()

    run_cfg.MODEL_NAME = base_model
    run_cfg.freeze()

    return run_cfg