from yacs.config import CfgNode as CN
import numpy as np
import os
from datetime import datetime
from pathlib import Path

_CN = CN()

##############  ↓  PetitS Pipeline  ↓  ##############
_CN.PROJ = CN()
_CN.PROJ.base_model= 'petits'
_CN.PROJ.pretrained_ckpt = ""
_CN.PROJ.cwd = ""  # fill in finalize step
_CN.PROJ.data_path = ""  # fill in finalize step


##############  ↓  Physical Model Config  ↓  ##############
_CN.PHYS = CN()
_CN.PHYS.dtype        = "numpy"   # options: "numpy" | "torch"
_CN.PHYS.coeff_path   = ""        # fill in finalize, e.g. f"{cfg.project.cwd}/src/physical_model/coeff/coefficients_9um.npz"


##############  ↓  Data Config  ↓  ##############
_CN.DATA = CN()
_CN.DATA.wl = "11um"                     # "9um" | "11um"
_CN.DATA.wls = ["9um", "11um", "pan"]
_CN.DATA.sample_shape = [128, 128]
_CN.DATA.width  = 256
_CN.DATA.height = 256
_CN.DATA.image_shape = [256, 256]
_CN.DATA.fine_scale = 2
_CN.DATA.coarse_scale = 8
_CN.DATA.fine_window_size = 8
_CN.DATA.seed = 42
_CN.DATA.mono_or_color = "simulator"           # "simulator" | "color"
_CN.DATA.min_clip = 0
_CN.DATA.max_clip = 65535                 # simulator default; override if color
_CN.DATA.dynamic_range = 65535            # usually set to max_clip in finalize
_CN.DATA.subs_num = 1
_CN.DATA.overscan_scale = 3

# splits / loader
_CN.DATA.restart_dataset = False
_CN.DATA.train_ratio = 0.85
_CN.DATA.val_ratio   = 0.15
_CN.DATA.test_ratio  = 0.15

# sampling / geometry
_CN.DATA.samples_num_per_image = 1
_CN.DATA.debug_ratio = 1.0
_CN.DATA.covisibility_tolerance = 0.1

# folds / seeds
_CN.DATA.k_folds = 5
_CN.DATA.data_path = r"F:\yakirs_thesis\thesis_code\data"
_CN.DATA.raw_data_path = r"F:\yakirs_thesis\thesis_code\data\raw"
_CN.DATA.master_seed = 42
_CN.DATA.seeds = [42, 34, 52, 298, 789]

# dataset variant tag
_CN.DATA.dataset_variant = "v9" # v9 = 9000nm, v11 = 1100nm

##############  ↓  Train Config  ↓  ##############
_CN.TRAIN = CN()
_CN.TRAIN.max_epochs = 50


_CN.TRAIN.logger = "mlflow"                 # ["mlflow","tensorboard"]
_CN.TRAIN.weight_decay = 1e-1
_CN.TRAIN.batch_size = 8
_CN.TRAIN.val_interval = 1
_CN.TRAIN.seed = 42
_CN.TRAIN.log_dir = "logs"
_CN.TRAIN.px_thresholds = [0.5, 1.0, 2.0, 5.0]

# distributed
_CN.TRAIN.world_size = int(os.environ.get("WORLD_SIZE", 1))

# profiler
_CN.TRAIN.profiler_name = "pytorch"

# optimizer
_CN.TRAIN.optimizer = "adamw"               # ["adam","adamw"]
_CN.TRAIN.initial_lr = 1e-4
_CN.TRAIN.adamw_decay = 1e-3
_CN.TRAIN.adam_decay  = 1e-3

# warmup
_CN.TRAIN.warmup_ratio = 0.1
_CN.TRAIN.warmup_epochs = 5

# scheduler
_CN.TRAIN.scheduler = "CosineAnnealing"     # ["MultiStepLR","CosineAnnealing","ExponentialLR"]
_CN.TRAIN.scheduler_interval = "epoch"      # ["epoch","step"]
_CN.TRAIN.cosa_tmax = 15                    # typically 3 * warmup_epochs
_CN.TRAIN.eta_min = 1e-6

# plotting

_CN.TRAIN.enable_plotting = True
_CN.TRAIN.val_plot_interval = 3
_CN.TRAIN.n_val_pairs_to_plot = 4
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
_CN.RUN.num_workers = 1             # set in finalize per platform
_CN.RUN.prefetch_factor = 4
_CN.RUN.log_every_n_steps = 1
_CN.RUN.num_train_samples = 25278
_CN.RUN.num_val_samples = 5416
_CN.RUN.num_test_samples = 5416





def finalize_cfg(cfg):
    """Populate derived/run-time fields (CN version of your __post_init__)."""
    cfg.defrost()

    # 1) Run platform & sanity toggle
    cfg.RUN.sanity_check = False
    cfg.RUN.run_platform = 'hpc' if os.environ.get('SLURM_JOB_ID') else 'local'
    is_hpc = (cfg.RUN.run_platform == 'hpc')

    # 2) Platform-dependent paths & system knobs
    cfg.PROJ.cwd = ("/home/ARO.local/yakirh/Projects/yakirs_thesis/thesis"
        if is_hpc else r"F:/yakirs_thesis/thesis_code"
                     )
    cfg.PROJ.data_path = f"{cfg.PROJ.cwd}/data"
    cfg.PROJ.checkpoints_path = f"{cfg.PROJ.cwd}/checkpoints"

    cfg.PHYS.coeff_path = (
        f"{cfg.PROJ.cwd}/src/dataset/physical_model/coeff/coefficients_9um.npz"
    )

    cfg.pretrained_ckpt = f"{cfg.PROJ.cwd}/checkpoints/loftr_outdoor.ckpt" if cfg.PROJ.base_model == "loftr" else f"{cfg.PROJ.cwd}/checkpoints/aspan_outdoor.ckpt"

    cfg.RUN.tracking_uri = (
        "http://10.26.36.96:5000" if is_hpc else "http://127.0.0.1:5000"
    )

    ### change this for each fold
    cfg.RUN.fold_idx = 0
    cfg.DATA.wl= "11um"
    ###

    cfg.RUN.num_workers = 24 if is_hpc else 1
    cfg.RUN.num_nodes   = int(os.environ.get("SLURM_JOB_NUM_NODES", 1)) if is_hpc else 1
    cfg.RUN.multi_nodes = is_hpc

    # 3) Dataset size & batch size (as in your dataclass)
    cfg.RUN.sanity_dataset_size = 50
    cfg.TRAIN.batch_size = 1

    # 4) Name this run
    cfg.RUN.run_type = 'sanity' if cfg.RUN.sanity_check else 'full'

    # If you already have a builder, use it; else inline the simple one:
    def _build_experiment_name(config):
        model_name = config.PROJ.base_model
        sanity = 'sanity' if config.RUN.sanity_check else 'full'
        wl = config.DATA.wl
        return f"{model_name}_{wl}_{sanity}"

    cfg.RUN.experiment_name = _build_experiment_name(cfg)
    cfg.RUN.run_name = datetime.now().strftime("%d_%m__%H_%M")


    cfg.freeze()
    return cfg



def get_petits_defaults():
    return _CN.clone()

def get_config():
    petits_config = get_petits_defaults()
    finalize_cfg(petits_config)
    return petits_config


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}