import platform
from enum import Enum
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from typing import Union



@dataclass
class Tau2Intrinsics:
    fx: float = 1117.65  # Focal length in pixels (horizontal)
    fy: float = 1117.65  # Focal length in pixels (vertical)
    cx: float = 320.0    # Principal point x-coordinate (image center)
    cy: float = 256.0    # Principal point y-coordinate (image center)
    skew: float = 0.0    # Skew coefficient (assumed zero)
    width: int = 640     # Image width in pixels
    height: int = 512    # Image height in pixels

    # more properties
    focal_length_mm: float = 19 # Focal length in mm (19 mm)
    pixel_pitch_mm: float = 0.017 # Pixel sensor pitch in mm (17um)
    pixel_tolerance: float = 1  # Pixel tolerance (we allow 1 pixel tolerance)

    def K(self):
        """Returns the camera intrinsic matrix."""
        return np.array([
            [self.fx, self.skew, self.cx],
            [0.0,     self.fy,   self.cy],
            [0.0,     0.0,       1.0]
        ], dtype=np.float32)

@dataclass
class SimulationParams:
    # input - True for dataset generation, False for simulation
    dataset_mode: bool = True
    sim_level: int = 2   # input - Levels: 1 (mild), 2 (moderate), 3 (strong)
    truncated_mu: float = 0.75  # input - Truncated Gaussian mean for simulation kpts distribution
    truncated_sigma: float = 0.2  # input - Truncated Gaussian std for simulation kpts distribution
    plane2sat: float = (500 * 10**3) / 2500 # 500km satellite altitude, 2.5km plane altitude
    pitch_std: float = field(init=False)
    roll_std: float = field(init=False)
    yaw_std: float = field(init=False)
    tx_std: float = field(init=False)
    ty_std: float = field(init=False)
    tz_std: float = field(init=False)

    def scale_simulation(self, scale: float):
        """Scales the simulation parameters by a given factor."""
        self.pitch_std *= scale
        self.roll_std *= scale
        self.yaw_std *= scale
        self.tx_std *= scale
        self.ty_std *= scale
        self.tz_std *= scale

    def __post_init__(self):
        SIM_LEVELS = {
            1: dict(
                # High‐precision GNSS‐based POD (40% at decimeter, 50% at centimeter level) & XACT ADCS
                pitch=0.0039,  # ≈0.0042°–0.0117° (3σ) on MinXSS-1’s XACT system
                roll=0.0039,
                yaw=0.0039,
                tx=0.1,  # high end GNSS
                ty=0.1,
                tz=0.1,
            ),
            2: dict(
                # Standard civilian GPS receiver & simple magnetometer‐only ADCS
                pitch=2.0,  # ≈2° attitude knowledge with only a magnetometer
                roll=2.0,
                yaw=2.0,
                tx=10,  # pseudorange σ ≈10 m for C/A‐code GPS
                ty=10,
                tz=10,
            ),
            3: dict(
                # TLE‐only orbit propagation & no active ADCS
                pitch=10.0,  # pointing errors up to ≈10° are common without fine ADCS
                roll=10.0,
                yaw=10.0,
                tx=500,  # NORAD TLE‐propagated orbits ±0.5 km (500 m)
                ty=500,
                tz=500,
            ),
        }

        params = SIM_LEVELS[self.sim_level] if not self.dataset_mode else SIM_LEVELS[2]

        self.pitch_std = params['pitch']
        self.roll_std = params['roll']
        self.yaw_std = params['yaw']
        self.tx_std = params['tx'] / self.plane2sat
        self.ty_std = params['ty'] / self.plane2sat
        self.tz_std = params['tz'] / self.plane2sat

class LwirChannel(Enum):
    """An enumeration for Flir's LWIR channels obtained by applying bandpass filters (or not) in nano-meter."""
    pan = 0
    nm8000 = 8000
    nm9000 = 9000
    nm10000 = 10000
    nm11000 = 11000
    nm12000 = 12000
    def __str__(self) -> str:
        return str(self.value) + "nm" if self.value > 0 else "pan"

class RGB(Enum):
    red = 0
    green = 1
    blue = 2

@dataclass
class ProjectConfig:
    """A general class with configurations relevant to all configs."""
    # paths:
    cwd: Path = Path(r"") #placeholder
    data_path: Path = Path(r"./data")
    checkpoints_path: Path = Path(r"./checkpoints")
    logs_path: Path = Path(r"./logs")
    results_path: Path = Path(r"./results")
    priors_path: Path = Path(r"./data/petit_s/priors")
    base_model: str = "loftr"

    # device:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # thresholds:
    conf_thresh: float = 0.8

    # misc:
    coarse_scale: int = 8
    fine_scale: int = 2

    pretrained_ckpt: Path = field(init=False)

    def __post_init__(self):
        self.pretrained_ckpt = Path(f"./checkpoints/{self.base_model}/outdoor_ds.ckpt")

@dataclass
class DataConfig(ProjectConfig):
    """Sets the base configurations for the dataset."""
    wls: List[LwirChannel] = field(default_factory=lambda: [LwirChannel.nm9000, LwirChannel.pan, LwirChannel.pan])
    sample_shape: tuple[int, int] = field(default_factory=lambda: [128, 128])
    width: int = 256
    height: int = 256
    image_shape: tuple[int, int] = field(default_factory=lambda: (256, 256))
    fine_window_size: int = 8
    seed: int = 42
    mono_or_color: str = "simulator"  # simulator
    min_clip: int = 0
    max_clip: int = 255 if mono_or_color == "color" else  65535
    dynamic_range = max_clip
    subs_num: int = 1
    overscan_scale = 3

    restart_dataset: bool = True
    train_ratio: float = 0.85
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    min_iou: float = 0.1
    max_iou: float = 0.3
    samples_num_per_image: int = 1

    debug_ratio: float = 1

    covisibility_tolerance: float = 0.1
    # data normalizatoin values:
    # LWIR (9um & 11um):

    wl01_mean = np.array([ 0.05003805086016655, 0.04886174947023392])
    wl01_std = np.array([0.0002582324086688459,  0.00023569920449517667])

    # RGB:
    rgb_mean = np.array([0.25279155, 0.32981821, 0.21969282])
    rgb_std = np.array([0.14092436, 0.1237567,  0.11398276])

    # K-folds:
    k_folds: int = 5

    # seeds
    master_seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [41, 42, 43, 44, 45])  # seeds for different runs

    #dataset variants:
    dataset_variant: str = 'v1'


@dataclass
class PhysModelConfig(ProjectConfig):
    """Sets the physical model configuration."""
    use_phys: dict = field(default_factory=lambda:{'backbone': False, 'coarse': False, 'fine': False})
    dtype: Union[np.ndarray, torch.Tensor] = np.ndarray
    coeff_path: Path = Path(r"") # placeholder

@dataclass
class BackboneConfig(ProjectConfig):
    """Sets the backbone configuration."""
    def __init__(self, use_phys: bool = False):
        self.use_phys: bool = use_phys
        self.concat_phys: bool = False
        self.inject_phys_coarse: bool = False
        self.inject_phys_fine: bool = False
        self.backbone_type: tuple[int, int] = (8, 2)  # options: (8, 2) or (16, 4)
        self.initial_dim: int = 128
        self.block_dims: list[int] = [128, 196, 256]
        self.sft_lambda: float = 1.0

@dataclass
class PosEncConfig(ProjectConfig):
    d_model: int = 256
    max_shape: tuple[int, int] = (128, 128)

@dataclass
class CoarseConfig(ProjectConfig):
    """Sets the coarse model configuration."""
    def __init__(self, use_phys: bool = False):
        self.use_phys: bool = use_phys

        self.phys_lambda: float = 1

        self.match_type: str = "dual_softmax"
        self.dsmax_temperature: float = 0.1
        self.border_rm: int = 2
        self.thr: float = 0.2
        self.train_coarse_percent: float = 0.2
        self.train_pad_num_gt_min: int = 10
        self.num_heads: int = 8
        self.d_model: int = 256
        self.layer_names: List[str] = ['self', 'cross'] * 4
        self.attention_type: str = "linear"
        self.sparse_spvs: bool = False


@dataclass
class FineConfig(ProjectConfig):
    def __init__(self, use_phys: bool = False):
        self.use_phys: bool = use_phys
        self.fine_window_size: int = 5
        self.fine_concat_coarse_feat: bool = True
        self.unfold_stride: int = 8
        self.attention_type: str = "linear"
        self.num_heads: int = 8
        self.d_model: int = 128
        self.layer_names: List[str] = ['self', 'cross'] * 1



@dataclass
class ModelConfig(ProjectConfig):
    """Sets the model configuration."""
    use_phys: bool = None
    phys: PhysModelConfig = field(default_factory=PhysModelConfig)
    backbone: BackboneConfig = None
    pe: PosEncConfig = None
    coarse: CoarseConfig = None
    fine: FineConfig = None

    def __post_init__(self):
        use_phys = self.phys.use_phys
        self.model_name = 'loftr'
        self.use_phys = any(use_phys.values())
        self.backbone = BackboneConfig(use_phys=use_phys['backbone'])
        self.pe = PosEncConfig()
        self.coarse = CoarseConfig(use_phys=use_phys['coarse'])
        self.fine = FineConfig(use_phys=use_phys['fine'])


@dataclass
class TrainConfig:
    """Sets the base configurations for training."""
    max_epochs: int = 20
    stage1_epochs: int = 5  # stage 1 - warm-up new layers
    stage2_epochs: int = 20  # stage 2 - end2end training

    logger: str = "mlflow"  # mlflow, tensorboard
    weight_decay: float = 10e-2
    batch_size: int = 32
    val_interval: int = 1
    seed: int = 42
    log_dir: str = "logs"
    px_thresholds: List[float] = field(default_factory=lambda: [0.5, 1, 2, 5])

    # distributed training:
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))

    # profiler:
    profiler_name: str = 'pytorch'

    # optimizer:
    optimizer: str = "adamw"  # [adam, adamw]
    initial_lr: Optional[float] = 1e-4
    adamw_decay: float = 1e-3
    adam_decay: float = 1e-3

    # step-based warm-up
    warmup_ratio: float = 0.1
    warmup_epochs: int = 5

    # learning rate scheduler
    scheduler: str = 'CosineAnnealing'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
    scheduler_interval: str = 'epoch'  # [epoch, step]
    cosa_tmax: int = 3 * warmup_epochs # COSA: CosineAnnealing
    eta_min: float = 1e-6  # Minimum learning rate at the end of training

    # plotting related
    enable_plotting: bool = True
    n_val_pairs_to_plot: int = 8  # number of val/test pairs for plotting
    plot_mode: str = 'evaluation'  # ['evaluation', 'confidence']
    plot_matches_alpha: str = 'dynamic'
    plot_every_n_epochs: int = 1  # how often to plot matches during training

    # geometric metrics and pose solver
    epi_err_thr: list = field(default_factory=lambda: [1e-5, 1e-4, 3e-4])  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue) -> We use 1e-4 since its outdoor dataset    pose_geo_model: str = 'E'  # ['E', 'F', 'H']
    pose_estimation_method: str = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
    ransac_pixel_thr: float = 0.5
    ransac_conf: float = 0.99999
    ransac_max_iters: int = 10000
    use_magsacpp: bool = False

    # data sampler for train_dataloader
    data_sampler: str = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
    # 'scene_balance' config
    n_samples_per_subset: int = 200
    sb_subset_sample_replacement: bool = True  # whether sample each scene with replacement or not
    sb_subset_shuffle: bool = True  # after sampling from scenes, whether shuffle within the epoch or not
    sb_repeat: int = 1  # repeat N times for training the sampled data
    # 'random' config
    rdm_replacement: bool = True
    rdm_num_samples: Optional[int] = None

    # gradient clipping
    gradient_clipping: float = 0.5


@dataclass()
class RunConfig:
    """Sets the base configurations for running the model."""
    run_platform: str = ""  # local, hpc - placeholder
    sanity_check: bool = None  # placeholder
    sanity_dataset_size: int = None  # placeholder
    num_gpus: int = torch.cuda.device_count()
    num_nodes: int = 0 # placeholder
    multi_nodes: bool = None  # placeholder


    accelerator: str = "gpu"
    tracking_uri: str = "" # placeholder
    experiment_name: str = "" # placeholder
    run_name: str = "" # placeholder
    num_workers: int = None # placeholder, will be set in __post_init__
    prefetch_factor: int = 4  # number of batches to prefetch
    log_every_n_steps: int = 1

@dataclass
class InferenceConfig:
    """Sets the base configurations for inference."""
    inference_path : Path = Path(r"./checkpoints/best")  # path to save inference results
    base_model : str = "loftr"  # name of the model to use for inference
    sub_model: str = "pretrained_rgb" # no_phys, full_phys, pretrained_rgb



@dataclass
class LossConfig:
    """Sets the base configurations for the loss function."""
    # coarse level
    coarse_type: str =  'focal'  # ['focal', 'cross_entropy']
    coarse_weight: float = 1.0

    # fine level
    fine_type: str = 'l2_with_std'  # ['l2_with_std', 'l2']
    fine_weight: float = 1.0
    fine_correct_thr: float = 1.0

    # focal loss
    pos_weight: float = 1.0
    neg_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

@dataclass
class Config(ProjectConfig):
    """Master configuration that combines all individual configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fine: FineConfig = field(default_factory=FineConfig)
    coarse: CoarseConfig = field(default_factory=CoarseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    sim: SimulationParams = field(default_factory=lambda: SimulationParams(dataset_mode=True))
    sensor: Tau2Intrinsics = field(default_factory=Tau2Intrinsics)
    phys: PhysModelConfig = field(default_factory=PhysModelConfig)

    def __post_init__(self):
        ####
        # Define the run:
        self.run.sanity_check = True

        ####
        self.run.run_platform = 'hpc' if os.environ.get("SLURM_JOB_ID") else 'local'  # local, hpc
        # configs that depend on the run platform:
        self.cwd = Path(r"/home/ARO.local/yakirh/Projects/yakirs_thesis/thesis") if self.run.run_platform == 'hpc' else Path(
            r"/")
        self.run.tracking_uri = "http://10.26.36.96:5000" if self.run.run_platform == "hpc" else "http://127.0.0.1:5000"
        self.model.phys.coeff_path = Path(f"{self.cwd}/src/physical_model/coeff/coefficients_9um.npz")
        self.run.num_workers = 24 if self.run.run_platform == "hpc" else 1
        self.run.num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"]) if self.run.run_platform == "hpc" else 1
        self.run.multi_nodes = True if self.run.run_platform == "hpc" else False

        self.run.sanity_dataset_size = 5000 if self.run.run_platform == "hpc" else 5000
        self.train.batch_size = 4 if self.run.run_platform == "hpc" else 4
        self.run_type = 'sanity' if self.run.sanity_check else 'full'
        self.run.experiment_name = self.get_experiment_name()
        self.run.run_name = "{time}".format(time=datetime.now().strftime("%d_%m__%H_%M"))

        self.data.coarse_factor = self.model.backbone.backbone_type[0]
        self.data.fine_factor = self.model.backbone.backbone_type[1]



    def get_experiment_name(self):
        model_name = self.model.model_name
        use_phys = self.model.use_phys
        sft_phys = self.model.backbone.use_phys and (self.model.backbone.inject_phys_fine or self.model.backbone.inject_phys_coarse)
        coarse_phys = self.model.coarse.use_phys
        phys_str = 'phys' if use_phys else 'no_phys'
        ds_size = str(self.run.sanity_dataset_size) if self.run.sanity_check else 'full'
        concat_phys = self.model.backbone.concat_phys
        if use_phys and concat_phys:
            phys_str += '_concat'
        elif sft_phys:
            phys_str += '_sft'
        if coarse_phys:
            phys_str += '_coarse'
        exp_name = f"{model_name}_{phys_str}_{self.data.dataset_variant}_{ds_size}"
        return exp_name

