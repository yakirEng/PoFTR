import os
import io
import numpy as np
import tarfile
from tqdm import tqdm
from pathlib import Path
import json
from typing import Any, Union
import copy
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from loguru import logger

from src.utils.misc import lower_config
from src.configs.poftr_configs import get_config


class DatasetStats:
    def __init__(self, config, distribution_type:str = None, dataset_version: str = None, dataset_type: str = 'train', ablation_type: str = 'standard'):
        self.config = config
        self.distribution_type = distribution_type if distribution_type is not None else config['sim']['distribution_type']
        self.dataset_version = dataset_version
        self.ablation_type = ablation_type
        self.dataset_type = dataset_type
        if ablation_type != 'standard':
            self.dataset_path = Path(config["proj"]["data_path"]) / 'simulated' / 'datasets' / self.distribution_type /'ablations' / f"{dataset_version}_{ablation_type}" / dataset_type
        else:
            self.dataset_path = Path(config["proj"]["data_path"]) / 'simulated' / 'datasets' / self.distribution_type / dataset_version / dataset_type
        self.stats_filename = f"{dataset_type}_distribution_stats.json"
        self.stats_path = self.dataset_path.parent / 'dist_stats'
        self.stats_path.mkdir(parents=True, exist_ok=True)
        self.plots_path = self.stats_path / 'plots'
        self.plots_path.mkdir(parents=True, exist_ok=True)
        self.co_visibility = []
        self.valid_pixels = []
        self.sim_levels = []


    @classmethod
    def get_sample_stats(cls, sample_dict: dict):
        co_visibility = cls.compute_co_visibility(sample_dict)
        valid_pixels = cls.compute_valid_pixels_precent(sample_dict)
        stats = {
            'co_visibility': co_visibility,
            'valid_pixels': valid_pixels,
        }
        return stats

    @classmethod
    def get_dataset_stats(cls, config, distribution_type, dataset_version, dataset_type: str = 'train', ablation_type: str = 'standard'):

        obj = cls(config=config, distribution_type=distribution_type, dataset_version=dataset_version, dataset_type=dataset_type, ablation_type=ablation_type)
        ds_config = copy.deepcopy(config)
        ds_config["phys"]["use_phys"] = False
        tar_files = [f for f in os.listdir(obj.dataset_path) if f.endswith('.tar')]
        pbar = tqdm(tar_files, desc="Initializing... ", leave=False)
        logger.info(f"Processing {len(tar_files)} tar files in {obj.dataset_path}")
        for i, tar_file_name in enumerate(pbar):
            pbar.set_description(f"Processing {tar_file_name}... ")
            tar_path = os.path.join(obj.dataset_path, tar_file_name)
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    npz_files = [m for m in tar.getmembers() if m.name.endswith('.npz')]

                    if not npz_files:
                        print(f"Warning: No .npz files found in {tar_path}")
                        raise Exception("No .npz files found in tar file")

                    for member in npz_files:
                        file_buffer = tar.extractfile(member).read()

                        with io.BytesIO(file_buffer) as f:
                            try:
                                sample = np.load(f)
                                obj.co_visibility.append(obj.compute_co_visibility(sample) if sample["co_visibility"] is None else float(sample['co_visibility']))
                                obj.valid_pixels.extend(obj.compute_valid_pixels_precent(sample) if sample["valid_pixels"] is None else list(map(float, sample['valid_pixels'])))

                            except Exception as e:
                                print(f"\nWarning: Could not process file {member.name} in {tar_path}. Error: {e}")
                                continue
            except Exception as e:
                print(f"\nError: Could not open or read tar file {tar_path}. Error: {e}")
                continue
        obj.save_to_json()
        return obj


    @staticmethod
    def compute_co_visibility(sample: Union[Any, dict]):
        h, w = sample['image0'].shape[:2] if len(sample['image0'].shape) == 3 else sample['image0'].shape # source image size
        H0 = (sample['H0']).astype(np.float32)
        H1 = (sample['H1']).astype(np.float32)

        # 1) build grid of source‐pixel coordinates
        ys, xs = np.indices((h, w))
        pts = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs).ravel()], axis=0).astype(np.float32)  # shape (3, N)

        # 2) warp through H0 and H1
        dst0 = H0 @ pts  # shape (3, N)
        dst0 /= dst0[2:3]  # normalize homogeneous
        u0, v0 = dst0[0], dst0[1]  # each shape (N,)

        dst1 = H1 @ pts
        dst1 /= dst1[2:3]
        u1, v1 = dst1[0], dst1[1]

        # 3) check which are in‐bounds
        valid0 = (u0 >= 0) & (u0 < w) & (v0 >= 0) & (v0 < h)
        valid1 = (u1 >= 0) & (u1 < w) & (v1 >= 0) & (v1 < h)

        # 4) intersection: these source pixels map into _both_ warps
        both = valid0 & valid1

        num_both = both.sum()
        all = h * w

        return num_both / all

    @staticmethod
    def compute_valid_pixels_precent(sample: Union[Any, dict]):
        mask0 = sample['mask0']
        mask1 = sample['mask1']
        valid_pixels0 = np.sum(mask0 > 0) / np.prod(mask0.shape)
        valid_pixels1 = np.sum(mask1 > 0) / np.prod(mask1.shape)
        valid_pixels = [valid_pixels0, valid_pixels1]
        return valid_pixels

    def save_to_json(self):
        stats = {
            'co_visibility': self.co_visibility,
            'valid_pixels': self.valid_pixels,
        }
        full_stats_path = self.stats_path / self.stats_filename
        with open(full_stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f'{self.dataset_version} {self.dataset_type} stats saved to {full_stats_path}')

    def load_from_json(self):
        full_stats_path = self.stats_path / self.stats_filename
        with open(full_stats_path, 'r') as f:
            stats = json.load(f)
        self.co_visibility = stats['co_visibility']
        self.valid_pixels = stats['valid_pixels']
        logger.info(f'{self.dataset_version} {self.dataset_type} stats loaded from {full_stats_path}')

    def show_covisibility_histogram(self):
        """
        Public API:
        - Draw empirical co-visibility histogram.
        - Overlay the theoretical distribution:
          * truncated normal, or
          * mixture of truncated normals,
          depending on config["sim"]["distribution_type"].
        """
        # 1) load data
        self.load_from_json()
        data = np.array(self.co_visibility)

        # 2) empirical histogram
        counts, bins, _ = plt.hist(
            data,
            bins=30,
            alpha=0.7,
            label="Co-Visibility Histogram",
        )
        plt.xlabel("%")
        plt.ylabel("Frequency")
        plt.title(f"Histogram for {self.distribution_type} {self.dataset_version} {self.dataset_type} Dataset")

        # 3) x-domain for theoretical curves
        x = np.linspace(0, 1, 500)

        if self.distribution_type == "truncnorm":
            self._overlay_truncnorm_pdf(x, data, bins)
        elif self.distribution_type == "mixture_norm":
            self._overlay_mixture_pdf(x, data, bins)
        else:
            raise ValueError(f"Unknown distribution_type: {self.distribution_type}")

        plt.legend()

        plot_filename = f"{self.distribution_type}_{self.dataset_version}_{self.dataset_type}_co_visibility_histogram.png"
        full_plot_path = self.plots_path / plot_filename
        plt.savefig(full_plot_path)
        plt.close()

    def _overlay_truncnorm_pdf(self, x: np.ndarray, data: np.ndarray, bins: np.ndarray):
        """
        Overlay a single truncated normal PDF on top of an existing histogram.
        Assumes the histogram has already been drawn.
        """
        mu = self.config["sim"]["truncated_mu"]
        sigma = self.config["sim"]["truncated_sigma"]

        # truncated-normal params for [0,1]
        a = (0.0 - mu) / sigma
        b = (1.0 - mu) / sigma

        pdf = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)

        # scale PDF to match histogram counts
        bin_width = bins[1] - bins[0]
        pdf_scaled = pdf * len(data) * bin_width

        plt.plot(
            x,
            pdf_scaled,
            linewidth=2,
            label=f"TruncNorm PDF\nμ={mu}, σ={sigma}",
        )

    def _overlay_mixture_pdf(self, x: np.ndarray, data: np.ndarray, bins: np.ndarray):
        """
        Overlay a mixture of truncated normals on top of an existing histogram.
        Uses:
          - sim.mixture_mus
          - sim.mixture_stds
          - sim.mixture_weights
        from the config.
        """
        mus = np.array(self.config["sim"]["mixture_mus"], dtype=np.float32)
        sigmas = np.array(self.config["sim"]["mixture_sigmas"], dtype=np.float32)
        weights = np.array(self.config["sim"]["mixture_weights"], dtype=np.float32)
        weights = weights / weights.sum()

        pdf = np.zeros_like(x)
        components_desc = []

        for mu, sigma, w in zip(mus, sigmas, weights):
            a = (0.0 - mu) / sigma
            b = (1.0 - mu) / sigma
            pdf_k = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)
            pdf += w * pdf_k
            components_desc.append(f"μ={mu:.2f}, σ={sigma:.2f}, w={w:.2f}")

        bin_width = bins[1] - bins[0]
        pdf_scaled = pdf * len(data) * bin_width

        label = "Mixture TruncNorm PDF\n" + "\n".join(components_desc)
        plt.plot(
            x,
            pdf_scaled,
            linewidth=2,
            label=label,
        )

    def show_kpts_precent_vs_sim_level(self, dataset_name: str = 'train'):
        import matplotlib.pyplot as plt
        self.load_from_json(dataset_name)
        plt.scatter(self.sim_levels, self.co_visibility, alpha=0.5)
        plt.xlabel('Simulation Level')
        plt.ylabel('Correct Keypoints Percent')
        plt.title(f'Valid Keypoints vs Simulation Level for {dataset_name} Dataset')
        plt.grid()
        plt.show()


def main():
    config = lower_config(get_config())
    distribution_types = ["truncnorm", "mixture_norm"]
    dataset_versions = ["9um_11um"]
    dataset_types = ['train', 'val', 'test']
    ablation_versions = ['standard', 'upper_bound']  # 'noisy_priors', 'zeroed_priors'
    for distribution_type in distribution_types:
        for dataset_version in dataset_versions:
            for ablation_type in ablation_versions:
                for dataset_type in dataset_types:
                    stats_config = copy.deepcopy(config)
                    stats = DatasetStats.get_dataset_stats(config=stats_config["poftr"], distribution_type=distribution_type, dataset_version=dataset_version, ablation_type=ablation_type, dataset_type=dataset_type)
                    stats.show_covisibility_histogram()

if __name__ == "__main__":
    main()