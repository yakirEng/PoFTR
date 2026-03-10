import os
import numpy as np
from tqdm import tqdm
import pickle
from scipy.stats import truncnorm
from pathlib import Path
import copy
from scipy.interpolate import PchipInterpolator

from src.utils.misc import lower_config
from src.configs.poftr_configs import get_config
from src.dataset.stats.dataset_stats import DatasetStats



class DistributionManager:

    """ class to control the distribution of the simulation"""

    def __init__(self, config, distribution_type=None, mu=None, std=None, weights=None):
        self.config = config
        self.interpolator = None
        self.distribution_type = self.config["sim"]["distribution_type"] if distribution_type is None else distribution_type
        self._define_mus_std(mu, std, weights)
        self.calibration_path = Path(self.config["proj"]["data_path"]) / 'simulated' / 'dist_management'
        self.calibration_path.mkdir(parents=True, exist_ok=True)
        self.num_calibration_samples = self.config["sim"]["num_calibration_samples"]
        self.dataset_path = Path(self.config["proj"]["data_path"]) / 'raw' / 'images' / '9um'
        self.calibration_filename = 'co_visibility_calibration6.npz'
        self.interpolator_filename = 'co_visibility_interp6.pkl'
        self._inverse_lookup = None

    def _define_mus_std(self, mu, std, weights):
        if self.distribution_type == "truncnorm":
            self.mu = self.config["sim"]["truncated_mu"] if mu is None else mu
            self.std = self.config["sim"]["truncated_sigma"] if std is None else std

        elif self.distribution_type == "mixture_norm":
            self.mixture_mus = np.array(self.config["sim"]["mixture_mus"], np.float32) if mu is None else np.array(mu, np.float32)
            self.mixture_sigmas = np.array(self.config["sim"]["mixture_sigmas"], np.float32) if std is None else np.array(std, np.float32)
            self.mixture_weights = np.array(self.config["sim"]["mixture_weights"], np.float32) if weights is None else np.array(weights, np.float32)
            self.mixture_weights /= self.mixture_weights.sum() # weights must sum to 1
        else:
            raise ValueError(f"Unknown distribution_type: {self.distribution_type}")

    def _get_kpts_pct(self, scales: np.ndarray) -> np.ndarray:
        from src.dataset.simulator.monosample import MonoSample
        files = list((self.dataset_path).iterdir())
        files = np.random.choice(files, size=min(len(files), self.num_calibration_samples), replace=False)
        M = len(scales)
        mean_pct = np.zeros(M, dtype=np.float32)

        num_samples_per_scale = 100  # average over multiple samples

        for i, s in enumerate(scales):
            per_image_pct = []
            for file_path in tqdm(files, desc=f"Scale {i + 1}/{M}"):
                scale_samples = []
                for _ in range(num_samples_per_scale):  # Multiple samples per image
                    sample = MonoSample.generate(
                        self.config,
                        image_name=file_path.name,
                        idx=i,
                        sim_scale=s
                    ).to_dict()
                    scale_samples.append(DatasetStats.compute_co_visibility(sample))
                per_image_pct.append(np.mean(scale_samples))  # Average over samples
            mean_pct[i] = float(np.mean(per_image_pct))
        return mean_pct

    def fit(self, num_scales: int = 30):
        """Calibrate r_of_s by sampling `num_scales` between 0-1."""
        scales = np.linspace(1/num_scales, 1, num_scales, dtype=np.float32)
        r_vals = self._get_kpts_pct(scales)
        self.interpolator = PchipInterpolator(scales, r_vals, extrapolate=True)

        # persist raw and object
        np.savez(self.calibration_path / self.calibration_filename,
                 scales=scales, r_vals=r_vals)
        with open(self.calibration_path / self.interpolator_filename, 'wb') as f:
            pickle.dump(self.interpolator, f)

    def predict(self, scales: np.ndarray):
        """
        Given an array of scales ∈[0,1], returns the predicted valid‐keypoint%
        (same shape as `scales`).
        """
        self._ensure_interpolator()
        return self.interpolator(scales)


    def _ensure_interpolator(self):
        if self.interpolator is None:
            with open(self.calibration_path / self.interpolator_filename, 'rb') as f:
                self.interpolator = pickle.load(f)

    def _build_inverse_lookup(self):
        """Build and cache the inverse lookup table."""
        if self._inverse_lookup is not None:
            return

        self._ensure_interpolator()

        # Build lookup table
        scales_grid = np.linspace(0.01, 1.0, 1000, dtype=np.float32)
        r_grid = self.interpolator(scales_grid)

        # Sort by r values
        order = np.argsort(r_grid)
        self._inverse_lookup = {
            'r_sorted': r_grid[order],
            'scales_sorted': scales_grid[order]
        }

    def covisibility_to_scale(self, r: np.ndarray) -> np.ndarray:
        """Inverse mapping r -> scale."""
        self._build_inverse_lookup()

        r_sorted = self._inverse_lookup['r_sorted']
        scales_sorted = self._inverse_lookup['scales_sorted']

        # Clip and interpolate
        r_clipped = np.clip(r, r_sorted[0], r_sorted[-1])
        scales = np.interp(r_clipped, r_sorted, scales_sorted)
        return scales

    def sample_covisibility_truncnorm(self, n: int) -> np.ndarray:
        """
        Single truncated normal in [0,1], controlled by (mu, std).
        """
        mu, sigma = self.mu, self.std
        a = (0.0 - mu) / sigma
        b = (1.0 - mu) / sigma
        r = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n).astype(np.float32)
        return np.clip(r, 0.0, 1.0)

    def sample_covisibility_mixture(self, n: int) -> np.ndarray:
        """
        Mixture of truncated normals:
        - allows explicit control of hard/mid/easy covisibility.
        """
        K = len(self.mixture_mus)
        comps = np.random.choice(K, size=n, p=self.mixture_weights)

        r = np.empty(n, dtype=np.float32)
        for k in range(K):
            idx = comps == k
            if not np.any(idx):
                continue
            mu = self.mixture_mus[k]
            sigma = self.mixture_sigmas[k]
            a = (0.0 - mu) / sigma
            b = (1.0 - mu) / sigma
            r[idx] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=idx.sum())
        return np.clip(r, 0.0, 1.0)

    @classmethod
    def sample_targets(cls, config, n: int, distribution_type=None, mu = None, std = None, weights = None) -> np.ndarray:
        """
        Main API: sample `n` simulation scales such that the resulting
        co-visibility distribution matches the desired hardness distribution.

        Chooses between:
        - truncnorm(mu, std)      if distribution_type == "truncnorm"
        - mixture (hard/mid/easy) if distribution_type == "mixture_norm"
        """
        dm = cls(config, distribution_type, mu, std, weights)
        if dm.distribution_type == "mixture_norm":
            r = dm.sample_covisibility_mixture(n)
        elif dm.distribution_type == "truncnorm":
            r = dm.sample_covisibility_truncnorm(n)
        else:
            raise ValueError(f"Unknown distribution_type: {dm.distribution_type}")
        return r


def main():
    config = lower_config(get_config())["poftr"]
    os.chdir(config["proj"]["cwd"])
    dm = DistributionManager(config=config)
    dm.fit(num_scales=30)
    print("Distribution Manager trained and saved.")



if __name__ == '__main__':
    main()