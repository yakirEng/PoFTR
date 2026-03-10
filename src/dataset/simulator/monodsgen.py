import random
import os
import gc
import psutil
import numpy as np
import copy
from scipy.stats import truncnorm
from tqdm import tqdm
import shutil
from pathlib import Path
import webdataset as wds
import io
import csv
from typing import List
from loguru import logger

from src.dataset.simulator.monosample import MonoSample
from src.configs.poftr_configs import get_config, get_K_from_cfg
from src.utils.misc import lower_config



class MonoDsGen:
    """generate color dataset, and saves it to a csv file"""
    def __init__(self, config, dataset_version: str = None, distribution_type: str = None, wls: List = None, is_generate=False, ablation_type: str = "standard"):
        # Configuration and control flags
        self.wls = wls if wls is not None else config["data"]["wls"][:2]
        self.config = config["poftr"]
        self.data_config = self.config["data"]

        self.is_generate = is_generate
        self.dataset_version = dataset_version
        self.distribution_type = distribution_type if distribution_type is not None else self.config["sim"]["distribution_type"]
        self.restart_dataset = self.data_config["restart_dataset"]
        self.ablation_type = ablation_type

        # Paths setup
        self.data_path = Path(self.config["proj"]["data_path"])
        self.webdataset_path = self.data_path / 'simulated' / "datasets" / self.distribution_type
        self.webdataset_path.mkdir(parents=True, exist_ok=True)


        # Source directories for different bands
        self.wl9_path = self.data_path / 'raw' / 'images' / str(self.data_config["wls"][0])
        self.wl9_prior_path = self.data_path / 'raw' / "priors" /str(self.data_config["wls"][0])

        self.wl11_path = self.data_path / 'raw' / 'images' / str(self.data_config["wls"][1])
        self.wl11_prior_path = self.data_path / 'raw' / "priors" / str(self.data_config["wls"][1])

        self.pan_path = self.data_path / 'raw'/ 'images' / str(self.data_config["wls"][2])
        self.pan_prior_path = self.data_path / 'raw' / "priors" / str(self.data_config["wls"][2])

        # Dataset split ratios
        self.train_ratio = self.data_config["train_ratio"]
        self.val_ratio = self.data_config["val_ratio"]
        self.test_ratio = self.data_config["test_ratio"]

        # Sampling parameters
        self.samples_num_per_image = self.data_config["samples_num_per_image"]

        # Initialize randomness
        self.seed_all()

    def seed_all(self, seed_=None):
        seed = seed_ if seed_ is not None else self.data_config["master_seed"]
        random.seed(seed)

    def run(self):
        """Run the dataset generation process."""
        train_files, val_files, test_files = self.split_dataset()
        datasets_dict = {'train': train_files, 'val': val_files, 'test': test_files}
        for dataset_type in datasets_dict:
            self.create_dataset(dataset_type=dataset_type, files_names=datasets_dict[dataset_type])

    def _get_target_r(self, N):
        from src.dataset.stats.distribution_manager import DistributionManager
        return DistributionManager.sample_targets(self.config, N, self.distribution_type)

    def _get_dm_scales(self, target_r):
        from src.dataset.stats.distribution_manager import DistributionManager
        dm = DistributionManager(config=self.config, distribution_type=self.distribution_type)
        return dm.covisibility_to_scale(target_r)
    #
    # def _generate_sample_with_target_cov(
    #         self,
    #         image_name: str,
    #         init_scale: float,
    #         target_r: float,
    #         max_tries: int = 10,
    #         eps: float = 0.03,
    # ):
    #     """
    #     Try several random views (and slightly adjust scale) until the
    #     observed co-visibility is close to target_r. Returns the best sample.
    #     """
    #     from src.dataset.stats.dataset_stats import DatasetStats
    #     best_sample = None
    #     best_err = float("inf")
    #     best_r = None
    #     scale = float(init_scale)
    #
    #     for _ in range(max_tries):
    #         clean_config = copy.deepcopy(self.config)
    #
    #         sample = MonoSample.generate(
    #             config=clean_config,
    #             image_name=image_name,
    #             idx=0,
    #             sim_scale=scale,
    #             wls=self.wls,
    #             ablation_type=self.ablation_type,
    #         )
    #
    #         sample_dict = sample.to_dict()
    #         r_obs = float(DatasetStats.compute_co_visibility(sample_dict))
    #         err = abs(r_obs - target_r)
    #
    #         if err < best_err:
    #             best_err = err
    #             best_sample = sample
    #             best_r = r_obs
    #
    #         if err <= eps and (r_obs > 0.01 or target_r < 0.01) and r_obs != 0.0:
    #             break  # good enough
    #
    #
    #         # CASE 1: The "Emergency Rescue" (Fixes the tail at 0)
    #         # We see nothing (0.0). We are too far to the Right on your graph.
    #         # We must drastically move LEFT (Decrease Scale) to find the scene.
    #         if r_obs < 0.001 and target_r > 0.01:
    #             scale = max(0.01, scale - 0.15)
    #
    #             # CASE 2: Not enough visibility (Sample is "Too Hard")
    #         # Logic: We are too low on the Y-axis.
    #         # To go UP, we must move LEFT on the X-axis (Decrease Scale).
    #         elif r_obs < target_r:
    #             scale = max(0.01, scale - 0.5 * err)
    #
    #         # CASE 3: Too much visibility (Sample is "Too Easy")
    #         # Logic: We are too high on the Y-axis.
    #         # To go DOWN, we must move RIGHT on the X-axis (Increase Scale).
    #         else:
    #             scale = min(1.0, scale + 0.5 * err)
    #
    #
    #     # store achieved covisibility on the sample (optional but useful)
    #     best_sample.co_visibility = best_r
    #     return best_sample

    def _generate_sample_with_target_cov(
            self,
            image_name: str,
            init_scale: float,
            target_r: float,
            max_tries: int = 20,  # Give it a fair chance
            eps: float = 0.03,
    ):
        from src.dataset.stats.dataset_stats import DatasetStats

        best_sample = None
        best_err = float("inf")
        best_r = 0.0
        scale = float(init_scale)

        # 1. The Optimization Loop
        for _ in range(max_tries):
            clean_config = copy.deepcopy(self.config)

            sample = MonoSample.generate(
                config=clean_config,
                image_name=image_name,
                idx=0,
                sim_scale=scale,
                wls=self.wls,
                ablation_type=self.ablation_type,
            )

            sample_dict = sample.to_dict()
            r_obs = float(DatasetStats.compute_co_visibility(sample_dict))
            err = abs(r_obs - target_r)

            # --- SMART UPDATE (The Fix for the "Best Error" Trap) ---
            # Always prefer a valid sample (r > 0) over a zero sample,
            # even if the zero sample has a mathematically smaller error.
            is_valid = r_obs > 0.01
            best_is_valid = best_r > 0.01

            update_best = False
            if best_sample is None:
                update_best = True
            elif is_valid and not best_is_valid:
                update_best = True
            elif is_valid == best_is_valid and err < best_err:
                update_best = True

            if update_best:
                best_err = err
                best_sample = sample
                best_r = r_obs

            # --- SUCCESS CHECK ---
            if err <= eps and (r_obs > 0.01 or target_r < 0.01):
                best_sample.co_visibility = r_obs
                return best_sample

            # --- CORRECTION LOGIC ---
            # 1. Emergency Rescue
            if r_obs < 0.001 and target_r > 0.01:
                scale = max(0.01, scale - 0.2)
            # 2. Too Hard -> Zoom In (Decrease Scale)
            elif r_obs < target_r:
                scale = max(0.01, scale - 0.5 * err)
            # 3. Too Easy -> Zoom Out (Increase Scale)
            else:
                scale = min(1.0, scale + 0.5 * err)

        # 2. THE ZERO BAN (The Safety Net)
        # If we exit the loop and the best we have is still garbage (0.0),
        # and we actually WANTED a valid sample...
        if best_r < 0.01 and target_r > 0.01:
            # FORCE an easy sample. Do not let the zero pass.
            clean_config = copy.deepcopy(self.config)
            sample = MonoSample.generate(
                config=clean_config,
                image_name=image_name,
                idx=0,
                sim_scale=0.1,  # Safe scale, usually ~80% overlap
                wls=self.wls,
                ablation_type=self.ablation_type,
            )
            # Recalculate stats for the fallback
            sample.co_visibility = float(DatasetStats.compute_co_visibility(sample.to_dict()))
            return sample

        # 3. Return Best
        best_sample.co_visibility = best_r
        return best_sample

    def create_dataset(self, dataset_type, files_names=None):
        """Generate and save samples in WebDataset tar shards using .npz format."""
        config = self.config
        # Prepare file list and counts
        files_names = files_names[: int(len(files_names) * self.data_config["debug_ratio"])]
        num_of_samples = len(files_names) * self.samples_num_per_image

        logger.info(
            f"Creating {self.dataset_version} {dataset_type} WebDataset, ablation - {self.ablation_type} "
            f"total samples: {num_of_samples} in {self.webdataset_path}"
        )

        target_r = self._get_target_r(num_of_samples)
        dm_scales = self._get_dm_scales(target_r)

        pbar = tqdm(total=num_of_samples, desc=f"creating {dataset_type}")

        # ShardWriter setup
        BATCH_SIZE = 100
        REFRESH_EVERY = 500
        if self.ablation_type != "standard":
            if self.ablation_type not in self.dataset_version:
                self.dataset_version = f"{self.dataset_version}_{self.ablation_type}"
            shard_dir = self.webdataset_path / 'ablations' / self.dataset_version / dataset_type
        else:
            shard_dir = self.webdataset_path / self.dataset_version / dataset_type
        shard_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = shard_dir / "shard-%06d.tar"
        output_pattern_posix = f"file:{output_pattern.as_posix()}"
        sink = wds.ShardWriter(output_pattern_posix, maxcount=BATCH_SIZE, verbose=False)

        samples_written = 0
        for img_name in files_names:
            for _ in range(self.samples_num_per_image):
                # 1) Generate and prepare sample
                clean_config = copy.deepcopy(self.config)
                init_scale = float(dm_scales[samples_written])
                r_tgt = float(target_r[samples_written])
                sample = self._generate_sample_with_target_cov(
                    image_name=img_name,
                    init_scale=init_scale,
                    target_r=r_tgt,
                    max_tries=6,  # or from config
                    eps=self.config["sim"].get("cov_eps", 0.03),
                )

                sample_dict = sample.to_dict()

                # Add camera intrinsics
                sample_dict['K0'] = get_K_from_cfg(config)
                sample_dict['K1'] = get_K_from_cfg(config)

                #add scene idx
                sample_dict['idx'] = str(int(img_name.split('.')[0]) + self.samples_num_per_image)

                # 2) Serialize sample_dict into .npz in-memory
                with io.BytesIO() as npz_buffer:
                    # convert all values to NumPy arrays if not already
                    converted = {k: (v.cpu().numpy() if hasattr(v, "cpu") else np.array(v))
                                 for k, v in sample_dict.items()}

                    np.savez(npz_buffer, **converted)
                    npz_buffer.seek(0)
                    npz_data = npz_buffer.read()  # Read the data

                key = f"{samples_written:08d}"
                sink.write({
                    "__key__": key,
                    "npz": npz_data
                })

                samples_written += 1
                pbar.update(1)

                # 3) Periodic GC and stats update
                if samples_written % REFRESH_EVERY == 0:
                    ram_perc = psutil.virtual_memory().percent
                    pbar.set_postfix({'RAM': f"{ram_perc:.1f}%"})

        # Finalize
        sink.close()
        pbar.close()

        print(f"Created and saved WebDataset '{dataset_type}' successfully!")

    @staticmethod
    def _save_to_csv(files_list, path):
        """Save a list of filenames to CSV (one per line)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            for file_name in files_list:
                writer.writerow([file_name])

    @staticmethod
    def _load_list_from_csv(path: Path) -> List[str]:
        """Load a CSV list of filenames."""
        with open(path, 'r', newline='') as f:
            return [row[0] for row in csv.reader(f)]

    def _retrieve_trainval_files(self, test_files: List[str]) -> List[str]:
        """Retrieve training and validation files, excluding test files."""
        if (self.webdataset_path / 'trainval.csv').exists():
            # If restart is enabled and trainval.csv exists, load it
            trainval_files = self._load_list_from_csv(self.webdataset_path / 'trainval.csv')
            return trainval_files

        else:
            all_files = set(os.listdir(self.wl9_path))
            trainval_files = list(all_files - set(test_files))

            # Ensure the files are sorted for consistency
            trainval_files.sort()

            # Save the trainval files to CSV
            self._save_to_csv(trainval_files, self.webdataset_path / 'trainval.csv')

        return trainval_files


    def split_dataset(self):
        test_files = self._load_list_from_csv(self.webdataset_path.parent / 'canonical_test.csv')
        trainval_files = self._retrieve_trainval_files(test_files=test_files)
        random.shuffle(trainval_files)
        train_files = trainval_files[:int(self.train_ratio * len(trainval_files))]
        val_files = trainval_files[int(self.train_ratio * len(trainval_files)):]
        return train_files, val_files, test_files



def main():
    config = lower_config(get_config())
    os.chdir(config["poftr"]["proj"]["cwd"])
    wls_pairs = [["9um", "11um"], ["9um", "pan"], ["11um", "pan"]]
    distribution_types = ["truncnorm"]
    ablation_types = ["standard"] # "noisy_priors", "zeroed_priors",
    for distribution_type in distribution_types:
        for wls in wls_pairs:
            for ablation_type in ablation_types:
                dataset_version = f"{wls[0]}_{wls[1]}"
                generator = MonoDsGen(config, distribution_type=distribution_type, dataset_version=dataset_version, wls=wls, is_generate=True, ablation_type=ablation_type)
                generator.run()

if __name__ == "__main__":
    main()
