import os
from pathlib import Path
import random
import gc
import io
import numpy as np
import psutil
import webdataset as wds
from tqdm import tqdm
import csv
from typing import List, Tuple
from loguru import logger


from src.dataset.physical_model.petit_s.dataset.petits_sample import PetitsSample
from src.dataset.physical_model.petit_s.utils.petits_configs import get_config, lower_config

class DatasetGenPetitS:
    """Generate dataset for the Petit-S physical model."""
    def __init__(self, config, is_generate=False):
        # Configuration and control flags
        self.data_config = config['data']
        self.config = config
        self.is_generate = is_generate
        self.restart_dataset = self.data_config['restart_dataset']

        # Paths setup
        self.raw_data_path = Path(self.data_config['raw_data_path'])
        self.data_path = Path(self.raw_data_path.parent) / 'petits'
        self.webdataset_path = self.data_path / 'webdataset'
        self.webdataset_path.mkdir(parents=True, exist_ok=True)

        # Source directories for different bands
        self.source_path = Path(r"F:\yakirs_thesis\data\aerial_ds\land_frames\pan")
        self.wls = ["9um", "11um"]
        self.wl0_path = self.raw_data_path / str(self.data_config['wls'][0])
        self.wl1_path = self.raw_data_path / str(self.data_config['wls'][1])
        self.pan_path = self.raw_data_path / str(self.data_config['wls'][2])

        # Dataset split ratios
        self.train_ratio = self.data_config['train_ratio']
        self.val_ratio = self.data_config['val_ratio']
        self.test_ratio = self.data_config['test_ratio']
        self.canonical_test_ratio = self.data_config['test_ratio']

        # Sampling parameters
        self.samples_num_per_image = self.data_config['samples_num_per_image']


    def seed_all(self, seed):
        random.seed(seed)

    def get_canonical_test(self):
        test_csv_path = self.webdataset_path / "canonical_test.csv"
        if self.restart_dataset or not test_csv_path.exists():
            self.create_canonical_test()
        # load existing canonical test
        with open(test_csv_path, "r", newline="") as f:
            canonical_test = [row[0] for row in csv.reader(f)]
        test_set = set(canonical_test)

        return test_set

    def create_canonical_test(self):
        """Create a canonical test dataset - this dataset would be the test dataset
         for petit-s and the following matching algorithms."""
        master_files = os.listdir(self.wl0_path)
        rng = random.Random(int(self.data_config['master_seed']))
        idx = list(range(len(master_files)))
        rng.shuffle(idx)
        n_test = max(1, int(round(self.test_ratio * len(master_files))))
        test_idx = set(idx[:n_test])
        canonical_test = [master_files[i] for i in sorted(test_idx)]
        test_csv_path = self.webdataset_path / "canonical_test.csv"
        self._save_to_csv(canonical_test, test_csv_path)

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

    def _partition_kfold(self, files: List[str], k: int, seed: int) -> List[List[str]]:
        """
        Create K disjoint folds from `files`, with sizes as even as possible.
        The shuffle is controlled by `seed` so folds are reproducible per seed.
        Returns: list of length K, each element is a list of filenames (one fold).
        """
        rng = random.Random(seed)
        idx = list(range(len(files)))
        rng.shuffle(idx)

        folds = []
        n = len(idx)
        base = n // k  # minimum size per fold
        extra = n % k  # first 'extra' folds get +1 element
        start = 0
        for i in range(k):
            sz = base + (1 if i < extra else 0)
            fold_idx = idx[start:start + sz]
            folds.append([files[j] for j in fold_idx])
            start += sz
        return folds

    def _save_folds(self, folds: List[List[str]]):
        """
        Save the folds to CSV files in the webdataset path.
        Each fold is saved as a separate file named 'train.csv', 'val.csv', etc.
        """
        base = self.webdataset_path  # parent dir = "webdataset"
        base.mkdir(parents=True, exist_ok=True)

        for i, fold in enumerate(folds):
            fold_dir = base / f"seed_{self.data_config.master_seed}" / f"fold_{i}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            val_files = fold
            train_files = [item for j, fold_j in enumerate(folds) if j != i for item in fold_j]
            self._save_to_csv(train_files, fold_dir / "train.csv")
            self._save_to_csv(val_files, fold_dir / "val.csv")

    def _get_trainval_pool(self, test_set: set) -> List[str]:
        master_files = os.listdir(self.wl0_path)
        trainval_pool = [f for f in master_files if f not in set(test_set)]
        if len(trainval_pool) == 0:
            raise RuntimeError("No files left for train/val after removing canonical test.")
        return trainval_pool

    def create_folds(self):
        """Create K-folds for the dataset."""
        self.seed_all(self.data_config['master_seed'])
        test_set = self.get_canonical_test()
        trainval_pool = self._get_trainval_pool(test_set)
        folds = self._partition_kfold(trainval_pool, k=self.data_config['k_folds'], seed=self.data_config['master_seed'])
        self._save_folds(folds)

    def load_fold(self, fold_idx) -> Tuple[List[str], List[str], List[str]]:
        logger.info(f"Processing fold {fold_idx + 1}/{self.data_config['k_folds']}")
        # Load train and validation files for this fold
        train_csv_path = self.webdataset_path / f"seed_{self.data_config['master_seed']}" / f"fold_{fold_idx}" / "train.csv"
        val_csv_path = self.webdataset_path / f"seed_{self.data_config['master_seed']}" / f"fold_{fold_idx}" / "val.csv"

        if not train_csv_path.exists() or not val_csv_path.exists():
            raise FileNotFoundError(f"Fold {fold_idx} files not found.")

        train_files = self._load_list_from_csv(train_csv_path)
        val_files = self._load_list_from_csv(val_csv_path)
        test_files = list(self.get_canonical_test())

        if not train_files or not val_files:
            raise ValueError(f"No training or validation files found for fold {fold_idx}.")
        return train_files, val_files, test_files

    def run(self):
        """Run the dataset generation process."""
        for wl in self.wls:
            for fold_idx in range(self.data_config['k_folds']):
                train_files, val_files, test_files = self.load_fold(fold_idx)
                datasets_dict = {'train': train_files, 'val': val_files, 'test': test_files}
                for dataset_type in datasets_dict:
                    self.create_dataset(fold_idx= fold_idx, dataset_type=dataset_type, wl=wl, files_names=datasets_dict[dataset_type])


    def create_dataset(self, fold_idx, wl, dataset_type, files_names=None):
        """Generate and save samples in WebDataset tar shards using .npz format."""
        # Prepare file list and counts
        files_names = files_names[: int(len(files_names) * self.data_config['debug_ratio'])]
        num_of_samples = len(files_names)

        logger.info(
            f"Creating Petit_S {dataset_type} WebDataset for wl {wl} fold {fold_idx} with {num_of_samples} samples: "
        )

        gc.disable()
        pbar = tqdm(total=num_of_samples, desc=f"creating petit_s {dataset_type}")

        # ShardWriter setup
        BATCH_SIZE = 100
        REFRESH_EVERY = 500
        shard_dir = self.webdataset_path / f'seed_{self.data_config["master_seed"]}' / f'fold_{fold_idx}' / wl / dataset_type
        shard_dir.mkdir(parents=True, exist_ok=True)
        output_basename = "shard-%06d.tar"

        if os.name == "nt":
            # Resolve the directory path
            resolved_dir = shard_dir.resolve()
            # Format as a POSIX file URI
            formatted_dir_uri = f"file:{resolved_dir.as_posix()}"
            # Re-join with the basename pattern
            formatted_pattern = f"{formatted_dir_uri}/{output_basename}"
        else:
            # On Linux/macOS, a standard absolute path is fine
            formatted_pattern = str(shard_dir.resolve() / output_basename)

        sink = wds.ShardWriter(formatted_pattern, maxcount=BATCH_SIZE, verbose=False)

        samples_written = 0
        for image_name in files_names:
            for _ in range(self.samples_num_per_image):
                # 1) Generate and prepare sample
                sample = PetitsSample.generate(
                    config=self.config,
                    image_name=image_name,
                    idx=samples_written,
                )
                sample_dict = sample.to_dict()

                # 2) Serialize sample_dict into .npz in-memory
                npz_buffer = io.BytesIO()
                # convert all values to NumPy arrays if not already
                converted = {k: (v.cpu().numpy() if hasattr(v, "cpu") else np.array(v)) for k, v in
                             sample_dict.items()}
                np.savez(npz_buffer, **converted)
                npz_buffer.seek(0)

                key = f"{samples_written:08d}"
                sink.write({
                    "__key__": key,
                    "npz": npz_buffer.read()
                })

                del sample, sample_dict, npz_buffer, converted
                samples_written += 1
                pbar.update(1)

                # 3) Periodic GC and stats update
                if samples_written % REFRESH_EVERY == 0:
                    gc.collect()
                    ram_perc = psutil.virtual_memory().percent
                    pbar.set_postfix({'RAM': f"{ram_perc:.1f}%"})

        # Finalize
        sink.close()
        pbar.close()
        gc.enable()

        print(f"Created and saved WebDataset '{dataset_type}' successfully!")


def main():
    config = lower_config(get_config())
    os.chdir(config['proj']['cwd'])
    petit_s_dataset_gen = DatasetGenPetitS(config=config)
    # petit_s_dataset_gen.create_folds()
    petit_s_dataset_gen.run()


if __name__ == "__main__":
    main()

