import os
import io
import numpy as np
from webdataset.compat import WebDataset
from webdataset.shardlists import split_by_node, split_by_worker
from pathlib import Path
from torch.utils.data import IterableDataset
import tarfile
import math
import json
from itertools import islice


from src.dataset.physical_model.petit_s.dataset.dataset_helper import (
    normalize_sample,
    to_tensor,
)


def make_wds_shards(shards_paths):
    out = []
    for p in shards_paths:
        rp = p.resolve()
        if os.name == "nt":  # Windows
            out.append("file:" + rp.as_posix())
        else:                 # Linux/macOS
            out.append(os.fspath(rp))
    return out

class PetitsDs(IterableDataset):
    def __init__(
        self,
        config,
        dataset_type="train",
        is_tensor=True,
        is_augment=False,
        is_normalize=True,
        sanity_check=None,
        wl=None,
        fold_idx=None,
    ):
        self.config       = config
        self.dataset_type = dataset_type
        self.is_tensor    = is_tensor
        self.is_augment   = is_augment
        self.is_normalize= is_normalize
        self.sanity_check = sanity_check if sanity_check is not None else config['run']['sanity_check']
        self.wl = wl if wl is not None else config['data']['wl']
        self.fold_idx = fold_idx if fold_idx is not None else config['run']['fold_idx']


        base = (Path(self.config['proj']['data_path']) / config['proj']['base_model'] / "webdataset" /
                f"seed_{self.config['data']['master_seed']}" / f"fold_{self.fold_idx}" / self.wl / self.dataset_type)
        shards = sorted(base.glob("shard-*.tar"))
        if not shards:
            print("CWD: ", os.getcwd())
            raise FileNotFoundError(f"No shards found in {base}")

        self.shards = make_wds_shards(shards)

        self.shuffle_size = self.shuffle_size = max(self.config['train']['batch_size'] * 100, 10_000)

        self.stats = self.load_stats(base.parents[2])

    def load_stats(self, stats_path):
        try:
            with open(stats_path/ f'stats_{self.wl}.json' , 'r') as f:
                stats = json.load(f)
            print(f"stats_{self.wl}.json loaded successfully!")

        except FileNotFoundError:
            print(f"Error: Could not find stats.json at the expected path: {stats_path}")
        except json.JSONDecodeError:
            print(f"Error: The file {stats_path} is not a valid JSON file.")
        return stats['fold_' + str(self.fold_idx)]

    def _process_sample(self, data, for_train=False):

        # Decode the sample from bytes to a dictionary
        key, npz_bytes = data
        with io.BytesIO(npz_bytes) as f:
            npz_data = np.load(f, allow_pickle=True)
            sample = {k: npz_data[k] for k in npz_data.files}

        if for_train:
            keys_to_pop = ['image_name', 'idx']
            for key in keys_to_pop:
                sample.pop(key, None)

        if self.is_normalize:
            sample = normalize_sample(sample, self.stats)

        if self.is_tensor:
            sample = to_tensor(sample)

        return sample

    def __len__(self):
        if self.config['run']['sanity_check']:
            return self.config['run']['sanity_dataset_size']
        elif self.dataset_type == "train":
            return self.config['run']['num_train_samples']
        elif self.dataset_type == "val":
            return self.config['run']['num_val_samples']
        elif self.dataset_type == "test":
            return self.config['run']['num_test_samples']
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")


    def __iter__(self):
        pipeline = (
            WebDataset(
                self.shards,
                shardshuffle=False,
                nodesplitter=split_by_node,
                workersplitter=split_by_worker,
                resampled=True if self.dataset_type == "train" else False
            )
            .shuffle(self.shuffle_size)
            .to_tuple("__key__", "npz")
            .map(self._process_sample)
        )

        # Apply sanity check if needed
        if self.sanity_check:
            pipeline = islice(pipeline, self.config["run"]["sanity_dataset_size"])

        return iter(pipeline)




