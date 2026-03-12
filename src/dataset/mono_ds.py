import os
import io
import numpy as np
from webdataset.compat import WebDataset
from webdataset.shardlists import split_by_node, split_by_worker
import torch
import torch.nn.functional as F
import pathlib
import sys
from pathlib import Path
from torch.utils.data import IterableDataset
import json
from itertools import islice


from src.dataset.dataset_helper import (
    apply_augmentation,
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

class MonochromeDs(IterableDataset):
    def __init__(
        self,
        config,
        distribution_type: str = None,
        dataset_version: str = "9um_11um",
        dataset_type: str = "train",
        ablation_version: str = None,
        is_tensor=True,
        is_augment=True,
        is_normalize=True,
        sanity_check=None,
        is_analysis_mode=False,
    ):
        self.config       = config
        self.distribution_type = distribution_type if distribution_type is not None else config['sim']['distribution_type']
        self.dataset_version = dataset_version
        self.ablation_version = ablation_version if ablation_version is not None else config['data']['ablation_version']
        self.dataset_type = dataset_type
        self.coarse_scale = config["data"]["coarse_scale"]
        self.is_tensor    = is_tensor
        self.is_augment   = is_augment and (dataset_type == "train")
        self.is_normalize= is_normalize
        self.is_analysis_mode = is_analysis_mode
        self.sanity_check = sanity_check if sanity_check is not None else config['run']['sanity_check']

        # 1) find your shard files explicitly
        self.base = Path(self.config['proj']['cwd']) / "data" / "AeroSync" / self.dataset_version / dataset_type
        shards = sorted(self.base.glob("shard-*.tar"))
        if not shards:
            raise FileNotFoundError(f"No shards found in {self.base}")

        self.shards = make_wds_shards(shards) # [p.as_posix() for p in shards]

        self.shuffle_size = max(self.config['train']['batch_size'] * 100, 10_000)

        self.stats = self.load_stats(self.base.parent)

        print(f"[PoFTR Dataset] {self.dataset_type.upper()} loading from: {self.base}")

    def load_stats(self, stats_path):
        try:
            with open(stats_path/ 'stats.json' , 'r') as f:
                stats = json.load(f)
            print("stats.json loaded successfully!")

        except FileNotFoundError:
            print(f"Error: Could not find stats.json at the expected path: {stats_path}")
        except json.JSONDecodeError:
            print(f"Error: The file {stats_path} is not a valid JSON file.")
        return stats


    def _process_sample(self, data):
        # Decode the sample from bytes to a dictionary
        key, npz_bytes = data
        with io.BytesIO(npz_bytes) as f:
            npz_data = np.load(f, allow_pickle=True)
            sample = {k: npz_data[k] for k in npz_data.files}

        sample['sample_id'] = key

        if self.is_augment and self.dataset_type == "train":
            sample = apply_augmentation(sample, self.config)

        if self.is_normalize:
            sample = normalize_sample(sample, self.stats)

        if self.is_tensor:
            sample = to_tensor(sample)

        if sample['mask0'] is not None:  # img_padding is True

            sample.update({'pixel_mask0': sample['mask0'], 'pixel_mask1': sample['mask1']})

            if self.is_tensor:
                [ts_mask_0, ts_mask_1] = \
                F.interpolate(torch.stack([sample['mask0'], sample['mask1']], dim=0)[None].float(),
                              scale_factor=1 / self.coarse_scale,
                              mode='nearest',
                              recompute_scale_factor=False)[0].bool()
            else:
                [ts_mask_0, ts_mask_1] = np.stack([sample['mask0'], sample['mask1']], axis=0)[:, ::self.coarse_scale,
                                         ::self.coarse_scale]

            # Update coarse masks for Transformer/Loss
            sample.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        if self.config['phys']['use_phys']:
            if self.is_tensor:
                # --- TENSOR PATH ---
                pm0 = sample['pixel_mask0']
                if pm0.ndim == 2: pm0 = pm0.unsqueeze(0)

                pm1 = sample['pixel_mask1']
                if pm1.ndim == 2: pm1 = pm1.unsqueeze(0)

                sample['image0'] = torch.cat([sample['image0'], sample['phys0'], pm0.float()], dim=0)
                sample['image1'] = torch.cat([sample['image1'], sample['phys1'], pm1.float()], dim=0)
            else:
                # --- NUMPY PATH ---
                pm0 = sample['pixel_mask0']

                if pm0.ndim == 2: pm0 = pm0[None, ...]

                pm1 = sample['pixel_mask1']
                if pm1.ndim == 2: pm1 = pm1[None, ...]


                sample['image0'] = np.concatenate([sample['image0'], sample['phys0'], pm0.astype(np.float32)], axis=0)
                sample['image1'] = np.concatenate([sample['image1'], sample['phys1'], pm1.astype(np.float32)], axis=0)

        if not self.is_analysis_mode:
            sample.pop('phys0')
            sample.pop('phys1')
            sample.pop('co_visibility')
            sample.pop('valid_pixels')

        return sample

    def __len__(self):
        if self.sanity_check:
            return self.config['run']['sanity_dataset_size']
        elif self.dataset_type == "train":
            return self.config['data']['train_size']
        elif self.dataset_type == "val":
            return self.config['data']['val_size']
        elif self.dataset_type == "test":
            return self.config['data']['test_size']
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
            pipeline = islice(pipeline, self.config['run']['sanity_dataset_size'])

        return iter(pipeline)


