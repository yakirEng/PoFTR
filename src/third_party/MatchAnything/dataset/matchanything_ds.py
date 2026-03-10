import os
import io
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from pathlib import Path
import json
from webdataset.compat import WebDataset
from webdataset.shardlists import split_by_node, split_by_worker
from torch.utils.data import IterableDataset


class MatchAnythingDs(IterableDataset):
    """
    Dataset for MatchAnything (Zero-Shot).
    - STRICTLY uses config['data']['root_dir'] to find .tar shards.
    """

    def __init__(
            self,
            config,
            dataset_version: str,
            dataset_type: str = "test",
            is_analysis_mode: bool = False,
    ):
        self.config = config
        self.dataset_version = dataset_version
        self.dataset_type = dataset_type
        self.is_analysis_mode = is_analysis_mode

        # 1. ImageNet Normalization
        self.imagenet_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 2. Resolve Path (STRICT)
        # The eval script MUST inject 'root_dir' before calling this.
        if 'data' not in config or 'root_dir' not in config['data']:
            raise ValueError(f"MatchAnythingDs config missing 'root_dir'. Config keys: {config.keys()}")

        self.base = Path(config['data']['root_dir'])

        self.stats = self.load_stats(self.base.parent)

        # 3. Find Shards
        self.shards = sorted(self.base.glob("*.tar"))

        if not self.shards:
            raise FileNotFoundError(f"No .tar shards found in {self.base}. \n"
                                    f"Ensure eval.py is constructing the path correctly.")

        print(f"[{dataset_version}] Loaded {len(self.shards)} shards from: {self.base}")

        # Convert to WDS format
        self.shard_list = self._make_wds_shards(self.shards)

    def _make_wds_shards(self, shards_paths):
        out = []
        for p in shards_paths:
            rp = p.resolve()
            if os.name == "nt":  # Windows
                out.append("file:" + rp.as_posix())
            else:  # Linux/Mac
                out.append(os.fspath(rp))
        return out

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

    def _safe_to_tensor(self, sample):
        """Converts numpy parts to torch float tensors, handles strings safely."""
        keys_to_remove = []

        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                # 1. Handle String Arrays
                if v.dtype.kind in {'U', 'S'}:
                    # Convert numpy strings to Python strings (list) so collate accepts them
                    # Or just remove them if you don't need them.
                    try:
                        sample[k] = str(v)
                    except:
                        keys_to_remove.append(k)
                    continue

                # 2. Convert Numeric Arrays to Tensor
                sample[k] = torch.from_numpy(v).float()

            elif isinstance(v, (int, float)):
                sample[k] = torch.tensor(v).float()

        # Cleanup any keys that failed conversion
        for k in keys_to_remove:
            sample.pop(k, None)

        return sample

    def _robust_normalize_tau2(self, img_tensor, valid_mask):
        """
        Percentile-based min-max normalization.
        Uses the provided 'valid_mask' to calculate statistics only on valid pixels.
        """
        img_tensor = img_tensor.float()

        # Ensure mask matches image dimensions (handle broadcasting)
        # img: (1, H, W) or (3, H, W)
        # mask: (H, W) or (1, H, W)
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0)

        # If mask is missing (safety check), assume full validity
        if valid_mask is None:
            valid_mask = torch.ones_like(img_tensor, dtype=torch.bool)

        # Safety: If image is completely empty/invalid, return as is
        if valid_mask.sum() == 0:
            return img_tensor

        # 1. Extract ONLY valid pixels for statistics
        #    We must mask all channels if input has multiple (though usually it's 1 here)
        if valid_mask.shape[0] != img_tensor.shape[0]:
            valid_mask = valid_mask.expand_as(img_tensor)

        valid_pixels = img_tensor[valid_mask]

        # 2. Calculate quantiles on valid data
        val_min = torch.quantile(valid_pixels, 0.01)
        val_max = torch.quantile(valid_pixels, 0.99)

        # 3. Clamp the original image
        img_tensor = torch.clamp(img_tensor, min=val_min, max=val_max)

        # 4. Normalize
        denom = val_max - val_min
        if denom < 1e-6:
            denom = 1.0

        norm_img = (img_tensor - val_min) / denom

        # 5. Re-apply Mask
        #    Forces invalid pixels (padding) to be exactly 0.0
        norm_img[~valid_mask] = 0.0

        return norm_img

    def _process_sample(self, data):
        # 1. Decode
        key, npz_bytes = data

        with io.BytesIO(npz_bytes) as f:
            npz_data = np.load(f, allow_pickle=True)
            sample = {k: npz_data[k] for k in npz_data.files}

        sample['sample_id'] = key
        # 2. To Tensor
        sample = self._safe_to_tensor(sample)

        # 3. Process Images
        #    We iterate pairs (image0, pixel_mask0) and (image1, pixel_mask1)
        for i in [0, 1]:
            img_key = f'image{i}'
            mask_key = f'mask{i}'

            img = sample[img_key]
            if img.ndim == 2: img = img.unsqueeze(0)

            # Fetch the mask for this image
            mask = sample.get(mask_key)
            if mask is not None:
                mask = mask.bool()
                if mask.ndim == 2: mask = mask.unsqueeze(0)
            else:
                # Fallback if mask is missing (should not happen in PoFTR)
                raise ValueError(f"Missing expected mask key: {mask_key}")

            # Robust Norm (Using the explicit mask)
            img = self._robust_normalize_tau2(img, mask)

            # Channel Expansion (1 -> 3)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            # ImageNet Norm
            sample[img_key] = self.imagenet_norm(img)

        # 4. Process Masks mask0/mask1- coarse, pixel_mask0/pixel_mask1 - pixel-level
        if sample.get('mask0') is not None:
            sample['pixel_mask0'] = sample['mask0'].bool()
            sample['pixel_mask1'] = sample['mask1'].bool()

            [ts_mask_0, ts_mask_1] = \
                F.interpolate(torch.stack([sample['mask0'], sample['mask1']], dim=0)[None].float(),
                              scale_factor=1/8,
                              mode='nearest',
                              recompute_scale_factor=False)[0].bool()
                # Update coarse masks for Transformer/Loss
            sample.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})


        # 5. Cleanup
        if not self.is_analysis_mode:
            for k in ['phys0', 'phys1', 'co_visibility', 'valid_pixels']:
                sample.pop(k, None)

        return sample

    def __iter__(self):
        pipeline = (
            WebDataset(
                self.shard_list,
                shardshuffle=False,
                nodesplitter=split_by_node,
                workersplitter=split_by_worker,
                resampled=False
            )
            .to_tuple("__key__", "npz")
            .map(self._process_sample)
        )
        return iter(pipeline)