import os

# [FIX] Prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import json
import logging
import shutil
import tempfile
import gc
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from src.dataset.physical_model.petit_s.utils.petits_configs import get_config, lower_config
from src.dataset.physical_model.petit_s.utils.petits_data_module import PetitSDataModule
from src.dataset.physical_model.petit_s.model.pl_petit_s import PL_PetitS

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ---------- I/O HELPERS ----------
def _load_csv_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    with open(path, "r", newline="") as f:
        return [row[0] for row in csv.reader(f)]


def _best_ckpt_in(dirpath: Path) -> Path:
    if not dirpath.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {dirpath}")

    cands = sorted(dirpath.glob("*.ckpt"))
    if not cands:
        raise FileNotFoundError(f"No checkpoints found under {dirpath}")

    best_ckpt = None
    best_metric = float('inf')
    target_metrics = ["val_lpips=", "val_lpips-", "lpips", "mae=", "mae-"]

    for ckpt in cands:
        name = ckpt.name
        if name == "last.ckpt": continue

        for metric_key in target_metrics:
            if metric_key in name:
                try:
                    part = name.split(metric_key)[1]
                    num_str = part.split("-")[0].split(".ckpt")[0]
                    val = float(num_str)
                    if val < best_metric:
                        best_metric = val
                        best_ckpt = ckpt
                    break
                except (IndexError, ValueError):
                    continue

    if best_ckpt:
        logger.info(f"Selected best checkpoint based on metric ({best_metric}): {best_ckpt.name}")
        return best_ckpt

    last_ckpt = dirpath / "last.ckpt"
    if last_ckpt.exists():
        logger.info(f"Selected 'last.ckpt' (fallback): {last_ckpt.name}")
        return last_ckpt

    logger.warning(f"Using last alphabetical: {cands[-1].name}")
    return cands[-1]


# ---------- LOADER / MODEL ----------
def load_dataloader(config, split: str, wl: str, fold_idx: int) -> Iterable[Dict]:
    dm = PetitSDataModule(config=config, splits=[split], wl=wl, fold_idx=fold_idx)
    dm.setup()
    if split == "inference" and hasattr(dm, "inference_dataloader"):
        return dm.inference_dataloader()
    elif split == "test" and hasattr(dm, "test_dataloader"):
        return dm.test_dataloader()
    elif split == "val":
        return dm.val_dataloader()
    else:
        loader_name = f"{split}_dataloader"
        if hasattr(dm, loader_name):
            return getattr(dm, loader_name)()
        raise ValueError(f"No dataloader found for split='{split}'")


def load_model_for_fold(config, fold_idx: int, wl: str, device: torch.device) -> PL_PetitS:
    ckpt_dir = Path(config["proj"]["checkpoints_path"]) / config["proj"]["base_model"] / wl / f"fold_{fold_idx}"
    ckpt_path = _best_ckpt_in(ckpt_dir)
    logger.info(f"Loading model: {wl} | Fold {fold_idx} | {ckpt_path.name}")

    # Load to CPU first
    model = PL_PetitS.load_from_checkpoint(
        ckpt_path,
        config=config,
        fold_idx=fold_idx,
        map_location='cpu'
    )
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)
    return model


def load_stats(stats_path: Path, fold_idx: int) -> Dict[str, float]:
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load stats from {stats_path}: {e}")

    fold_key = f'fold_{fold_idx}'
    if fold_key not in stats:
        raise KeyError(f"Fold '{fold_key}' not found.")
    return stats[fold_key]


def _denormalize(T_normed: np.ndarray, stats) -> np.ndarray:
    try:
        mean = float(stats['pan_temp_map']['mean'])
        std = float(stats['pan_temp_map']['std'])
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid stats structure: {e}")
    return T_normed * std + mean


# ---------- UNIFIED INFERENCE CORE ----------
@torch.no_grad()
def generate_priors(config, wl: str, fold_idx: int, split: str, device: torch.device,
                    is_canonical_test: bool = False) -> None:
    seed = int(config["data"]["master_seed"])
    out_root = Path(config["proj"]["data_path"]) / "raw" / "priors" / f"{wl}"
    out_root.mkdir(parents=True, exist_ok=True)

    if is_canonical_test:
        split_csv = Path(config["proj"]["data_path"]) / "petits" / "webdataset" / "canonical_test.csv"
        loader_split_name = "test"
        desc_label = "TEST (Champion)"
    else:
        split_csv = Path(config["proj"][
                             "data_path"]) / "petits" / "webdataset" / f"seed_{seed}" / f"fold_{fold_idx}" / f"{split}.csv"
        loader_split_name = split
        desc_label = f"VAL (Fold {fold_idx})"

    must_predict_ids = set([Path(x).stem for x in _load_csv_list(split_csv)])

    model = None
    try:
        stats_path = Path(config["proj"]["data_path"]) / "petits" / "webdataset" / f"seed_{seed}" / f"stats_{wl}.json"
        fold_stats = load_stats(stats_path, fold_idx)

        loader = load_dataloader(config, split=loader_split_name, wl=wl, fold_idx=fold_idx)
        model = load_model_for_fold(config, fold_idx, wl, device)


        prec_val = 32
        use_amp = False
        autocast_dtype = torch.float32
        # [FIX] Precision Logic (Now forced to 16-mixed via config override)
        # prec_val = getattr(config["train"], "precision", 32)
        # use_amp = str(prec_val) in ("16", "16-mixed", "bf16-mixed")
        # autocast_dtype = torch.bfloat16 if str(prec_val) == "bf16-mixed" else torch.float16

        logger.info(f"Precision: {prec_val} | AMP Enabled: {use_amp} | Dtype: {autocast_dtype}")

        seen_ids = set()
        pbar = tqdm(loader, desc=f"[{wl}] {desc_label}", ncols=120)

        for batch_idx, batch in enumerate(pbar):
            image_names = batch.get("image_name")
            if not image_names: raise KeyError("Batch missing 'image_name'")
            image_names = [Path(str(x)).stem for x in image_names]

            def _to_dev(x):
                return x.to(device, non_blocking=True) if torch.is_tensor(x) else x

            mono = _to_dev(batch.get("mono_image"))
            t_fpa = _to_dev(batch.get("t_fpa"))

            # [DEBUG] Log input shape on first batch
            if batch_idx == 0:
                logger.info(f"Input Shape: {mono.shape} (Type: {mono.dtype})")

            # Forward with AMP
            context = torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_amp else torch.no_grad()
            with context:
                pred = model(mono=mono, t_fpa=t_fpa)

            T = pred.get("T_pred_c", pred.get("pred")) if isinstance(pred, dict) else pred
            if T.ndim == 4 and T.shape[1] == 1: T = T[:, 0]

            T_batch_np = T.detach().float().cpu().numpy()

            for i, img_id in enumerate(image_names):
                if img_id not in must_predict_ids: continue

                T_final = _denormalize(T_batch_np[i], fold_stats)
                save_path = out_root / f"{img_id}.npy"
                np.save(save_path, T_final)
                seen_ids.add(img_id)

        if missing := must_predict_ids - seen_ids:
            logger.warning(f"[{desc_label}] Missing {len(missing)} IDs.")
        else:
            logger.info(f"[{desc_label}] Successfully generated {len(seen_ids)} priors.")

    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[{desc_label}] GPU Memory cleared.")


def determine_best_folds_from_csv(csv_path: Path, metric: str = "val_lpips") -> Dict[str, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found at {csv_path}")

    best_folds = {}
    best_scores = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wl = row["wl"]
            fold = int(row["fold_idx"])
            score = float(row[metric])

            if wl not in best_scores: best_scores[wl] = float('inf')
            if score < best_scores[wl]:
                best_scores[wl] = score
                best_folds[wl] = fold

    logger.info(f"Auto-selected champions based on {metric}: {best_folds}")
    return best_folds


def run_production_priors():
    try:
        # [FIX] Global disable
        torch.set_grad_enabled(False)

        config = lower_config(get_config())
        os.chdir(config["proj"]["cwd"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



        if "train" not in config: config["train"] = {}


        results_csv_path = Path(
            config["proj"]["cwd"]) / "src" / "dataset" / "physical_model" / "petit_s" / "inference" / "results.csv"
        best_folds_map = determine_best_folds_from_csv(results_csv_path)

        k_folds = 5
        wls = ["9um", "11um"]

        for wl in wls:
            logger.info(f"=== PROCESSING BAND: {wl} ===")
            for fold_idx in range(k_folds):
                generate_priors(config, wl, fold_idx, "val", device, False)

            best_fold = best_folds_map.get(wl)
            logger.info(f"--- Generating Test Set Priors (Champion: Fold {best_fold}) ---")
            generate_priors(config, wl, best_fold, "test", device, True)

        logger.info("[DONE] All production priors generated successfully.")

    except Exception as e:
        logger.critical(f"Process failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_production_priors()