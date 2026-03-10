import gc
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from src.configs.poftr_configs import get_config
from src.utils.misc import lower_config
from sim2real.general_helpers import compute_ransac_metrics, load_stats
from sim2real.XoFTR.xoftr_utils import prepare_xoftr_batch, run_xoftr, load_xoftr_model, safe_center_crop_numpy


def nm_to_um(wl: str) -> str:
    """e.g. '9000nm' -> '9um', '11000nm' -> '11um', 'pan' -> 'pan'"""
    if wl.endswith("nm") and wl[:-2].isdigit():
        return f"{int(wl[:-2]) // 1000}um"
    return wl


def evaluate_all_pairs_xoftr(
    csv_path,
    data_dir,
    config,
    wl0: str,
    wl1: str,
    ckpt_base: Path,
    stats_base: Path,
    distance_threshold=10.0,
    crop_size=256,
    ransac_thresh=3.0,
    model_name="XoFTR",
    save_results=True,
):
    data_dir  = Path(data_dir)
    ckpt_base = Path(ckpt_base)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # VRAM diagnostics
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_vram  = torch.cuda.mem_get_info()[0] / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} | Total VRAM: {total_vram:.1f} GB | Free: {free_vram:.1f} GB")

    # 1. Load stats
    global_stats = load_stats(nm_to_um(wl0), nm_to_um(wl1), stats_base=stats_base)

    # 2. Resolve checkpoint path (uses um convention)
    ckpt_wl0  = nm_to_um(wl0)
    ckpt_wl1  = nm_to_um(wl1)
    ckpt_path = ckpt_base / f"{ckpt_wl0}_{ckpt_wl1}" / "xoftr" / "no_phys" / "best.ckpt"
    print(f"Checkpoint: {ckpt_path}")

    # 3. Load model ONCE
    config['poftr']['phys']['use_phys'] = False
    model = load_xoftr_model(config, checkpoint_path=ckpt_path, device=device)

    # 4. Load and filter pairs
    df = pd.read_csv(csv_path)
    df_filtered = df[df['Distance_meters'] < distance_threshold].reset_index(drop=True)
    print(f"Found {len(df_filtered)} pairs with distance < {distance_threshold}m")

    # 5. Loop
    results = []
    failed  = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"{model_name} | {wl0}<->{wl1}"):
        idx0    = int(row[wl0])
        idx1    = int(row[wl1])
        pair_id = f"{idx0}_{idx1}"

        batch = kpts0 = kpts1 = conf = None

        try:
            f0 = np.load(data_dir / f"{wl0}/{idx0}.npz")
            f1 = np.load(data_dir / f"{wl1}/{idx1}.npz")

            img0 = safe_center_crop_numpy(f0["image"].astype(np.float32), crop_size)
            img1 = safe_center_crop_numpy(f1["image"].astype(np.float32), crop_size)

            batch = prepare_xoftr_batch(
                img0_np=img0, img1_np=img1,
                global_stats=global_stats, crop_size=crop_size,
            )
            kpts0, kpts1, conf = run_xoftr(batch, model=model, device=device)

            n_inliers, inlier_ratio = compute_ransac_metrics(kpts0, kpts1, ransac_thresh=ransac_thresh)

            results.append({
                'pair_id':      pair_id,
                f'idx_{wl0}':   idx0,
                f'idx_{wl1}':   idx1,
                'distance_m':   row['Distance_meters'],
                'n_matches':    len(kpts0),
                'n_inliers':    n_inliers,
                'inlier_ratio': inlier_ratio,
                'mean_conf':    float(conf.mean()) if len(conf) > 0 else 0.0,
            })



        except Exception as e:
            print(f"  [FAILED] {pair_id}: {e}")
            failed.append(pair_id)

        finally:
            del batch, kpts0, kpts1, conf
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Incremental save every 50 pairs
        if save_results and len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(f"{model_name}_{wl0}_{wl1}_partial.csv", index=False)

    # 6. Aggregate
    results_df = pd.DataFrame(results)
    summary = {
        'model':                 model_name,
        'wl0':                   wl0,
        'wl1':                   wl1,
        'n_pairs_evaluated':     len(results_df),
        'n_pairs_failed':        len(failed),
        'mean_inlier_ratio':     float(results_df['inlier_ratio'].mean()),
        'std_inlier_ratio':      float(results_df['inlier_ratio'].std()),
        'median_inlier_ratio':   float(results_df['inlier_ratio'].median()),
        'mean_n_inliers':        float(results_df['n_inliers'].mean()),
        'std_n_inliers':         float(results_df['n_inliers'].std()),
        'pairs_with_no_matches': int((results_df['n_matches'] == 0).sum()),
        'pairs_with_no_inliers': int((results_df['n_inliers'] == 0).sum()),
    }

    # 7. Save
    if save_results:
        out_dir = Path("evaluation_results")
        out_dir.mkdir(exist_ok=True)
        results_df.to_csv(out_dir / f"{model_name}_{wl0}_{wl1}_detailed.csv", index=False)
        with open(out_dir / f"{model_name}_{wl0}_{wl1}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out_dir}/")

    # 8. Print summary
    print("\n" + "=" * 50)
    print(f"SUMMARY — {model_name} | {wl0} <-> {wl1}")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return results_df, summary



if __name__ == "__main__":
    # --- Paths ---
    DATA_DIR   = Path(r"F:\yakirs_thesis\data\aerial_ds\land_frames")
    CKPT_BASE  = Path(r"F:\yakirs_thesis\thesis_code\checkpoints\best")
    STATS_BASE = Path(r"F:\yakirs_thesis\thesis_code\data\simulated\datasets\truncnorm")
    CSV_DIR    = Path(r"F:\yakirs_thesis\thesis_code\sim2real\csv_outputs")

    # --- Pair to evaluate ---
    wl0, wl1 = "9000nm", "11000nm"

    # --- Config ---
    config = lower_config(get_config())

    # --- Run ---
    results_df, summary = evaluate_all_pairs_xoftr(
        csv_path           = CSV_DIR / f"sorted_matched_pairs_{wl0}_{wl1}.csv",
        data_dir           = DATA_DIR,
        config             = config,
        wl0                = wl0,
        wl1                = wl1,
        ckpt_base          = CKPT_BASE,
        stats_base         = STATS_BASE,
        distance_threshold = 10.0,
        crop_size          = 256,
        ransac_thresh      = 3.0,
        model_name         = "XoFTR",
        save_results       = True,
    )