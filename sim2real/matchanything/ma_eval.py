import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sim2real.general_helpers import compute_ransac_metrics
from sim2real.matchanything.ma_utils import prepare_matchanything_batch,  safe_center_crop_numpy
from src.third_party.MatchAnything.model.match_anything import MatchAnythingHFWrapper


def evaluate_all_pairs_matchanything(
    csv_path,
    data_dir,
    wl0,  # e.g., '11000nm' or '9000nm'
    wl1,  # e.g., 'pan'
    distance_threshold=10.0,
    crop_size=256,
    ransac_thresh=3.0,
    model_name="MatchAnything",
    save_results=True,
):

    # 1. Load and filter pairs based on the provided CSV
    df = pd.read_csv(data_dir / csv_path)
    df_filtered = df[df['Distance_meters'] < distance_threshold].reset_index(drop=True)
    print(f"Found {len(df_filtered)} pairs for {wl0}-{wl1} with distance < {distance_threshold}m")

    # 2. Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MatchAnythingHFWrapper(
        model_id="zju-community/matchanything_eloftr"
    ).to(device)
    model.eval()

    # 3. Loop over pairs
    results = []
    failed = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"{model_name}_{wl0}_{wl1}"):
        idx0 = int(row[wl0])
        idx1 = int(row[wl1])
        pair_id = f"{idx0}_{idx1}"

        try:
            # Load raw files dynamically using wl0 and wl1
            f0 = np.load(data_dir / f"{wl0}/{idx0}.npz")
            f1 = np.load(data_dir / f"{wl1}/{idx1}.npz")

            # Extract data safely (handles both 'image' and file-list indexing)
            img0_raw = f0['image'] if 'image' in f0 else f0[f0.files[0]]
            img1_raw = f1['image'] if 'image' in f1 else f1[f1.files[0]]

            # Crop using the shared helper
            img0 = safe_center_crop_numpy(img0_raw.astype(np.float32), crop_size)
            img1 = safe_center_crop_numpy(img1_raw.astype(np.float32), crop_size)

            # Prepare batch and move to device
            batch = prepare_matchanything_batch(img0_np=img0, img1_np=img1, crop_size=crop_size)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Inference
            with torch.no_grad():
                outputs = model(batch)

            kpts0 = outputs['mkpts0_f'].view(-1, 2).cpu().numpy()
            kpts1 = outputs['mkpts1_f'].view(-1, 2).cpu().numpy()
            conf = outputs['mconf'].view(-1).cpu().numpy()

            # RANSAC — Ensure this matches the 3.0px AeroSync protocol
            n_inliers, inlier_ratio = compute_ransac_metrics(kpts0, kpts1, ransac_thresh=ransac_thresh)

            results.append({
                'pair_id': pair_id,
                f'idx_{wl0}': idx0,
                f'idx_{wl1}': idx1,
                'distance_m': row['Distance_meters'],
                'n_matches': len(kpts0),
                'n_inliers': n_inliers,
                'inlier_ratio': inlier_ratio,
                'mean_conf': float(conf.mean()) if len(conf) > 0 else 0.0,
            })

        except Exception as e:
            print(f"  [FAILED] {pair_id}: {e}")
            failed.append(pair_id)
            continue

    # 4. Aggregate results
    results_df = pd.DataFrame(results)

    # Avoid errors if no pairs were found/succeeded
    if len(results_df) == 0:
        print("No pairs were successfully evaluated.")
        return None, None

    summary = {
        'model': model_name,
        'config': f"{wl0}_{wl1}",
        'n_pairs_evaluated': len(results_df),
        'n_pairs_failed': len(failed),
        'mean_inlier_ratio': float(results_df['inlier_ratio'].mean()),
        'std_inlier_ratio': float(results_df['inlier_ratio'].std()),
        'mean_n_inliers': float(results_df['n_inliers'].mean()),
        'std_n_inliers': float(results_df['n_inliers'].std()),
        'pairs_with_no_inliers': int((results_df['n_inliers'] == 0).sum()),
    }

    # 5. Save results using a specific sub-folder to avoid overwriting bands
    if save_results:
        out_dir = Path(f"evaluation_results/{wl0}_{wl1}")
        out_dir.mkdir(exist_ok=True)
        results_df.to_csv(out_dir / f"{model_name}_detailed.csv", index=False)
        with open(out_dir / f"{model_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out_dir}/")

    # 6. Print summary
    print("\n" + "="*50)
    print(f"SUMMARY — {model_name}")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return results_df, summary

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    wl0, wl1 = "9000nm", "11000nm"
    data_dir = Path(r"F:\yakirs_thesis\data\aerial_ds\land_frames")
    csv_path = Path(r"F:\yakirs_thesis\thesis_code\sim2real\csv_outputs")
    csv_file = f"sorted_matched_pairs_{wl0}_{wl1}.csv"

    results_df_ma, summary_ma = evaluate_all_pairs_matchanything(
        csv_path           = csv_path / csv_file,
        data_dir           = data_dir,
        distance_threshold = 10.0,
        ransac_thresh      = 3.0,
        model_name         = "MatchAnything",
        wl0                = wl0,
        wl1                = wl1,
        save_results       = True
    )

