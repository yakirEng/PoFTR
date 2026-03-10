import gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

from sim2real.matchanything.ma_utils import prepare_matchanything_batch, safe_center_crop_numpy, run_matchanything
from sim2real.general_helpers import plot_sim2real_matches
from src.third_party.MatchAnything.model.match_anything import MatchAnythingHFWrapper

# --- Config ---
ROOT_DATA = Path(r"F:\yakirs_thesis\data\aerial_ds\land_frames")
BASE_SAVE = Path(r"F:\yakirs_thesis\thesis_code\sim2real\figs\matchanything")

DIST_THRESH = 10.0
N_SAMPLES   = 30
CROP_SIZE   = 256
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
wl0, wl1    = "9000nm", "11000nm"
csv_dir     = Path(r"F:\yakirs_thesis\thesis_code\sim2real\csv_outputs")

current_save_dir = BASE_SAVE / f"{wl0}_{wl1}"
current_save_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_dir / f"sorted_matched_pairs_{wl0}_{wl1}.csv")
print(f"Loaded {len(df)} pairs from CSV for {wl0} <-> {wl1}.")

df_filtered = df[df['Distance_meters'] < DIST_THRESH].copy()
print(f"Filtered pairs (distance < {DIST_THRESH}m): {len(df_filtered)}")

n = min(N_SAMPLES, len(df_filtered))
df_sample = df_filtered.sample(n=n, random_state=42).reset_index(drop=True)

# --- VRAM diagnostics before model load ---
if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_vram  = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} | Total VRAM: {total_vram:.1f} GB | Free: {free_vram:.1f} GB")

print("Loading MatchAnything model once...")
ma_model = MatchAnythingHFWrapper(model_id="zju-community/matchanything_eloftr").to(DEVICE)
ma_model.eval()

print(f"Processing {n} pairs for {wl0} <-> {wl1}...")
print(f"Saving to: {current_save_dir}")

# --- Main loop ---
for i, row in tqdm(df_sample.iterrows(), total=n):
    idx0 = int(row[wl0])
    idx1 = int(row[wl1])
    dist = row['Distance_meters']

    # FIX: All locals that hold GPU tensors are declared here so `finally` can always delete them
    batch_ma = kpts0_ma = kpts1_ma = conf_ma = None

    try:
        # FIX: Log VRAM per iteration to catch gradual leaks early
        if torch.cuda.is_available():
            used_vram = torch.cuda.memory_allocated() / 1e9
            print(f"  [{i}] VRAM in use: {used_vram:.2f} GB")

        file0 = np.load(ROOT_DATA / wl0 / f"{idx0}.npz")
        file1 = np.load(ROOT_DATA / wl1 / f"{idx1}.npz")

        img0 = safe_center_crop_numpy(file0["image"], CROP_SIZE)
        img1 = safe_center_crop_numpy(file1["image"], CROP_SIZE)

        # FIX: img0/img1 are already cropped — prepare_matchanything_batch no longer re-crops
        batch_ma = prepare_matchanything_batch(img0, img1, crop_size=CROP_SIZE)
        kpts0_ma, kpts1_ma, conf_ma = run_matchanything(batch_ma, device=DEVICE, wl0=wl0, wl1=wl1, model=ma_model)

        save_name = f"pair_{i:03d}_{wl0}_{idx0}_{wl1}_{idx1}_d{dist:.1f}m.png"
        save_path = current_save_dir / save_name

        plot_sim2real_matches(
            img0, img1, kpts0_ma, kpts1_ma,
            model_name=f"MatchAnything | {wl0}-{wl1} | {i+1}/{n} | d={dist:.1f}m",
            save_path=str(save_path)
        )

    except Exception as e:
        print(f"Error on pair {idx0}-{idx1}: {e}")

    finally:
        # FIX: Always clean up GPU memory regardless of success or exception
        del batch_ma, kpts0_ma, kpts1_ma, conf_ma
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

print(f"\nFinished processing {wl0}-{wl1}. Images saved to: {current_save_dir.name}")