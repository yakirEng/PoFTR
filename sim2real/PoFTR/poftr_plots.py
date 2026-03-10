import gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

from src.configs.poftr_configs import get_config
from src.utils.misc import lower_config
from sim2real.PoFTR.poftr_utils import prepare_poftr_batch, run_poftr, load_poftr_model, safe_center_crop_numpy
from sim2real.general_helpers import plot_sim2real_matches, load_stats
from sim2real.physical_model.petit_gan_pm import ThermalRegress


# ==========================================
# CONFIG
# ==========================================

ROOT_DATA  = Path(r"F:\yakirs_thesis\data\aerial_ds\land_frames")
BASE_SAVE  = Path(r"F:\yakirs_thesis\thesis_code\sim2real\figs\poftr")
CKPT_BASE  = Path(r"F:\yakirs_thesis\thesis_code\checkpoints\best")
COEFF_DIR  = Path(r"F:\yakirs_thesis\thesis_code\sim2real\physical_model\coeffs")
STATS_BASE = Path(r"F:\yakirs_thesis\thesis_code\data\simulated\datasets\truncnorm")
csv_dir    = Path(r"F:\yakirs_thesis\thesis_code\sim2real\csv_outputs")

DIST_THRESH = 10.0
N_SAMPLES   = 30
CROP_SIZE   = 256
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

wl0, wl1 = "9000nm", "11000nm"   # ← change here for other pairs


# ==========================================
# HELPERS
# ==========================================

def nm_to_um(wl: str) -> str:
    """e.g. '9000nm' -> '9um', '11000nm' -> '11um', 'pan' -> 'pan'"""
    if wl.endswith("nm") and wl[:-2].isdigit():
        return f"{int(wl[:-2]) // 1000}um"
    return wl


# ==========================================
# SETUP
# ==========================================

current_save_dir = BASE_SAVE / f"{wl0}_{wl1}"
current_save_dir.mkdir(parents=True, exist_ok=True)

# Checkpoint uses um convention + phys subfolder
ckpt_path = CKPT_BASE / f"{nm_to_um(wl0)}_{nm_to_um(wl1)}" / "xoftr" / "phys" / "best.ckpt"

# CSV & sampling
df = pd.read_csv(csv_dir / f"sorted_matched_pairs_{wl0}_{wl1}.csv")
print(f"Loaded {len(df)} pairs from CSV for {wl0} <-> {wl1}.")

df_filtered = df[df['Distance_meters'] < DIST_THRESH].copy()
print(f"Filtered pairs (distance < {DIST_THRESH}m): {len(df_filtered)}")

n = min(N_SAMPLES, len(df_filtered))
df_sample = df_filtered.sample(n=n, random_state=42).reset_index(drop=True)

# Stats
global_stats = load_stats(nm_to_um(wl0), nm_to_um(wl1), stats_base=STATS_BASE)

# Physical model
phys_model = ThermalRegress()
phys_model.load(COEFF_DIR, wl0=wl0, wl1=wl1)

# VRAM diagnostics
if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_vram  = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} | Total VRAM: {total_vram:.1f} GB | Free: {free_vram:.1f} GB")

# Load model ONCE
print(f"Checkpoint: {ckpt_path}")
config = lower_config(get_config())
config['poftr']['phys']['use_phys'] = True  # PoFTR uses physics backbone

poftr_model = load_poftr_model(config, checkpoint_path=ckpt_path, device=DEVICE)
print(f"Processing {n} pairs for {wl0} <-> {wl1}...")
print(f"Saving to: {current_save_dir}")


# ==========================================
# MAIN LOOP
# ==========================================

for i, row in tqdm(df_sample.iterrows(), total=n):
    idx0 = int(row[wl0])
    idx1 = int(row[wl1])
    dist = row['Distance_meters']

    batch_poftr = kpts0 = kpts1 = conf = None

    try:

        f0 = np.load(ROOT_DATA / wl0 / f"{idx0}.npz")
        f1 = np.load(ROOT_DATA / wl1 / f"{idx1}.npz")

        img0 = safe_center_crop_numpy(f0["image"].astype(np.float32), CROP_SIZE)
        img1 = safe_center_crop_numpy(f1["image"].astype(np.float32), CROP_SIZE)

        fpa0 = np.array(float(f0["fpa"]) / 100.0)
        fpa1 = np.array(float(f1["fpa"]) / 100.0)

        # Physical priors
        prior0 = phys_model.predict(img0, t_fpa=fpa0, direction='rad_to_temp', band=wl0).astype(np.float32)
        prior1 = phys_model.predict(img1, t_fpa=fpa1, direction='rad_to_temp', band=wl1).astype(np.float32)

        batch_poftr = prepare_poftr_batch(
            img0_np=img0, img1_np=img1,
            p0_np=prior0, p1_np=prior1,
            global_stats=global_stats,
            crop_size=CROP_SIZE,
            debug_mode=False,
        )
        kpts0, kpts1, conf = run_poftr(batch_poftr, model=poftr_model, device=DEVICE)

        save_name = f"pair_{i:03d}_{wl0}_{idx0}_{wl1}_{idx1}_d{dist:.1f}m.png"
        save_path = current_save_dir / save_name

        plot_sim2real_matches(
            img0, img1, kpts0, kpts1,
            model_name=f"PoFTR | {wl0}-{wl1} | {i+1}/{n} | d={dist:.1f}m",
            save_path=str(save_path)
        )

    except Exception as e:
        print(f"Error on pair {idx0}-{idx1}: {e}")

    finally:
        del batch_poftr, kpts0, kpts1, conf
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

print(f"\nFinished processing {wl0}-{wl1}. Images saved to: {current_save_dir.name}")