import gc
import json
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from src.configs.poftr_configs import get_config
from src.utils.misc import lower_config
from sim2real.general_helpers import compute_ransac_metrics, load_stats
from sim2real.XoFTR.xoftr_utils import prepare_xoftr_batch, run_xoftr, load_xoftr_model, safe_center_crop_numpy
from sim2real.PoFTR.poftr_utils import prepare_poftr_batch, run_poftr, load_poftr_model
from sim2real.matchanything.ma_utils import prepare_matchanything_batch
from sim2real.physical_model.petit_gan_pm import ThermalRegress
from src.third_party.MatchAnything.model.match_anything import MatchAnythingHFWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nm_to_um(wl: str) -> str:
    """e.g. '9000nm' -> '9um', '11000nm' -> '11um', 'pan' -> 'pan'"""
    if wl.endswith("nm") and wl[:-2].isdigit():
        return f"{int(wl[:-2]) // 1000}um"
    return wl


def resolve_ckpt(ckpt_base, wl0, wl1, model_name, use_phys):
    """Resolve checkpoint path from base dir."""
    phys_dir = 'phys' if use_phys else 'no_phys'
    return Path(ckpt_base) / f"{nm_to_um(wl0)}_{nm_to_um(wl1)}" / model_name / phys_dir / 'best.ckpt'


def load_image_pair(data_dir, wl0, wl1, idx0, idx1, crop_size):
    """Load and crop a pair of .npz images."""
    f0 = np.load(Path(data_dir) / wl0 / f"{idx0}.npz")
    f1 = np.load(Path(data_dir) / wl1 / f"{idx1}.npz")
    img0 = safe_center_crop_numpy(f0["image"].astype(np.float32), crop_size)
    img1 = safe_center_crop_numpy(f1["image"].astype(np.float32), crop_size)
    return f0, f1, img0, img1


def aggregate_summary(model_name, wl0, wl1, results, failed):
    """Compute summary statistics from per-pair results."""
    df = pd.DataFrame(results)
    return {
        'model':                 model_name,
        'wl0':                   wl0,
        'wl1':                   wl1,
        'n_pairs_evaluated':     len(df),
        'n_pairs_failed':        len(failed),
        'mean_inlier_ratio':     float(df['inlier_ratio'].mean()),
        'std_inlier_ratio':      float(df['inlier_ratio'].std()),
        'median_inlier_ratio':   float(df['inlier_ratio'].median()),
        'mean_n_inliers':        float(df['n_inliers'].mean()),
        'std_n_inliers':         float(df['n_inliers'].std()),
        'pairs_with_no_matches': int((df['n_matches'] == 0).sum()),
        'pairs_with_no_inliers': int((df['n_inliers'] == 0).sum()),
    }, df


def save_outputs(summary, results_df, out_dir, model_name, wl0, wl1):
    """Save per-pair CSV and summary JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / f"{model_name}_{wl0}_{wl1}_detailed.csv", index=False)
    with open(out_dir / f"{model_name}_{wl0}_{wl1}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_dir}/")


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Per-model evaluation functions
# ---------------------------------------------------------------------------

def eval_xoftr(cfg_eval, config, wl0, wl1, df_filtered, global_stats, device):
    ckpt_path = resolve_ckpt(cfg_eval['checkpoint_base'], wl0, wl1, 'xoftr', use_phys=False)
    config['poftr']['phys']['use_phys'] = False
    model   = load_xoftr_model(config, checkpoint_path=ckpt_path, device=device)
    results = []
    failed  = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"XoFTR | {wl0}<->{wl1}"):
        idx0, idx1 = int(row[wl0]), int(row[wl1])
        batch = kpts0 = kpts1 = conf = None
        try:
            _, _, img0, img1 = load_image_pair(
                cfg_eval['sim2real_data_root'], wl0, wl1, idx0, idx1, cfg_eval['crop_size']
            )
            batch        = prepare_xoftr_batch(img0_np=img0, img1_np=img1,
                                               global_stats=global_stats,
                                               crop_size=cfg_eval['crop_size'])
            kpts0, kpts1, conf = run_xoftr(batch, model=model, device=device)
            n_inliers, inlier_ratio = compute_ransac_metrics(
                kpts0, kpts1, ransac_thresh=cfg_eval['ransac_thresh']
            )
            results.append({
                'pair_id': f"{idx0}_{idx1}", f'idx_{wl0}': idx0, f'idx_{wl1}': idx1,
                'distance_m': row['Distance_meters'], 'n_matches': len(kpts0),
                'n_inliers': n_inliers, 'inlier_ratio': inlier_ratio,
                'mean_conf': float(conf.mean()) if len(conf) > 0 else 0.0,
            })
        except Exception as e:
            print(f"  [FAILED] {idx0}_{idx1}: {e}")
            failed.append(f"{idx0}_{idx1}")
        finally:
            del batch, kpts0, kpts1, conf
            cleanup()

    return aggregate_summary('XoFTR', wl0, wl1, results, failed)


def eval_poftr(cfg_eval, config, wl0, wl1, df_filtered, global_stats, phys_model, device):
    ckpt_path = resolve_ckpt(cfg_eval['checkpoint_base'], wl0, wl1, 'xoftr', use_phys=True)
    config['poftr']['phys']['use_phys'] = True
    model   = load_poftr_model(config, checkpoint_path=ckpt_path, device=device)
    results = []
    failed  = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"PoFTR | {wl0}<->{wl1}"):
        idx0, idx1 = int(row[wl0]), int(row[wl1])
        batch = kpts0 = kpts1 = conf = None
        try:
            f0, f1, img0, img1 = load_image_pair(
                cfg_eval['sim2real_data_root'], wl0, wl1, idx0, idx1, cfg_eval['crop_size']
            )
            fpa0   = np.array(float(f0["fpa"]) / 100.0)
            fpa1   = np.array(float(f1["fpa"]) / 100.0)
            prior0 = phys_model.predict(img0, t_fpa=fpa0, direction='rad_to_temp', band=wl0).astype(np.float32)
            prior1 = phys_model.predict(img1, t_fpa=fpa1, direction='rad_to_temp', band=wl1).astype(np.float32)
            batch  = prepare_poftr_batch(img0_np=img0, img1_np=img1, p0_np=prior0, p1_np=prior1,
                                         global_stats=global_stats, crop_size=cfg_eval['crop_size'])
            kpts0, kpts1, conf = run_poftr(batch, model=model, device=device)
            n_inliers, inlier_ratio = compute_ransac_metrics(
                kpts0, kpts1, ransac_thresh=cfg_eval['ransac_thresh']
            )
            results.append({
                'pair_id': f"{idx0}_{idx1}", f'idx_{wl0}': idx0, f'idx_{wl1}': idx1,
                'distance_m': row['Distance_meters'], 'fpa_wl0': float(fpa0), 'fpa_wl1': float(fpa1),
                'n_matches': len(kpts0), 'n_inliers': n_inliers, 'inlier_ratio': inlier_ratio,
                'mean_conf': float(conf.mean()) if len(conf) > 0 else 0.0,
            })
        except Exception as e:
            print(f"  [FAILED] {idx0}_{idx1}: {e}")
            failed.append(f"{idx0}_{idx1}")
        finally:
            del batch, kpts0, kpts1, conf
            cleanup()

    return aggregate_summary('PoFTR', wl0, wl1, results, failed)


def eval_matchanything(cfg_eval, wl0, wl1, df_filtered, device):
    model = MatchAnythingHFWrapper(model_id="zju-community/matchanything_eloftr").to(device)
    model.eval()
    results = []
    failed  = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"MatchAnything | {wl0}<->{wl1}"):
        idx0, idx1 = int(row[wl0]), int(row[wl1])
        try:
            _, _, img0, img1 = load_image_pair(
                cfg_eval['sim2real_data_root'], wl0, wl1, idx0, idx1, cfg_eval['crop_size']
            )
            batch = prepare_matchanything_batch(img0_np=img0, img1_np=img1, crop_size=cfg_eval['crop_size'])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(batch)
            kpts0 = outputs['mkpts0_f'].view(-1, 2).cpu().numpy()
            kpts1 = outputs['mkpts1_f'].view(-1, 2).cpu().numpy()
            conf  = outputs['mconf'].view(-1).cpu().numpy()
            n_inliers, inlier_ratio = compute_ransac_metrics(
                kpts0, kpts1, ransac_thresh=cfg_eval['ransac_thresh']
            )
            results.append({
                'pair_id': f"{idx0}_{idx1}", f'idx_{wl0}': idx0, f'idx_{wl1}': idx1,
                'distance_m': row['Distance_meters'], 'n_matches': len(kpts0),
                'n_inliers': n_inliers, 'inlier_ratio': inlier_ratio,
                'mean_conf': float(conf.mean()) if len(conf) > 0 else 0.0,
            })
        except Exception as e:
            print(f"  [FAILED] {idx0}_{idx1}: {e}")
            failed.append(f"{idx0}_{idx1}")

    return aggregate_summary('MatchAnything', wl0, wl1, results, failed)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(all_summaries):
    """Print unified results table matching Table 2 in the paper."""
    headers = ['Dataset', 'Model', 'Inlier Ratio (%) ↑', '# Inliers ↑']
    col_w   = 22
    sep     = '+' + '+'.join(['-' * (col_w + 2)] * len(headers)) + '+'

    print(f"\n{'='*60}")
    print("Sim-to-Real Generalization Results")
    print(f"{'='*60}")
    print(sep)
    print('|' + '|'.join(f' {h:^{col_w}} ' for h in headers) + '|')
    print(sep)

    rows = []
    for (wl0, wl1), model_summaries in all_summaries.items():
        dataset_label = f"{nm_to_um(wl0)}-{nm_to_um(wl1)}"
        for summary in model_summaries:
            inlier_ratio = summary['mean_inlier_ratio'] * 100
            std_ratio    = summary['std_inlier_ratio'] * 100
            n_inliers    = summary['mean_n_inliers']
            std_inliers  = summary['std_n_inliers']
            row = [
                dataset_label,
                summary['model'],
                f"{inlier_ratio:.1f} ± {std_ratio:.1f}",
                f"{n_inliers:.1f} ± {std_inliers:.1f}",
            ]
            rows.append(row)
            print('|' + '|'.join(f' {v:^{col_w}} ' for v in row) + '|')
        print(sep)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    with open('configs/eval_config.yaml', 'r') as f:
        cfg_eval = yaml.safe_load(f)

    config = lower_config(get_config())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    band_pairs  = cfg_eval.get('sim2real_band_pairs', [
        ('9000nm', 'pan'),
        ('11000nm', 'pan'),
        ('9000nm', '11000nm'),
    ])
    all_summaries = {}

    for wl0, wl1 in band_pairs:
        print(f"\n{'='*60}")
        print(f"Band pair: {wl0} <-> {wl1}")
        print(f"{'='*60}")

        # Load CSV and filter by distance
        csv_path    = Path(cfg_eval['sim2real_csv_dir']) / f"sorted_matched_pairs_{wl0}_{wl1}.csv"
        df          = pd.read_csv(csv_path)
        df_filtered = df[df['Distance_meters'] < cfg_eval['dist_thresh']].reset_index(drop=True)
        print(f"Loaded {len(df_filtered)} pairs (distance < {cfg_eval['dist_thresh']}m)")

        # Load shared stats
        stats_base   = Path(cfg_eval['stats_base'])
        global_stats = load_stats(nm_to_um(wl0), nm_to_um(wl1), stats_base=stats_base)

        # Load physical model for PoFTR
        coeff_dir  = Path(cfg_eval['coeff_dir'])
        phys_model = ThermalRegress()
        phys_model.load(coeff_dir, wl0=wl0, wl1=wl1)

        pair_summaries = []

        # --- MatchAnything ---
        summary_ma, df_ma = eval_matchanything(cfg_eval, wl0, wl1, df_filtered, device)
        pair_summaries.append(summary_ma)
        save_outputs(summary_ma, df_ma,
                     Path(cfg_eval['results_dir']) / f"{wl0}_{wl1}", 'MatchAnything', wl0, wl1)

        # --- XoFTR ---
        summary_xoftr, df_xoftr = eval_xoftr(cfg_eval, config, wl0, wl1, df_filtered, global_stats, device)
        pair_summaries.append(summary_xoftr)
        save_outputs(summary_xoftr, df_xoftr,
                     Path(cfg_eval['results_dir']) / f"{wl0}_{wl1}", 'XoFTR', wl0, wl1)

        # --- PoFTR ---
        summary_poftr, df_poftr = eval_poftr(cfg_eval, config, wl0, wl1, df_filtered,
                                              global_stats, phys_model, device)
        pair_summaries.append(summary_poftr)
        save_outputs(summary_poftr, df_poftr,
                     Path(cfg_eval['results_dir']) / f"{wl0}_{wl1}", 'PoFTR', wl0, wl1)

        all_summaries[(wl0, wl1)] = pair_summaries

    # --- Print & Save unified table ---
    rows    = print_results_table(all_summaries)
    headers = ['Dataset', 'Model', 'Inlier Ratio (%) ↑', '# Inliers ↑']
    out_csv = Path(cfg_eval['results_dir']) / 'sim2real_results.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=headers).to_csv(out_csv, index=False)
    print(f"\nUnified results saved to: {out_csv}")