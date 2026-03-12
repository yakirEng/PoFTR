import yaml
import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path

from src.configs.poftr_configs import get_config
from src.utils.misc import lower_config
from src.PoFTR.lightning.data_module import SATDataModule
from src.PoFTR.poftr import PoFTR
from src.PoFTR.lightning.pl_poftr import PL_PoFTR


def load_poftr_model(config, checkpoint_path, device='cuda'):
    """Load PoFTR model from checkpoint."""
    print(f"Loading PoFTR | base: {config['poftr']['proj']['base_model']} | "
          f"phys: {config['poftr']['phys']['use_phys']}")
    model = PoFTR(config).to(device)
    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['state_dict']
    clean_state_dict = {
        (k.replace("model.model.", "model.") if k.startswith("model.model.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    print("PoFTR model ready.")
    return model


def get_checkpoint_path(cfg_eval, band_pair):
    """Resolve checkpoint path from config."""
    use_phys   = cfg_eval.get('use_phys', True)
    phys_dir   = 'phys' if use_phys else 'no_phys'
    model_name = cfg_eval['model_name']
    ckpt_base  = Path(cfg_eval['checkpoint_base'])
    return ckpt_base / 'best' / band_pair / model_name / phys_dir / 'best.ckpt'


def run_eval(config, cfg_eval, band_pair, device):
    """Run evaluation for a single band pair and return metrics."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {band_pair}")
    print(f"{'='*50}")

    # Override dataset version for this band pair
    config['poftr']['data']['dataset_version'] = band_pair

    # Resolve and validate checkpoint path
    ckpt_path = get_checkpoint_path(cfg_eval, band_pair)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Setup datamodule (test split only)
    data_module = SATDataModule(config['poftr'], splits=('test',))
    data_module.setup()

    # Load model
    poftr_model = load_poftr_model(config, ckpt_path, device=device)

    # Wrap in Lightning module for test loop
    pl_model = PL_PoFTR(config, data_module)
    pl_model.model = poftr_model

    # Run test
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    results = trainer.test(pl_model, datamodule=data_module)

    return results[0] if results else {}


def print_results_table(all_results):
    """Print a formatted results table to terminal."""
    headers = ['Dataset', 'Pose Success ↑', 'MMA@3 ↑', '# Inliers ↑']
    col_w   = 20
    sep     = '+' + '+'.join(['-' * (col_w + 2)] * len(headers)) + '+'

    print(f"\n{'='*50}")
    print("AeroSync Benchmark Results")
    print(f"{'='*50}")
    print(sep)
    print('|' + '|'.join(f' {h:^{col_w}} ' for h in headers) + '|')
    print(sep)

    rows = []
    for band_pair, metrics in all_results.items():
        pose = metrics.get('test_Pose_Success_10px', float('nan'))
        mma  = metrics.get('test_MMA_3', float('nan'))
        inl  = metrics.get('test_Num_Matches', float('nan'))
        row  = [band_pair, f'{pose:.3f}', f'{mma:.3f}', f'{inl:.1f}']
        rows.append(row)
        print('|' + '|'.join(f' {v:^{col_w}} ' for v in row) + '|')

    print(sep)
    return rows


def save_results_csv(rows, headers, save_path):
    """Save results to CSV."""
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    # --- Load configs ---
    with open('configs/eval_config.yaml', 'r') as f:
        cfg_eval = yaml.safe_load(f)

    config = lower_config(get_config())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Override test-specific settings
    config['poftr']['test']['enable_plotting'] = False
    config['poftr']['run']['num_workers']      = cfg_eval.get('num_workers', 4)
    config['poftr']['run']['prefetch_factor']  = cfg_eval.get('prefetch_factor', 2)
    config['poftr']['train']['batch_size']     = cfg_eval.get('batch_size', 8)
    config['poftr']['phys']['use_phys']        = cfg_eval.get('use_phys', True)
    config['poftr']['proj']['base_model']      = cfg_eval.get('model_name', 'xoftr')

    band_pairs = cfg_eval.get('band_pairs', ['9um_pan', '11um_pan', '9um_11um'])

    # --- Run evaluation for all band pairs ---
    all_results = {}
    for band_pair in band_pairs:
        all_results[band_pair] = run_eval(config, cfg_eval, band_pair, device)

    # --- Print & Save ---
    headers = ['Dataset', 'Pose Success ↑', 'MMA@3 ↑', '# Inliers ↑']
    rows    = print_results_table(all_results)

    save_path = Path(cfg_eval.get('results_dir', 'evaluation_results')) / 'aerosync_results.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(rows, headers, save_path)