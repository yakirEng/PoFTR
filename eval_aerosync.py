import yaml
import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path

from src.configs.poftr_configs import get_config, get_method_config
from src.utils.misc import lower_config
from src.PoFTR.lightning.data_module import SATDataModule
from src.PoFTR.poftr import PoFTR
from src.PoFTR.lightning.pl_poftr import PL_PoFTR


def load_model(config, checkpoint_path, device='cuda'):
    """
    Universal model loader for all backbones.
    - PoFTR:              use_phys=True  (set in eval_config.yaml)
    - XoFTR/LoFTR/ASpan: use_phys=False (set in eval_config.yaml)
    """
    base_model = config['poftr']['proj']['base_model']
    use_phys   = config['poftr']['phys']['use_phys']
    print(f"Loading model | base: {base_model} | phys: {use_phys}")

    model = PoFTR(config).to(device)
    print(f"Loading weights from: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)['state_dict']
    clean_state_dict = {
        (k.replace("model.model.", "model.") if k.startswith("model.model.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    print("Model ready.")
    return model


def get_checkpoint_path(cfg_eval, band_pair, base_model, use_phys):
    """Resolve checkpoint path based on config."""
    phys_dir  = 'phys' if use_phys else 'no_phys'
    ckpt_base = Path(cfg_eval['checkpoint_base'])
    return ckpt_base / band_pair / base_model / phys_dir / 'best.ckpt'


def run_eval(config, cfg_eval, band_pair, model_label, device):
    """Run evaluation for a single model + band pair and return metrics."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_label} | {band_pair}")
    print(f"{'='*50}")

    # Override dataset version for this band pair
    config['poftr']['data']['dataset_version'] = band_pair

    # Resolve and validate checkpoint path
    base_model = config['poftr']['proj']['base_model']
    use_phys   = config['poftr']['phys']['use_phys']
    ckpt_path  = get_checkpoint_path(cfg_eval, band_pair, base_model, use_phys)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Setup datamodule (test split only)
    data_module = SATDataModule(config['poftr'], splits=('test',))
    data_module.setup()

    # Load model and wrap in Lightning module
    poftr_model        = load_model(config, ckpt_path, device=device)
    pl_model           = PL_PoFTR(config, data_module)
    pl_model.model     = poftr_model

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
    """Print a formatted results table to terminal.
    all_results: {model_label: {band_pair: metrics}}
    """
    headers = ['Model', 'Dataset', 'Pose Success ↑', 'MMA@3 ↑', '# Inliers ↑']
    col_w   = 20
    sep     = '+' + '+'.join(['-' * (col_w + 2)] * len(headers)) + '+'

    print(f"\n{'='*80}")
    print("AeroSync Benchmark Results")
    print(f"{'='*80}")
    print(sep)
    print('|' + '|'.join(f' {h:^{col_w}} ' for h in headers) + '|')
    print(sep)

    rows = []
    for model_label, band_results in all_results.items():
        for band_pair, metrics in band_results.items():
            pose = metrics.get('test_Pose_Success_10px', float('nan'))
            mma  = metrics.get('test_MMA_3', float('nan'))
            inl  = metrics.get('test_Num_Matches', float('nan'))
            row  = [model_label, band_pair, f'{pose:.3f}', f'{mma:.3f}', f'{inl:.1f}']
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

    # Fixed test-time settings
    config['poftr']['test']['enable_plotting'] = False
    config['poftr']['run']['num_workers']      = cfg_eval.get('num_workers', 4)
    config['poftr']['run']['prefetch_factor']  = cfg_eval.get('prefetch_factor', 2)
    config['poftr']['train']['batch_size']     = cfg_eval.get('batch_size', 8)

    band_pairs = cfg_eval.get('band_pairs', ['9um_pan', '11um_pan', '9um_11um'])
    models     = cfg_eval.get('models', [
        {'name': 'PoFTR',       'base_model': 'xoftr',       'use_phys': True},
        {'name': 'XoFTR',       'base_model': 'xoftr',       'use_phys': False},
        {'name': 'LoFTR',       'base_model': 'loftr',       'use_phys': False},
        {'name': 'ASpanFormer', 'base_model': 'aspanformer', 'use_phys': False},
    ])

    # --- Run evaluation for all models x band pairs ---
    all_results = {}  # {model_label: {band_pair: metrics}}
    for model_cfg in models:
        model_label = model_cfg['name']
        base_model  = model_cfg['base_model']
        use_phys    = model_cfg['use_phys']

        config['poftr']['proj']['base_model'] = base_model
        config['poftr']['phys']['use_phys']   = use_phys
        config['method'] = lower_config(get_method_config(base_model))

        all_results[model_label] = {}
        for band_pair in band_pairs:
            all_results[model_label][band_pair] = run_eval(
                config, cfg_eval, band_pair, model_label, device
            )

    # --- Print & Save ---
    headers   = ['Model', 'Dataset', 'Pose Success ↑', 'MMA@3 ↑', '# Inliers ↑']
    rows      = print_results_table(all_results)
    save_path = Path(cfg_eval.get('results_dir', 'evaluation_results')) / 'aerosync_results.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(rows, headers, save_path)