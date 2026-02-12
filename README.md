# PoFTR: Physics-Informed Transformer for Cross-Spectral Image Registration

This repository contains the official implementation of **PoFTR**. 

PoFTR is a physics-informed transformer designed for extreme cross-spectral image registration (e.g., 11μm LWIR to Visible PAN). By integrating physical priors through Spatial Feature Transform (SFT) layers, our method maintains geometric consistency even in the presence of extreme radiometric discrepancies where general foundation models fail.

## 🚀 Key Features

* **Physics-Aware Conditioning:** Integration of physical priors via SFT layers to ensure geometric consistency.
* **Cross-Spectral Robustness:** Specifically optimized for 9μm and 11μm thermal-to-visible registration challenges.
* **Modular Architecture:** Evaluation support for modified versions of LoFTR and ASpanFormer backbones.

## 🛠 Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/PoFTR-XXXX/
cd PoFTR

# Install dependencies
pip install -r requirements.txt
```

## 📊 Reproducing Results

### 1. Main Quantitative Results

To evaluate PoFTR on the 11μm–PAN test set and replicate our performance gains:

```bash
python test.py --config configs/poftr_11um.yaml --dataset_type test
```

### 2. Ablation Studies (Prior Fidelity)

We provide built-in support for the prior fidelity experiments. Use the `--ablation_version` flag to perturb the physical prior during inference to verify geometric dependency:

| Ablation Type | Description | Command |
|---------------|-------------|---------|
| **Clean (Ours)** | Standard Physics-Informed Prior | `--ablation_version standard` |
| **Noisy** | Gaussian noise added to prior | `--ablation_version noisy` |
| **Shuffled** | 16 × 16 block-wise spatial shuffle | `--ablation_version shuffled` |
| **Zeroed** | Null physical prior channels | `--ablation_version zeroed_priors` |

**Example:**
```bash
python test.py --config configs/poftr_11um.yaml --dataset_type test --ablation_version noisy
```

## 📂 Project Structure

* `src/dataset/`: Contains `MonochromeDs` with integrated perturbation logic for ablations.
* `src/models/`: Implementation of SFT layers and physics-aware conditioning modules.
* `configs/`: YAML configurations for different spectral bands and model variants.

## 📝 Citation

If you find this work useful for your research, please cite our ECCV submission:

```bibtex
@inproceedings{poftr2026,
  title={PoFTR: Physics-Informed Transformer for Cross-Spectral Image Registration},
  author={Anonymous Authors},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies:

```text
torch>=1.8.1
pytorch-lightning==1.3.5
torchmetrics==0.6.0
opencv-python==4.4.0.46
albumentations==0.5.1
kornia==0.4.1
numpy==1.24.4
h5py==3.1.0
yacs
loguru
tqdm
matplotlib
```

## 📧 Contact

For questions or issues, please open an issue in this repository or contact the anonymous authors through the conference review system.
