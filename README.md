<div align="center">

# MorVess

### Morphology-Aware Pulmonary Vessel Segmentation Network

[![Pattern Recognition](https://img.shields.io/badge/Pattern%20Recognition-Published-1f6feb.svg)](https://www.sciencedirect.com/science/article/abs/pii/S0031320326015141)
[![arXiv](https://img.shields.io/badge/arXiv-2606.24214-b31b1b.svg)](https://arxiv.org/abs/2606.24214)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official PyTorch implementation of the paper published in _Pattern Recognition_**

**Fuyou Mao · Yifei Chen · Beining Wu · Lixin Lin · Jinnan Dai · Zhiling Li · Yilei Chen · Yaqi Wang · Hao Zhang · Yan Tang · Huiyu Zhou · Feiwei Qin**

[Official Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320326015141) ·
[arXiv](https://arxiv.org/abs/2606.24214) ·
[PDF](https://arxiv.org/pdf/2606.24214) ·
[Code](https://github.com/MaoFuyou/MorVess) ·
[Citation](#citation)

</div>

> [!IMPORTANT]
> 🎉 **MorVess has been formally published online in _Pattern Recognition_ (2026).**  
> Please cite the journal article rather than the arXiv preprint.

---

## News

- **August 2026:** The official _Pattern Recognition_ article page is online.
- **June 2026:** The MorVess preprint and source code were released.

---

## Overview

Pulmonary vessel segmentation is challenging because the vascular tree is sparse, tortuous, highly multi-scale, and topologically complex. Conventional voxel-wise objectives frequently miss distal branches, break vascular connectivity, and produce geometrically inconsistent vessel trees.

**MorVess** reformulates pulmonary vessel segmentation as a joint semantic and geometric reconstruction problem. It adapts a frozen Segment Anything Model (SAM) encoder to volumetric chest CT and jointly predicts:

- a binary pulmonary vessel mask;
- a **Vessel Distance Map (VDM)** for boundary-aware geometric supervision;
- a **Vessel Thickness Map (VTM)** for local-caliber consistency.

A lightweight **2.5D Adapter** introduces inter-slice context into the SAM image encoder. A **Global–Local Fusion Block (GLFB)** combines multi-level semantic features with geometric cues to recover thin branches and preserve global vascular connectivity.

<p align="center">
  <img src="Fig1.png" alt="Overview of the MorVess framework" width="100%"/>
</p>

---

## Highlights

- **Explicit geometric priors.** VDM and VTM supervise vessel boundaries, centerline continuity, and smooth diameter transitions.
- **Parameter-efficient foundation-model adaptation.** A lightweight 2.5D adapter connects volumetric CT context with frozen 2D SAM representations.
- **Geometry-guided feature fusion.** GLFB integrates shallow, deep, decoder, distance, thickness, and gradient features.
- **Progressive optimization.** Training moves from macro-structural adaptation to micro-topological refinement.
- **Strong structural performance.** MorVess improves small-vessel recovery and global connectivity on Parse2022 and AIIB2023.
- **Low trainable-parameter footprint.** The reported configuration uses approximately **1.0M trainable parameters**.

---

## Method

### 1. Lightweight 2.5D Adapter

The adapter is inserted into the frozen SAM ViT encoder and processes a five-slice input stack. It injects cross-slice context without fully fine-tuning the foundation-model backbone.

### 2. Multi-head Geometric Decoder

The decoder jointly predicts the vessel mask, VDM, and VTM under a multi-task learning objective, allowing semantic and geometric representations to be optimized together.

### 3. Global–Local Fusion Block

GLFB aggregates shallow encoder features, deep encoder features, decoder features, VDM, VTM, and VDM gradients to refine distal branches while preserving the global vessel tree.

### Vessel Distance Map

<p align="center">
  <img src="Fig2.png" alt="Vessel Distance Map generation" width="100%"/>
</p>

VDM converts a discrete vessel boundary into a continuous boundary-aware potential field:

$$
\mathrm{VDM}(x)=\exp\left(-\lambda\min_{y\in\partial\Omega}\left\|(x-y)\odot S_p\right\|_2\right).
$$

### Vessel Thickness Map

<p align="center">
  <img src="Fig3.png" alt="Vessel Thickness Map generation" width="100%"/>
</p>

VTM propagates centerline diameter estimates to the complete vessel region:

$$
\mathrm{VTM}(x)=2D_{\mathrm{internal}}\left(\arg\min_{s\in S}\left\|(x-s)\odot S_p\right\|_2\right).
$$

### Training Objective

$$
\mathcal{L}_{\mathrm{total}}=
\lambda_1\mathcal{L}_{\mathrm{CE}}+
\lambda_2\mathcal{L}_{\mathrm{Dice}}+
\lambda_3\mathcal{L}_{\mathrm{clDice}}+
\lambda_4\mathcal{L}_{\mathrm{dist}}+
\lambda_5\mathcal{L}_{\mathrm{thick}}.
$$

| Loss | Role |
|---|---|
| $\mathcal{L}_{\mathrm{CE}}$ | Voxel-wise vessel classification |
| $\mathcal{L}_{\mathrm{Dice}}$ | Region-overlap optimization under class imbalance |
| $\mathcal{L}_{\mathrm{clDice}}$ | Centerline and topology preservation |
| $\mathcal{L}_{\mathrm{dist}}$ | Boundary-aware VDM regression |
| $\mathcal{L}_{\mathrm{thick}}$ | Scale-normalized VTM regression |

---

## Results

### Quantitative Performance

| Dataset | Dice ↑ | clDice ↑ | HD95 (mm) ↓ | AMR ↓ | DBR ↑ | DLR ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Parse2022 | **86.84 ± 4.18** | **83.22 ± 3.17** | **4.53 ± 3.06** | **0.12 ± 0.09** | **0.80 ± 0.08** | **0.83 ± 0.08** |
| AIIB2023 | **94.31 ± 3.52** | **89.34 ± 3.46** | **3.24 ± 4.81** | **0.07 ± 0.04** | **0.86 ± 0.09** | **0.89 ± 0.16** |

### Cross-domain Generalization

| Train Domain | Test Domain | Dice ↑ | clDice ↑ | HD95 ↓ |
|---|---|:---:|:---:|:---:|
| Parse2022 | HiPas | **81.14 ± 3.58** | **78.42 ± 4.20** | **7.18 ± 2.12** |
| AIIB2023 | ATM2022 | **89.25 ± 2.45** | **86.75 ± 3.10** | **4.22 ± 1.30** |

### Computational Efficiency

| Method | Trainable Params | Total Params | GMACs / 5-slice stack | Peak VRAM |
|---|:---:|:---:|:---:|:---:|
| nnU-Net | 32 M | 32 M | 180 | 18 GB |
| Diff-UNet | 64 M | 64 M | 340 | 32 GB |
| **MorVess** | **1.0 M** | **93.6 M** | **42** | **4.2 GB** |

### Qualitative Results

<p align="center">
  <img src="Fig5.png" alt="Three-dimensional pulmonary vessel segmentation results" width="100%"/>
</p>

MorVess better preserves thin terminal branches, reduces vessel discontinuities, and avoids geometrically implausible connections on both normal and pathological pulmonary CT data.

---

## Repository Structure

```text
MorVess/
├── README.md
├── CITATION.cff
├── CITATION.bib
├── LICENSE
├── train_hq_parse_stage1.py
├── train_hq_parse_stage2.py
├── test_parse_stage1.py
├── test_parse_stage2.py
├── generate_distance_map.py
├── generate_distance_process.py
├── generate_batch_distance_map.py
├── generate_thickness.py
├── generate_thickness_process.py
├── datasets/
├── preprocessing/
└── segment_anything/
    ├── build_sam.py
    └── modeling/
```

---

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

### Setup

```bash
git clone https://github.com/MaoFuyou/MorVess.git
cd MorVess

# PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Remaining dependencies
pip install SimpleITK nibabel scipy numpy pandas einops icecream \
    opencv-python Pillow tqdm h5py
```

### SAM Pretrained Weights

Download the SAM ViT-B checkpoint [`sam_vit_b_01ec64.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at:

```text
pretrained_weights/sam_vit_b_01ec64.pth
```

---

## Datasets

| Dataset | Task | Description |
|---|---|---|
| [Parse2022](https://parse2022.grand-challenge.org/) | Pulmonary artery segmentation | 100 high-resolution 3D chest CT volumes |
| [AIIB2023](https://zenodo.org/records/10041596) | Pulmonary vessel segmentation | Fibrotic CT data for robustness evaluation |

Please follow the license and data-use requirements of each original dataset.

### Preprocessing Pipeline

```text
Raw 3D CT and vessel mask
        │
        ├── HU clipping and intensity normalization
        ├── Vessel Distance Map generation
        ├── Vessel Thickness Map generation
        ├── 2.5D five-slice sample construction
        └── CSV index generation
```

```bash
# Generate Vessel Distance Maps
python generate_distance_map.py \
    -i /path/to/parse2022/train \
    -o /path/to/output \
    --batch -l 0.05

# Generate Vessel Thickness Maps
python generate_thickness.py \
    -i /path/to/parse2022/train \
    -o /path/to/thickness_output \
    --batch --out_subdir thickness_map

# Generate 2.5D samples and CSV indices
python preprocessing/util_script_parse2022_ok.py
```

---

## Training

MorVess uses a progressive two-stage optimization strategy.

### Stage I — Macro-structural Adaptation

```bash
python train_hq_parse_stage1.py \
    --root_path /path/to/2D_all_5slice \
    --output ./res_hq-par-512-stage1 \
    --ckpt ./pretrained_weights/sam_vit_b_01ec64.pth \
    --img_size 512 \
    --batch_size 1 \
    --max_epochs 400
```

### Stage II — Micro-topological Refinement

```bash
python train_hq_parse_stage2.py \
    --root_path /path/to/2D_all_5slice \
    --output ./res_hq-par-256-stage2 \
    --ckpt ./pretrained_weights/sam_vit_b_01ec64.pth \
    --img_size 256 \
    --batch_size 8 \
    --max_epochs 400
```

| Setting | Stage I | Stage II |
|---|---|---|
| Main goal | Spatial and cross-slice adaptation | Fine topology refinement |
| Resolution | 512 × 512 | 256 × 256 |
| Learning rate | $1\times10^{-5}$ | $5\times10^{-5}$ |
| Batch size | 1 | 8 |

---

## Evaluation

```bash
python test_parse_stage1.py \
    --task parse \
    --root_path /path/to/2D_all_5slice \
    --output_dir ./test_output \
    --num_classes 1 \
    --img_size 512 \
    --is_savenii
```

The evaluation scripts report Dice, clDice, HD95, and vessel-structure metrics. Predicted NIfTI masks are written to the selected output directory when `--is_savenii` is enabled.

---

## Citation

Please cite the formally published _Pattern Recognition_ article:

```bibtex
@article{mao2026morvess,
  title   = {MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network},
  author  = {Mao, Fuyou and Chen, Yifei and Wu, Beining and Lin, Lixin and
             Dai, Jinnan and Li, Zhiling and Chen, Yilei and Wang, Yaqi and
             Zhang, Hao and Tang, Yan and Zhou, Huiyu and Qin, Feiwei},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {Elsevier PII: S0031320326015141},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0031320326015141}
}
```

The arXiv version is retained for open preprint access, but the journal article above is the preferred citation. The DOI can be added once its final metadata is publicly indexed.

---

## Acknowledgements

This work was supported by the High-Performance Computing Center of Central South University, the Fundamental Research Funds for the Provincial Universities of Zhejiang (No. GK259909299001-006), the State Key Laboratory of CAD&CG, Zhejiang University (A2510), the Anhui Province Key Laboratory of Intelligent Educational Equipment and Technology (No. IEET202401), and the Postgraduate Scientific Research Innovation Project of Central South University (No. 1053320241117).

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions about the paper or code, please open an issue in this repository.

---

**Keywords:** pulmonary vessel segmentation, medical image segmentation, chest CT, SAM, foundation model adaptation, geometric priors, topology preservation, vessel distance map, vessel thickness map, 2.5D deep learning
