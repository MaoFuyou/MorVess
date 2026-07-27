<div align="center">

# MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network

[![Pattern Recognition](https://img.shields.io/badge/Pattern%20Recognition-Accepted-2ea44f.svg)](https://www.sciencedirect.com/journal/pattern-recognition)
[![arXiv](https://img.shields.io/badge/arXiv-2606.24214-b31b1b.svg)](https://arxiv.org/abs/2606.24214)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fuyou Mao · Yifei Chen · Beining Wu · Lixin Lin · Jinnan Dai · Zhiling Li · Yilei Chen · Yaqi Wang · Hao Zhang · Yan Tang · Huiyu Zhou · Feiwei Qin**

[Paper](https://arxiv.org/abs/2606.24214) · [PDF](https://arxiv.org/pdf/2606.24214) · [Code](https://github.com/MaoFuyou/MorVess)

</div>

> [!IMPORTANT]
> 🎉 **MorVess has been formally accepted by _Pattern Recognition_ (July 2026).**  
> The final journal page, DOI, volume, and page information are not yet available and will be added after online publication. The current manuscript is available on arXiv.

---

## Overview

Accurate pulmonary vessel segmentation remains difficult because pulmonary vasculature is sparse, tortuous, highly multi-scale, and topologically complex. Conventional voxel-wise supervision often misses small branches and produces discontinuous or geometrically inconsistent vessel trees.

**MorVess** is a morphology-aware segmentation framework that integrates differentiable geometric priors with parameter-efficient adaptation of the Segment Anything Model (SAM). In addition to the binary vessel mask, MorVess jointly predicts:

- **Vessel Distance Map (VDM):** a continuous boundary-aware potential field;
- **Vessel Thickness Map (VTM):** a continuous representation of local vessel caliber.

A lightweight **2.5D Adapter** injects inter-slice context into the frozen SAM image encoder, while a **Global–Local Fusion Block (GLFB)** combines multi-level semantic features with self-predicted geometric cues to reconstruct fine and topologically continuous pulmonary vessels.

---

## Highlights

- **Morphology-aware supervision.** VDM and VTM explicitly constrain vessel boundaries, centerline continuity, and smooth caliber transitions.
- **Parameter-efficient foundation-model adaptation.** The 2.5D Adapter bridges volumetric CT context and 2D SAM representations with only **1.0M trainable parameters**.
- **Geometry-guided feature fusion.** GLFB integrates shallow, deep, decoder, distance, thickness, and gradient features for high-fidelity vessel reconstruction.
- **Progressive two-stage optimization.** Training proceeds from macro-structural adaptation to micro-topological refinement.
- **Strong performance and low resource cost.** MorVess achieves leading results on Parse2022 and AIIB2023 while using approximately **4.2 GB** peak GPU memory under the reported setting.

---

## Method

### Overall Architecture

<p align="center">
  <img src="Fig1.png" alt="Overview of the MorVess framework" width="100%"/>
</p>

MorVess consists of three principal components:

1. **Lightweight 2.5D Adapter**  
   The adapter is inserted into the frozen SAM ViT encoder to model cross-slice spatial context from a five-slice input stack.

2. **Multi-head Geometric Decoder**  
   The decoder jointly predicts the segmentation mask, VDM, and VTM under a multi-task learning formulation.

3. **Global–Local Fusion Block (GLFB)**  
   GLFB aggregates shallow encoder features, deep encoder features, decoder features, VDM, VTM, and the VDM gradient to refine fine branches and preserve global vessel connectivity.

### Differentiable Geometric Priors

<p align="center">
  <img src="Fig2.png" alt="Vessel Distance Map generation" width="100%"/>
</p>

**Vessel Distance Map (VDM).** VDM converts the discrete vessel boundary into a continuous potential field that decays with the physical distance to the vessel wall:

$$
\mathrm{VDM}(x)=\exp\left(-\lambda\min_{y\in\partial\Omega}\left\|(x-y)\odot S_p\right\|_2\right).
$$

<p align="center">
  <img src="Fig3.png" alt="Vessel Thickness Map generation" width="100%"/>
</p>

**Vessel Thickness Map (VTM).** VTM propagates the diameter estimated from the maximum inscribed sphere along the vessel centerline to the complete vessel region:

$$
\mathrm{VTM}(x)=2D_{\mathrm{internal}}\left(\arg\min_{s\in S}\left\|(x-s)\odot S_p\right\|_2\right).
$$

### Training Objective

The overall objective combines voxel-level segmentation, topology preservation, and geometric regression:

$$
\mathcal{L}_{\mathrm{total}}=
\lambda_1\mathcal{L}_{\mathrm{CE}}+
\lambda_2\mathcal{L}_{\mathrm{Dice}}+
\lambda_3\mathcal{L}_{\mathrm{clDice}}+
\lambda_4\mathcal{L}_{\mathrm{dist}}+
\lambda_5\mathcal{L}_{\mathrm{thick}}.
$$

| Loss | Purpose |
|---|---|
| $\mathcal{L}_{\mathrm{CE}}$ | Voxel-wise foreground/background classification |
| $\mathcal{L}_{\mathrm{Dice}}$ | Region-overlap optimization under class imbalance |
| $\mathcal{L}_{\mathrm{clDice}}$ | Centerline and topological consistency |
| $\mathcal{L}_{\mathrm{dist}}$ | Boundary-aware VDM regression |
| $\mathcal{L}_{\mathrm{thick}}$ | Scale-normalized VTM regression |

---

## Results

### Quantitative Performance

| Dataset | Dice ↑ | clDice ↑ | HD95 (mm) ↓ | AMR ↓ | DBR ↑ | DLR ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Parse2022 | **86.84 ± 4.18** | **83.22 ± 3.17** | **4.53 ± 3.06** | **0.12 ± 0.09** | **0.80 ± 0.08** | **0.83 ± 0.08** |
| AIIB2023 | **94.31 ± 3.52** | **89.34 ± 3.46** | **3.24 ± 4.81** | **0.07 ± 0.04** | **0.86 ± 0.09** | **0.89 ± 0.16** |

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
├── MorVess_Development_Guide.md
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
        ├── image_encoder_hq.py
        ├── mask_decoder_hq.py
        ├── hq_refiner.py
        ├── sam_distance_hq.py
        ├── transformer.py
        └── prompt_encoder.py
```

---

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/MaoFuyou/MorVess.git
cd MorVess

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install the remaining dependencies
pip install SimpleITK nibabel scipy numpy pandas einops icecream \
    opencv-python Pillow tqdm h5py
```

### SAM Pretrained Weights

Download the SAM ViT-Base checkpoint [`sam_vit_b_01ec64.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it in:

```text
pretrained_weights/sam_vit_b_01ec64.pth
```

---

## Datasets

| Dataset | Task | Size / Characteristic |
|---|---|---|
| [Parse2022](https://parse2022.grand-challenge.org/) | Pulmonary artery segmentation | 100 high-resolution 3D chest CT volumes |
| [AIIB2023](https://zenodo.org/records/10041596) | Pulmonary vessel segmentation under pathological deformation | Fibrotic CT data for robustness and cross-domain evaluation |

Please follow the licenses and data-use requirements of the original datasets.

### Preprocessing Pipeline

```text
Raw 3D CT and vessel mask
        │
        ├── HU clipping and intensity normalization
        ├── VDM generation
        ├── VTM generation
        ├── 2.5D five-slice construction
        └── CSV index generation
```

Example commands:

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

MorVess uses a two-stage optimization strategy.

### Stage I: Macro-structural Adaptation

```bash
python train_hq_parse_stage1.py \
    --root_path /path/to/2D_all_5slice \
    --output ./res_hq-par-512-stage1 \
    --ckpt ./pretrained_weights/sam_vit_b_01ec64.pth \
    --img_size 512 \
    --batch_size 1 \
    --max_epochs 400
```

### Stage II: Micro-topological Refinement

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
| Trainable components | 2.5D Adapter, decoder, GLFB | Decoder, GLFB, geometric heads |
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

The evaluation scripts report Dice, clDice, HD95, and additional vessel-structure metrics. Predicted NIfTI masks are saved to the specified output directory when `--is_savenii` is enabled.

---

## Citation

The final bibliographic information for the _Pattern Recognition_ version is not yet available. Until the article is published online, please cite the arXiv preprint:

```bibtex
@article{mao2026morvess,
  title   = {MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network},
  author  = {Mao, Fuyou and Chen, Yifei and Wu, Beining and Lin, Lixin and
             Dai, Jinnan and Li, Zhiling and Chen, Yilei and Wang, Yaqi and
             Zhang, Hao and Tang, Yan and Zhou, Huiyu and Qin, Feiwei},
  journal = {arXiv preprint arXiv:2606.24214},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.24214}
}
```

> **Publication status:** Accepted by _Pattern Recognition_. The citation above will be replaced with the final journal citation after the DOI and publication metadata become available.

---

## Acknowledgements

This work was supported by the High-Performance Computing Center of Central South University, the Fundamental Research Funds for the Provincial Universities of Zhejiang (No. GK259909299001-006), the State Key Laboratory of CAD&CG, Zhejiang University (A2510), the Anhui Province Key Laboratory of Intelligent Educational Equipment and Technology (No. IEET202401), and the Postgraduate Scientific Research Innovation Project of Central South University (No. 1053320241117).

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions about the paper or code, please open a GitHub issue in this repository.
