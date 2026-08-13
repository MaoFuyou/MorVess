<div align="center">

# MorVess

### Morphology-Aware Pulmonary Vessel Segmentation Network

[![Pattern Recognition](https://img.shields.io/badge/Pattern%20Recognition-Published-1f6feb.svg)](https://www.sciencedirect.com/science/article/abs/pii/S0031320326015141)
[![arXiv](https://img.shields.io/badge/arXiv-2606.24214-b31b1b.svg)](https://arxiv.org/abs/2606.24214)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


**Fuyou Mao Â· Yifei Chen Â· Beining Wu Â· Lixin Lin Â· Jinnan Dai Â· Zhiling Li Â· Yilei Chen Â· Yaqi Wang Â· Hao Zhang Â· Yan Tang Â· Huiyu Zhou Â· Feiwei Qin**

[Official Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320326015141) Â·
[arXiv](https://arxiv.org/abs/2606.24214) Â·
[PDF](https://arxiv.org/pdf/2606.24214) Â·
[Code](https://github.com/MaoFuyou/MorVess) Â·
[Citation](#citation)

</div>

> [!IMPORTANT]
> ð **MorVess has been formally published online in _Pattern Recognition_ (2026).**  
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

A lightweight **2.5D Adapter** introduces inter-slice context into the SAM image encoder. A **GlobalâLocal Fusion Block (GLFB)** combines multi-level semantic features with geometric cues to recover thin branches and preserve global vascular connectivity.

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
- **Parameter-efficient foundation-model adaptation.** The article reports approximately **1.0M trainable parameters**.

---

## Method

### 1. Lightweight 2.5D Adapter

The adapter is inserted into the frozen SAM ViT encoder and processes a five-slice input stack. It injects cross-slice context without fully fine-tuning the foundation-model backbone.

### 2. Multi-head Geometric Decoder

The decoder jointly predicts the vessel mask, VDM, and VTM under a multi-task learning objective, allowing semantic and geometric representations to be optimized together.

### 3. GlobalâLocal Fusion Block

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

| Dataset | Dice â | clDice â | HD95 (mm) â | AMR â | DBR â | DLR â |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Parse2022 | **86.84 Â± 4.18** | **83.22 Â± 3.17** | **4.53 Â± 3.06** | **0.12 Â± 0.09** | **0.80 Â± 0.08** | **0.83 Â± 0.08** |
| AIIB2023 | **94.31 Â± 3.52** | **89.34 Â± 3.46** | **3.24 Â± 4.81** | **0.07 Â± 0.04** | **0.86 Â± 0.09** | **0.89 Â± 0.16** |

### Cross-domain Generalization

| Train Domain | Test Domain | Dice â | clDice â | HD95 â |
|---|---|:---:|:---:|:---:|
| Parse2022 | HiPas | **81.14 Â± 3.58** | **78.42 Â± 4.20** | **7.18 Â± 2.12** |
| AIIB2023 | ATM2022 | **89.25 Â± 2.45** | **86.75 Â± 3.10** | **4.22 Â± 1.30** |

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
âââ README.md
âââ CITATION.cff
âââ CITATION.bib
âââ LICENSE
âââ requirements.txt
âââ sam_fact_tt_image_encoder_hq.py
âââ trainer_hq_parse.py
âââ trainer_hq_parse_stage2.py
âââ utils.py
âââ train_hq_parse_stage1.py
âââ train_hq_parse_stage2.py
âââ test_parse_stage1.py
âââ test_parse_stage2.py
âââ generate_distance_map.py
âââ generate_distance_process.py
âââ generate_batch_distance_map.py
âââ generate_thickness.py
âââ generate_thickness_process.py
âââ datasets/
âââ preprocessing/
âââ segment_anything/
    âââ build_sam.py
    âââ modeling/
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
pip install -r requirements.txt
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
        â
        âââ HU clipping and intensity normalization
        âââ Vessel Distance Map generation
        âââ Vessel Thickness Map generation
        âââ 2.5D five-slice sample construction
        âââ CSV index generation
```

```bash
# Generate VDM boundary potentials and internal-distance maps.
# Output is intentionally the same root so every PA* case receives a
# potential_map/ directory beside image/ and label/.
python generate_distance_process.py \
    -i /path/to/parse2022/train \
    -o /path/to/parse2022/train \
    --batch --lambda 0.5

# Generate VTM thickness maps beside each case.
python generate_thickness.py \
    -i /path/to/parse2022/train \
    -o /path/to/parse2022/train \
    --batch --out_subdir thickness_map

# Write five-slice images, masks, VDM, internal-distance, and VTM .pkl files,
# then create training.csv and test.csv.
python preprocessing/util_sript_parse2022_distance.py \
    --data_root /path/to/parse2022/train \
    --output /path/to/2D_all_5slice \
    --build_csv
```

`generate_distance_process.py` writes `<case>/potential_map/*_boundary_potential.nii.gz`
and `<case>/potential_map/*_internal_distance.nii.gz`.
`preprocessing/util_sript_parse2022_distance.py` is the script that converts
those volumes into the five-slice `boundary_potential/2Dboundary_*.pkl` and
`internal_distance/2Dinternal_*.pkl` files used by `dataset_distance.py`.

---

## Training

MorVess uses a progressive two-stage optimization strategy.

### Stage I â Macro-structural Adaptation

```bash
python train_hq_parse_stage1.py \
    --root_path /path/to/2D_all_5slice \
    --output ./res_hq-par-512-stage1 \
    --ckpt ./pretrained_weights/sam_vit_b_01ec64.pth \
    --img_size 512 \
    --batch_size 1 \
    --max_epochs 400
```

### Stage II â Micro-topological Refinement

```bash
python train_hq_parse_stage2.py \
    --root_path /path/to/2D_all_5slice \
    --output ./res_hq-par-512-stage2 \
    --ckpt ./pretrained_weights/sam_vit_b_01ec64.pth \
    --adapt_ckpt ./res_hq-par-512-stage1/epoch_400.pth \
    --img_size 512 \
    --batch_size 4 \
    --max_epochs 200
```

| Setting | Stage I | Stage II |
|---|---|---|
| Main goal | Spatial and cross-slice adaptation | Fine topology refinement |
| Resolution | 512 Ã 512 | 512 Ã 512 |
| Learning rate | $1\times10^{-5}$ | $5\times10^{-5}$ |
| Batch size | 1 | 8 |

---

## Evaluation

```bash
python test_parse_stage1.py \
    --task parse \
    --data_path /path/to/2D_all_5slice \
    --adapt_ckpt ./res_hq-par-512-stage2/epoch_200.pth \
    --ckpt ./pretrained_weights/sam_vit_b_01ec64.pth \
    --num_classes 1 \
    --img_size 512 \
    --is_savenii
```

The evaluation scripts currently require label `.pkl` files and provide
evaluation, not predict-only clinical inference. They report the implemented
Dice metric and write NIfTI masks when `--is_savenii` is enabled; geometry from
the original CT is not preserved by this legacy evaluation format.

---

## Citation

Please cite the formally published _Pattern Recognition_ article:

```bibtex
@article{MAO2026114550,
title = {MorVess: Morphology-aware pulmonary vessel segmentation network},
journal = {Pattern Recognition},
volume = {180},
pages = {114550},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.114550},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326015141},
author = {Fuyou Mao and Yifei Chen and Beining Wu and Lixin Lin and Jinnan Dai and Zhiling Li and Yilei Chen and Yaqi Wang and Hao Zhang and Yan Tang and Huiyu Zhou and Feiwei Qin},
keywords = {Pulmonary vessel, Geometric priors, Topological integrity, Foundation model adaptation},
abstract = {Accurate pulmonary vessel segmentation remains challenging due to the sparse, tortuous, and multi-scale nature of vascular structures, where small branches are easily lost and topology integrity is difficult to preserve under voxel-wise supervision. Existing deep segmentation models primarily optimize binary masks, lacking explicit geometric constraints, thus struggling to recover continuous tubular morphology and fine vascular connectivity. In this study, we introduce MorVess, a morphology-aware segmentation framework that integrates differentiable geometric priors with large-scale foundation model adaptation to achieve fine-grained vascular parsing. MorVess jointly predicts vessel masks, distance maps, and thickness maps, providing explicit supervision for vascular boundaries, centerline consistency, and smooth diameter transitions. A lightweight 2.5D adapter bridges 3D spatial context and 2D SAM representations, while a global-local fusion block aggregates multi-level semantics and geometric cues for high-fidelity topology reconstruction. Across two challenging pulmonary CT benchmarks, MorVess delivers superior Dice, clDice, and HD95 scores, substantially improving small-vessel recovery and global connectivity. These results demonstrate that embedding geometric intelligence into pretrained vision models offers a principled and scalable pathway toward precise vessel analysis and clinically reliable structural quantification. Our source code is available at https://github.com/MaoFuyou/MorVess.}
}
```

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
