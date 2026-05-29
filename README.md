# GeoOpt-Net

[![arXiv](https://img.shields.io/badge/arXiv-2601.22723-b31b1b.svg)](https://arxiv.org/abs/2601.22723)

**GeoOpt-Net** is a graph learning framework for **single-step molecular geometry refinement**.  
Starting from inexpensive initial conformers, it predicts **DFT-quality molecular geometries** in a single forward pass, aiming to accelerate quantum-chemical workflows while preserving structural and electronic fidelity.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0766fba8-6a4b-4d7f-a672-b967372c0989" alt="GeoOpt-Net overview" width="100%">
</p>

---

## Overview

Accurate molecular geometries are essential for reliable quantum-chemical calculations, yet conventional DFT geometry optimization often becomes a major computational bottleneck in large-scale molecular studies. GeoOpt-Net addresses this challenge by directly refining low-cost initial conformers into high-quality geometries in a single forward pass, without iterative optimization during inference.

GeoOpt-Net is built on a multi-branch **SE(3)-equivariant graph neural architecture** and trained with a **two-stage multi-fidelity strategy**. By combining broad geometric pretraining with high-level fine-tuning, the model efficiently bridges the gap between inexpensive conformers and DFT-quality structures.

---

## Highlights

- **Single-step refinement** from low-cost conformers to high-quality molecular geometries
- **SE(3)-equivariant graph learning** for robust 3D structural modeling
- **Two-stage multi-fidelity training** for improved accuracy and transferability
- **Fidelity-aware feature modulation (FAFM)** for theory-level adaptation
- Improved compatibility with downstream **DFT optimization** and **property calculations**

---

## Preprint

This repository accompanies our preprint:

**A Cross-Domain Graph Learning Protocol for Single-Step Molecular Geometry Refinement**  
**Chengchun Liu, Wendi Cai, Boxuan Zhao, Fanyang Mo**  
**arXiv:2601.22723 (2026)**

**Preprint:** https://arxiv.org/abs/2601.22723

---

## Installation

> The codebase is currently being organized and cleaned for public release.

Clone the repository:

```bash
git clone https://github.com/ChengchunLiu/GeoOpt_Net.git
cd GeoOpt_Net
```
---

## Citation

If you find this project useful in your research, please cite:

```bash
@article{liu2026geooptnet,
  title={A Cross-Domain Graph Learning Protocol for Single-Step Molecular Geometry Refinement},
  author={Liu, Chengchun and Cai, Wendi and Zhao, Boxuan and Mo, Fanyang},
  journal={arXiv preprint arXiv:2601.22723},
  year={2026}
}
```


---

## Inference: Refining a User-Provided Geometry

GeoOpt-Net provides an inference script for refining an initial molecular geometry using a trained checkpoint.
The inference workflow takes a molecular identity, an initial 3D structure, and a trained model checkpoint as input, and outputs a refined molecular geometry in XYZ format.

### 1. Prepare an initial XYZ geometry

For example, an initial carbon monoxide geometry can be saved as `co_input.xyz`:

```xyz
2
CO initial geometry
C  0.000000  0.000000  0.000000
O  1.150000  0.000000  0.000000
```

### 2. Place the trained checkpoint

Put the trained checkpoint under the `model/` directory, for example:

```bash
model/best_model.pt
```

### 3. Run inference

Use `infer_geometry.py` to refine the initial geometry:

```bash
python infer_geometry.py \
  --smiles "[C-]#[O+]" \
  --input_xyz co_input.xyz \
  --checkpoint model/best_model.pt \
  --output_xyz co_refined.xyz \
  --project_root .
```

The refined geometry will be written to:

```bash
co_refined.xyz
```

### 4. Input and output

The inference script uses only:

* the molecular graph constructed from the SMILES string;
* the user-provided initial Cartesian coordinates;
* the frozen trained checkpoint.

The output is a refined XYZ geometry predicted by GeoOpt-Net in a single forward pass.


### Example workflow

```bash
# Clone the repository
git clone https://github.com/ChengchunLiu/GeoOpt_Net.git
cd GeoOpt_Net

# Create a model directory and place the checkpoint inside it
mkdir -p model

# Run GeoOpt-Net inference
python infer_geometry.py \
  --smiles "[C-]#[O+]" \
  --input_xyz co_input.xyz \
  --checkpoint model/best_model.pt \
  --output_xyz co_refined.xyz \
  --project_root .
```

After running the command, the refined molecular geometry can be found in `co_refined.xyz`.

