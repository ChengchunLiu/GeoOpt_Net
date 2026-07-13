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

## Publication

This repository accompanies the following publication:

**A Cross-Domain Graph Learning Protocol for Single-Step Molecular Geometry Refinement**
**Chengchun Liu, Wendi Cai, Boxuan Zhao, and Fanyang Mo**
*Journal of Chemical Theory and Computation* **2026**.
https://doi.org/10.1021/acs.jctc.6c01080

**Published article:** https://pubs.acs.org/doi/10.1021/acs.jctc.6c01080
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

# Case study: GeoOpt-Net inference and Kabsch-aligned RMSD evaluation

This folder provides a case study for comparing RDKit/MMFF, xTB, and GeoOpt-Net geometries against B3LYP/TZVP reference structures.

The complete workflow is:

1. construct a molecular graph from a SMILES string;
2. generate or provide an initial 3D geometry, for example from RDKit/MMFF;
3. refine the initial geometry using `inference/infer_geometry.py`;
4. optionally evaluate the refined geometry against a reference structure using `scripts/compute_aligned_rmsd.py`.


---

## 1. Directory structure

The case study is organized as follows:

```text
case_study/
├── GeoOpt-Net/
│   └── GeoOpt-Net-refined XYZ files
├── RDKit/
│   └── RDKit/MMFF initial XYZ files
├── opt-xyz-xtb/
│   └── xTB-optimized XYZ files
├── reference-B3LYP-TZVP/
│   └── B3LYP/TZVP reference XYZ files
└── readme.md
```


---

## 2. Prepare a molecular input from SMILES

For example, acetylene can be represented by the SMILES string:

```text
C#C
```

For example, an RDKit/MMFF-generated initial geometry can be saved as:

```text
case_study/RDKit/case_001_input.xyz
```

with coordinates such as:

```xyz
4
Acetylene initial geometry from RDKit/MMFF
C  -0.5933596492  0.0305018742  0.0602263398
C   0.5933601856 -0.0305036865  0.2296369672
H  -1.6471040249  0.0846735016 -0.0901968554
H   1.6471035480 -0.0846716911  0.3800690770
```


---

## 3. Place the trained model file

Put the trained model file under the `model/` directory, for example:

```text
model/best_model.pt
```

---

## 4. Run GeoOpt-Net single-step inference

Use `inference/infer_geometry.py` to refine the RDKit/MMFF initial geometry:

```bash
python inference/infer_geometry.py \
  --smiles "C#C" \
  --input_xyz case_study/RDKit/case_001_input.xyz \
  --checkpoint model/best_model.pt \
  --output_xyz case_study/GeoOpt-Net/case_001_output.xyz \
  --project_root .
```

The refined geometry will be written to:

```text
case_study/GeoOpt-Net/case_001_output.xyz
```

For example, the GeoOpt-Net-refined output may look like:

```xyz
4
Acetylene refined geometry by GeoOpt-Net
C  -0.5941444635  0.0083292369  0.0734921694
C   0.5944776535 -0.0070936605  0.2159397304
H  -1.6498693228  0.0199433640 -0.0533744097
H   1.6495361328 -0.0211789384  0.3436780274
```



---

## 5. Reference geometry for evaluation

The corresponding B3LYP/TZVP reference structure is stored under:

```text
case_study/reference-B3LYP-TZVP/
```

For example:

```text
case_study/reference-B3LYP-TZVP/case_001_reference.xyz
```

with coordinates such as:

```xyz
4
Acetylene reference geometry at B3LYP/TZVP
C   0.0000000000 -0.0000000000 -0.5989660025
C  -0.0000000000 -0.0000000000  0.5989660025
H   0.0000000000 -0.0000000000 -1.6615680456
H  -0.0000000000 -0.0000000000  1.6615680456
```


---

## 6. Compute Kabsch-aligned RMSD for one molecule

RMSD evaluation is performed using:

```text
scripts/compute_aligned_rmsd.py
```

For a single GeoOpt-Net output and its corresponding B3LYP/TZVP reference structure:

```bash
python scripts/compute_aligned_rmsd.py \
  --output-xyz case_study/GeoOpt-Net/case_001_output.xyz \
  --reference-xyz case_study/reference-B3LYP-TZVP/case_001_reference.xyz \
  --out-csv case_study/case_001_geooptnet_rmsd.csv
```

The script performs atom-order-preserving Kabsch alignment and reports the aligned RMSD in Å.

The output CSV contains columns such as:

```text
molecule_id,output_xyz,reference_xyz,n_atoms,aligned_rmsd_A,status,error_message
```

---

## 7. Batch RMSD evaluation for GeoOpt-Net

To evaluate all GeoOpt-Net-refined structures against the same B3LYP/TZVP references:

```bash
python scripts/compute_aligned_rmsd.py \
  --output-dir case_study/GeoOpt-Net \
  --reference-dir case_study/reference-B3LYP-TZVP \
  --out-csv case_study/geooptnet_aligned_rmsd.csv
```

The script matches files by molecule identifier. For example, the following pair will be matched automatically:

```text
case_study/GeoOpt-Net/case_001_output.xyz
case_study/reference-B3LYP-TZVP/case_001_reference.xyz
```

---

## 8. Batch RMSD evaluation for RDKit/MMFF and xTB

The same RMSD script can also be used to evaluate RDKit/MMFF and xTB geometries against the same B3LYP/TZVP reference structures.

For RDKit/MMFF:

```bash
python scripts/compute_aligned_rmsd.py \
  --output-dir case_study/RDKit \
  --reference-dir case_study/reference-B3LYP-TZVP \
  --out-csv case_study/rdkit_aligned_rmsd.csv
```

For xTB:

```bash
python scripts/compute_aligned_rmsd.py \
  --output-dir case_study/opt-xyz-xtb \
  --reference-dir case_study/reference-B3LYP-TZVP \
  --out-csv case_study/xtb_aligned_rmsd.csv
```

This ensures that RDKit/MMFF, xTB, and GeoOpt-Net are evaluated against the same B3LYP/TZVP references using the same atom-order-preserving Kabsch-aligned RMSD protocol.

---

## 9. Optional: write aligned XYZ files

To save the Kabsch-aligned GeoOpt-Net output structures:

```bash
python scripts/compute_aligned_rmsd.py \
  --output-dir case_study/GeoOpt-Net \
  --reference-dir case_study/reference-B3LYP-TZVP \
  --out-csv case_study/geooptnet_aligned_rmsd.csv \
  --write-aligned-dir case_study/GeoOpt-Net-aligned
```

The aligned XYZ files will be written to:

```text
case_study/GeoOpt-Net-aligned/
```

---

## Notes

- GeoOpt-Net is used as a single-step local geometry-refinement model.
- The input XYZ geometry should be chemically reasonable, for example an RDKit/MMFF-optimized structure.
- The B3LYP/TZVP reference XYZ files are used only for evaluation.
- The RMSD script does not reorder atoms. Atom counts and element ordering must match between the output XYZ and the reference XYZ.
