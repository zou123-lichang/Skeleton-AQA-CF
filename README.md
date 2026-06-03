# Skeleton-AQA-CF

**Interpretable Skeleton-Based Rehabilitation Assessment with Biomechanics-Constrained Counterfactual Feedback**

> Submitted to *Applied Sciences* (MDPI), Special Issue: Current Advances in Rehabilitation Technology.

---

## Overview

Existing skeleton-based AQA systems can score movement execution but often provide limited information about *what to correct*. This repository provides a three-stage interpretable pipeline:

1. **Score** — ST-GCN with a per-exercise tuned RankNet loss for rank-consistent quality regression
2. **Localize** — Post-hoc gradient-based saliency maps identifying error-prone joints and time windows, with faithfulness verified via occlusion tests
3. **Correct** — Biomechanics-constrained counterfactual optimization generating anatomically feasible corrective suggestions (bone-length deviation reduced by **46%** vs. unconstrained baseline)

Evaluated on all five exercises of the [KIMORE](https://drive.google.com/drive/folders/1S_y95vxwIQFYxrNNzNODqVnxaKakNcXb) rehabilitation dataset (mean SRCC = 0.830).

---

## Results Summary

| Exercise | Baseline SRCC | Ours (λ*) | Δ |
|---|---|---|---|
| Ex1 | 0.7956 | 0.7975 | +0.002 |
| Ex2 | 0.7891 | 0.7891 | +0.000 |
| Ex3 | 0.8257 | 0.8336 | +0.008 |
| Ex4 | 0.9330 | 0.9375 | +0.003 |
| Ex5 | 0.7790 | 0.7937 | +0.015 |
| **Mean** | **0.8245** | **0.8303** | **+0.006** |

Counterfactual feedback (N=75, full ex5 test set):

| Method | Δŷ↑ | Bone deviation↓ |
|---|---|---|
| CF-ScoreOnly | 4.350 | 0.224 |
| CF-L1+Smooth | 4.351 | 0.220 |
| **CF-Full (+Bone)** | **4.351** | **0.120 (−46%)** |

---

## Repository Structure

```
├── config.py                   # All hyperparameters and paths
├── dataset.py                  # KIMORE dataset loader
├── main_train.py               # Training script (single seed)
├── main_explain.py             # Counterfactual generation + evaluation
├── posthoc_importance.py       # Gradient saliency + faithfulness test
├── robustness_eval_v2.py       # Missing-joint robustness evaluation
├── models/
│   ├── st_gcn.py               # ST-GCN backbone
│   └── graph.py                # Kinect25 graph topology
├── scripts/
│   └── prepare_kimore_csv.py   # Preprocess raw KIMORE CSV → pkl
├── train_all_exercises.py      # Batch training across ex1–ex5
├── train_all_baseline.py       # Batch baseline training
├── sweep_per_exercise_val.py   # Per-exercise λ sweep (val-selected)
├── plot_lambda_curve.py        # λ sensitivity figure
└── csv_to_latex_table.py       # Export results to LaTeX tables
```

---

## Setup

```bash
conda create -n aqa python=3.10 -y
conda activate aqa
pip install -r requirements.txt
```

For CUDA-specific PyTorch builds, install PyTorch from the official selector first, then install the remaining dependencies with `pip install -r requirements.txt`.

---

## Data Preparation

Download the KIMORE dataset from the [official link](https://drive.google.com/drive/folders/1S_y95vxwIQFYxrNNzNODqVnxaKakNcXb) and place the exercise folders under `KIMORE_DATASET/`. Then run:

```bash
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex1"
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex2"
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex3"
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex4"
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex5"
```

Processed `.pkl` files will be saved to `data/`.

The raw KIMORE files and processed `.pkl` files are not included in this repository. Please follow the KIMORE dataset terms of use.

---

## Training

**Single exercise (ex5, optimal λ):**
```bash
python main_train.py
```

**All exercises with per-exercise λ sweep (validation-selected):**
```bash
python sweep_per_exercise_val.py
```

**Batch training all exercises with fixed λ=0.05:**
```bash
python train_all_exercises.py
```

---

## Evaluation

**Counterfactual feedback (full test set):**
```bash
python main_explain.py
```

**Saliency faithfulness test:**
```bash
python posthoc_importance.py
```

**Missing-joint robustness:**
```bash
python robustness_eval_v2.py
```

**λ sensitivity curve:**
```bash
python plot_lambda_curve.py
```

---

## Reproducibility Notes

The main random seeds are defined in `config.py` and `sweep_per_exercise_val.py`. Training outputs, checkpoints, generated figures, processed data files, and cached Python files are intentionally ignored by Git.

---

## Citation

If you find this repository useful, please cite our manuscript:

```bibtex
@misc{zou2026skeleton,
  title     = {Interpretable Skeleton-Based Rehabilitation Assessment with
               Biomechanics-Constrained Counterfactual Feedback},
  author    = {Zou, Lichang and Bai, Shusong},
  year      = {2026},
  note      = {Manuscript under review}
}
```

---

## License

MIT License. The KIMORE dataset is subject to its own terms of use — please refer to the original dataset page.
