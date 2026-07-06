# Skeleton-AQA-CF

Code for experiments on skeleton-based action quality assessment and
counterfactual feedback for rehabilitation exercises.

The repository accompanies the manuscript:

**Interpretable Skeleton-Based Rehabilitation Assessment with
Biomechanics-Constrained Counterfactual Feedback**

No journal-specific information is included here. The repository is intended to
provide the preprocessing, training, evaluation, and plotting code needed to
reproduce the reported experiments.

## Overview

The code implements a three-part workflow:

1. Train an ST-GCN based scorer for skeleton action quality assessment.
2. Compute gradient-based temporal and joint importance scores for trained
   models.
3. Generate counterfactual skeleton sequences with sparsity, smoothness, and
   bone-length constraints.

Experiments use the KIMORE rehabilitation dataset. The dataset is not distributed
in this repository; users should obtain it from the original dataset source and
follow its terms of use.

## Main results

The table below reports the values used in the manuscript. SRCC denotes
Spearman rank correlation.

| Exercise | MSE baseline SRCC | RankNet model SRCC | Difference |
|---|---:|---:|---:|
| Ex1 | 0.7956 | 0.7975 | +0.002 |
| Ex2 | 0.7891 | 0.7891 | +0.000 |
| Ex3 | 0.8257 | 0.8336 | +0.008 |
| Ex4 | 0.9330 | 0.9375 | +0.003 |
| Ex5 | 0.7790 | 0.7937 | +0.015 |
| Mean | 0.8245 | 0.8303 | +0.006 |

Counterfactual feedback was evaluated on the full Ex5 test split
(`N = 75`):

| Method | Mean predicted score change | Mean bone-length deviation |
|---|---:|---:|
| CF-ScoreOnly | 4.350 | 0.224 |
| CF-L1+Smooth | 4.351 | 0.220 |
| CF-Full (+Bone) | 4.351 | 0.120 |

## Repository structure

```text
config.py                    Global paths and experiment settings
dataset.py                   KIMORE dataset loader
main_train.py                Training script for one configuration
main_explain.py              Counterfactual generation and evaluation
posthoc_importance.py        Gradient saliency and occlusion test
robustness_eval_v2.py        Missing-joint robustness evaluation
models/
  st_gcn.py                  ST-GCN backbone
  graph.py                   Kinect25 graph topology
scripts/
  prepare_kimore_csv.py      Convert raw KIMORE CSV files to pkl files
train_all_exercises.py       Train the fixed-lambda model across exercises
train_all_baseline.py        Train the MSE baseline across exercises
sweep_per_exercise_val.py    Per-exercise lambda_rank validation sweep
plot_lambda_curve.py         Plot the lambda_rank sensitivity curve
plot_robust_compare.py       Plot robustness comparisons
csv_to_latex_table.py        Convert result CSV files to LaTeX tables
```

## Environment

The code was prepared with Python 3.10. A minimal setup is:

```bash
conda create -n aqa python=3.10 -y
conda activate aqa
pip install -r requirements.txt
```

If CUDA is used, install the PyTorch build matching the local CUDA version
before installing the remaining packages.

## Data preparation

Download KIMORE from the original dataset source and place the exercise folders
under:

```text
KIMORE_DATASET/
```

Then run:

```bash
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex1"
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex2"
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex3"
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex4"
python scripts/prepare_kimore_csv.py --root KIMORE_DATASET --ex "Kimore ex5"
```

The script writes processed files to `data/`. The train/test split is generated
deterministically from the seed in `scripts/prepare_kimore_csv.py` unless a
different seed is passed through the command line.

Raw data files, processed `.pkl` files, checkpoints, and generated results are
not included in this repository.

## Training

Train the default Ex5 configuration:

```bash
python main_train.py
```

Train the fixed-lambda model for all five exercises:

```bash
python train_all_exercises.py
```

Train the MSE baseline for all five exercises:

```bash
python train_all_baseline.py
```

Run the per-exercise validation sweep for `lambda_rank`:

```bash
python sweep_per_exercise_val.py
```

## Evaluation and figures

Counterfactual feedback:

```bash
python main_explain.py
```

Gradient saliency and occlusion-based faithfulness test:

```bash
python posthoc_importance.py
```

Missing-joint robustness:

```bash
python robustness_eval_v2.py
```

Lambda sensitivity figure:

```bash
python plot_lambda_curve.py
```

Generated outputs are written to `results/` or related result folders.

## Reproducibility notes

- Main random seeds are specified in `config.py`,
  `train_all_baseline.py`, and `sweep_per_exercise_val.py`.
- Ex1 uses five seeds in the reported experiments; the other exercises use
  three seeds.
- The preprocessing script creates deterministic train/test splits when the
  same seed is used.
- Checkpoints and generated result files are intentionally excluded from the
  repository.

## Citation

If you use this code, please cite the associated manuscript. The BibTeX entry
will be updated after publication.

```bibtex
@misc{zou2026skeleton,
  title  = {Interpretable Skeleton-Based Rehabilitation Assessment with
            Biomechanics-Constrained Counterfactual Feedback},
  author = {Zou, Lichang and Bai, Shusong},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

This code is released under the MIT License. The KIMORE dataset is governed by
its own terms of use and is not redistributed here.
