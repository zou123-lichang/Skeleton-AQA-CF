import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import numpy as np
from config import Config
from main_train import train_one_seed

EXERCISES = ["ex1", "ex2", "ex3", "ex4", "ex5"]

def run_baseline(ex_name):
    print(f"\n{'='*50}")
    print(f"[BASELINE] Training {ex_name}")
    print(f"{'='*50}")

    Config.TRAIN_DATA_PATH  = f"data/kimore_{ex_name}_train.pkl"
    Config.TRAIN_LABEL_PATH = f"data/kimore_{ex_name}_train_y.pkl"
    Config.TEST_DATA_PATH   = f"data/kimore_{ex_name}_test.pkl"
    Config.TEST_LABEL_PATH  = f"data/kimore_{ex_name}_test_y.pkl"
    Config.DATA_PATH        = Config.TRAIN_DATA_PATH
    Config.LABEL_PATH       = Config.TRAIN_LABEL_PATH
    Config.exp_name         = ex_name
    Config.variant          = "baseline"
    Config.SAVE_DIR         = f"checkpoints/{ex_name}/baseline"

    Config.use_augment            = False
    Config.lambda_rank            = 0.0
    Config.use_temporal_attention = False

    # ex1用5个seed，其余3个seed
    seeds = [2024, 2025, 2026, 2027, 2028] if ex_name == "ex1" else [2024, 2025, 2026]
    Config.seeds = seeds

    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    summaries = [train_one_seed(int(s)) for s in seeds]

    srccs = np.array([s["srcc"]  for s in summaries])
    plccs = np.array([s["plcc"]  for s in summaries])
    maes  = np.array([s["mae"]   for s in summaries])
    rmses = np.array([s["rmse"]  for s in summaries])

    result = {
        "ex":        ex_name,
        "srcc_mean": float(srccs.mean()),
        "srcc_std":  float(srccs.std(ddof=1)) if len(srccs) > 1 else 0.0,
        "plcc_mean": float(plccs.mean()),
        "plcc_std":  float(plccs.std(ddof=1)) if len(plccs) > 1 else 0.0,
        "mae_mean":  float(maes.mean()),
        "mae_std":   float(maes.std(ddof=1))  if len(maes)  > 1 else 0.0,
        "rmse_mean": float(rmses.mean()),
        "rmse_std":  float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
    }

    print(f"[{ex_name} BASELINE] SRCC={result['srcc_mean']:.4f}±{result['srcc_std']:.4f} "
          f"PLCC={result['plcc_mean']:.4f}±{result['plcc_std']:.4f} "
          f"MAE={result['mae_mean']:.4f}±{result['mae_std']:.4f}")
    return result


def main():
    os.makedirs("results", exist_ok=True)
    all_results = []

    for ex in EXERCISES:
        r = run_baseline(ex)
        all_results.append(r)

    out_path = "results/all_exercises_baseline.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ex", "srcc_mean", "srcc_std", "plcc_mean", "plcc_std",
            "mae_mean", "mae_std", "rmse_mean", "rmse_std"
        ])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[DONE] Results saved to {out_path}")
    print("\n===== BASELINE SUMMARY =====")
    print(f"{'Ex':<6} {'SRCC':>12} {'PLCC':>12} {'MAE':>12} {'RMSE':>12}")
    for r in all_results:
        print(f"{r['ex']:<6} "
              f"{r['srcc_mean']:.4f}±{r['srcc_std']:.4f}  "
              f"{r['plcc_mean']:.4f}±{r['plcc_std']:.4f}  "
              f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}  "
              f"{r['rmse_mean']:.4f}±{r['rmse_std']:.4f}")


if __name__ == "__main__":
    main()
