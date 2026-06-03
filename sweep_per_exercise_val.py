import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import numpy as np
from config import Config
from main_train import train_one_seed

EXERCISES = ["ex1", "ex2", "ex3", "ex4", "ex5"]
LAMBDAS   = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

SEEDS = {
    "ex1": [2024, 2025, 2026, 2027, 2028],
    "ex2": [2024, 2025, 2026],
    "ex3": [2024, 2025, 2026],
    "ex4": [2024, 2025, 2026],
    "ex5": [2024, 2025, 2026],
}

def run_one(ex_name, lam):
    Config.TRAIN_DATA_PATH  = f"data/kimore_{ex_name}_train.pkl"
    Config.TRAIN_LABEL_PATH = f"data/kimore_{ex_name}_train_y.pkl"
    Config.TEST_DATA_PATH   = f"data/kimore_{ex_name}_test.pkl"
    Config.TEST_LABEL_PATH  = f"data/kimore_{ex_name}_test_y.pkl"
    Config.DATA_PATH        = Config.TRAIN_DATA_PATH
    Config.LABEL_PATH       = Config.TRAIN_LABEL_PATH
    Config.exp_name         = ex_name
    Config.variant          = f"val_lam{lam}"
    Config.SAVE_DIR         = f"checkpoints/{ex_name}/val_lam{lam}"
    Config.use_augment            = False
    Config.lambda_rank            = float(lam)
    Config.use_temporal_attention = False

    seeds = SEEDS[ex_name]
    Config.seeds = seeds
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    summaries = [train_one_seed(int(s)) for s in seeds]

    # 用val SRCC均值选λ
    val_srccs  = np.array([s["best_val_srcc"] for s in summaries])
    test_srccs = np.array([s["srcc"]  for s in summaries])
    test_plccs = np.array([s["plcc"]  for s in summaries])
    test_maes  = np.array([s["mae"]   for s in summaries])
    test_rmses = np.array([s["rmse"]  for s in summaries])

    return {
        "ex": ex_name, "lambda": lam,
        "val_srcc_mean":  float(val_srccs.mean()),
        "test_srcc_mean": float(test_srccs.mean()),
        "test_srcc_std":  float(test_srccs.std(ddof=1)) if len(test_srccs)>1 else 0.0,
        "test_plcc_mean": float(test_plccs.mean()),
        "test_plcc_std":  float(test_plccs.std(ddof=1)) if len(test_plccs)>1 else 0.0,
        "test_mae_mean":  float(test_maes.mean()),
        "test_mae_std":   float(test_maes.std(ddof=1))  if len(test_maes)>1  else 0.0,
        "test_rmse_mean": float(test_rmses.mean()),
        "test_rmse_std":  float(test_rmses.std(ddof=1)) if len(test_rmses)>1 else 0.0,
    }

def main():
    os.makedirs("results", exist_ok=True)
    all_rows = []

    for ex in EXERCISES:
        print(f"\n{'='*55}")
        print(f"[SWEEP-VAL] {ex}")
        print(f"{'='*55}")

        for lam in LAMBDAS:
            print(f"  lambda={lam} ...")
            r = run_one(ex, lam)
            all_rows.append(r)
            print(f"  -> val_SRCC={r['val_srcc_mean']:.4f}  "
                  f"test_SRCC={r['test_srcc_mean']:.4f}±{r['test_srcc_std']:.4f}  "
                  f"test_PLCC={r['test_plcc_mean']:.4f}")

    # 保存全部结果
    out_all = "results/sweep_val_all.csv"
    fields = ["ex","lambda","val_srcc_mean","test_srcc_mean","test_srcc_std",
              "test_plcc_mean","test_plcc_std","test_mae_mean","test_mae_std",
              "test_rmse_mean","test_rmse_std"]
    with open(out_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\n[SAVE] {out_all}")

    # 按val SRCC选最优λ，报告对应test数字
    print("\n===== BEST-λ (selected by VAL SRCC) =====")
    print(f"{'Ex':<6} {'Best λ':>8} {'Val SRCC':>10} {'Test SRCC':>14} {'Test PLCC':>14} {'Test MAE':>12}")
    best_rows = []
    for ex in EXERCISES:
        rows_ex = [r for r in all_rows if r["ex"]==ex]
        best = max(rows_ex, key=lambda r: r["val_srcc_mean"])
        best_rows.append(best)
        print(f"{ex:<6} {best['lambda']:>8}  "
              f"{best['val_srcc_mean']:>8.4f}  "
              f"{best['test_srcc_mean']:.4f}±{best['test_srcc_std']:.4f}  "
              f"{best['test_plcc_mean']:.4f}±{best['test_plcc_std']:.4f}  "
              f"{best['test_mae_mean']:.4f}±{best['test_mae_std']:.4f}")

    out_best = "results/sweep_val_best.csv"
    with open(out_best, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(best_rows)
    print(f"[SAVE] {out_best}")

if __name__ == "__main__":
    main()
