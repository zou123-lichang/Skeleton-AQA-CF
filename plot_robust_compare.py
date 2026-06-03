import os
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    # level,srcc,plcc,mae,rmse
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:,0], data[:,1]  # level, srcc

def plot_one(name, base_csv, full_csv, out_png, xlabel):
    x1,y1 = load_csv(base_csv)
    x2,y2 = load_csv(full_csv)
    plt.figure(figsize=(6,3))
    plt.plot(x1, y1, marker="o", label="Baseline")
    plt.plot(x2, y2, marker="o", label="Full")
    plt.xlabel(xlabel); plt.ylabel("SRCC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

os.makedirs("results_compare", exist_ok=True)

plot_one(
    "noise",
    "results_baseline/robust_noise.csv",
    "results_full/robust_noise.csv",
    "results_compare/robust_noise_compare.png",
    "Gaussian noise σ"
)
plot_one(
    "jointdrop",
    "results_baseline/robust_jointdrop.csv",
    "results_full/robust_jointdrop.csv",
    "results_compare/robust_jointdrop_compare.png",
    "Joint missing ratio p"
)
plot_one(
    "timestretch",
    "results_baseline/robust_timestretch.csv",
    "results_full/robust_timestretch.csv",
    "results_compare/robust_timestretch_compare.png",
    "Time stretch factor"
)
print("Saved to results_compare/")