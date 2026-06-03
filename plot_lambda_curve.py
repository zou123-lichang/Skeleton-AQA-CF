import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv("results/sweep_val_all.csv")

# Ex1-Ex3, Ex5 一组；Ex4单独一组（SRCC范围差异太大）
GROUP_A   = ["ex1", "ex2", "ex3", "ex5"]
GROUP_B   = ["ex4"]
LABEL_MAP = {"ex1":"Ex1","ex2":"Ex2","ex3":"Ex3","ex4":"Ex4","ex5":"Ex5"}
MARKER    = {"ex1":"o","ex2":"s","ex3":"^","ex4":"D","ex5":"v"}
COLOR     = {"ex1":"#1f77b4","ex2":"#ff7f0e","ex3":"#2ca02c",
             "ex4":"#d62728","ex5":"#9467bd"}
BEST_LAMBDA = {"ex1":0.2,"ex2":0.0,"ex3":0.1,"ex4":0.2,"ex5":0.05}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                         gridspec_kw={"width_ratios":[3,1]})

def plot_group(ax, exs, title):
    for ex in exs:
        sub  = df[df["ex"]==ex].sort_values("lambda")
        lam  = sub["lambda"].values
        srcc = sub["test_srcc_mean"].values
        std  = sub["test_srcc_std"].values

        ax.plot(lam, srcc, marker=MARKER[ex], color=COLOR[ex],
                label=LABEL_MAP[ex], linewidth=1.8, markersize=6)
        ax.fill_between(lam, srcc-std, srcc+std, alpha=0.12, color=COLOR[ex])

        # 最优点标星
        best_lam = BEST_LAMBDA[ex]
        best_idx = np.argmin(np.abs(lam - best_lam))
        ax.plot(lam[best_idx], srcc[best_idx], "*", color=COLOR[ex],
                markersize=13, zorder=5)

    ax.set_xlabel(r"$\lambda_\mathrm{rank}$", fontsize=12)
    ax.set_ylabel("Test SRCC", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-0.01, 0.22)

plot_group(axes[0], GROUP_A,
           r"Ex1–Ex3, Ex5: effect of $\lambda_\mathrm{rank}$")
plot_group(axes[1], GROUP_B,
           r"Ex4: effect of $\lambda_\mathrm{rank}$")

# 注解：星号=val-selected最优点
axes[0].annotate("★ = val-selected optimum",
                 xy=(0.02, 0.04), xycoords="axes fraction",
                 fontsize=9, color="#555")

plt.suptitle(r"RankNet loss weight $\lambda$ sensitivity across KIMORE exercises",
             fontsize=12, y=1.01)
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/lambda_srcc_curve.pdf", dpi=300, bbox_inches="tight")
plt.savefig("results/lambda_srcc_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print("[SAVE] results/lambda_srcc_curve.pdf")
print("[SAVE] results/lambda_srcc_curve.png")
