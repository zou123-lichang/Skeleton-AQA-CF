import os, csv, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from dataset import AQADataset
from models.st_gcn import Model
from main_train import spearmanr_np

def load_model(ckpt_path):
    model = Model(
        in_channels=Config.coords_dim,
        num_class=1,
        graph_args={'layout': 'kinect25', 'strategy': 'spatial'},
        use_temporal_attention=False
    ).to(Config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    model.eval()
    return model

def apply_joint_drop(x, p, seed=0):
    rng = np.random.default_rng(seed)
    N, C, T, V = x.shape
    drop_idx = np.where(rng.random(V) < p)[0]
    if len(drop_idx) > 0:
        x = x.clone()
        x[:, :, :, drop_idx] = 0.0
    return x

@torch.no_grad()
def eval_srcc(model, loader, perturb_fn):
    preds, gts = [], []
    for x, t in loader:
        x = perturb_fn(x.to(Config.DEVICE))
        preds.append(model(x).view(-1).detach().cpu().numpy())
        gts.append(t.view(-1).numpy())
    return spearmanr_np(np.concatenate(preds), np.concatenate(gts))

def main():
    os.makedirs("results", exist_ok=True)

    # checkpoint路径
    ckpt_baseline = "checkpoints/ex5/baseline/best_model.pth"
    ckpt_full     = "checkpoints/ex5/val_lam0.05/best_model.pth"

    # 如果val_lam0.05不存在则fallback到full
    if not os.path.exists(ckpt_full):
        ckpt_full = "checkpoints/ex5/full/best_model.pth"
    if not os.path.exists(ckpt_full):
        print(f"[ERROR] full model checkpoint not found, tried:\n"
              f"  checkpoints/ex5/val_lam0.05/best_model.pth\n"
              f"  checkpoints/ex5/full/best_model.pth")
        return

    print(f"[LOAD] baseline: {ckpt_baseline}")
    print(f"[LOAD] full:     {ckpt_full}")

    model_base = load_model(ckpt_baseline)
    model_full = load_model(ckpt_full)

    test_set = AQADataset(
        "data/kimore_ex5_test.pkl", "data/kimore_ex5_test_y.pkl",
        train=False, target_frames=Config.max_frame,
        do_augment=False, use_dummy=False
    )
    loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False)

    ps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]

    srcc_base, srcc_full = [], []
    for p in ps:
        fn = lambda x, _p=p: apply_joint_drop(x, _p, seed=0)
        sb = eval_srcc(model_base, loader, fn)
        sf = eval_srcc(model_full, loader, fn)
        srcc_base.append(sb)
        srcc_full.append(sf)
        print(f"  p={p:.2f}  baseline={sb:.4f}  full={sf:.4f}")

    # 计算degradation（相对clean）
    drop_base = [srcc_base[0] - s for s in srcc_base]
    drop_full = [srcc_full[0] - s for s in srcc_full]

    # 保存csv
    with open("results/robust_jointdrop.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p", "srcc_baseline", "srcc_full", "drop_baseline", "drop_full"])
        for i, p in enumerate(ps):
            w.writerow([p, srcc_base[i], srcc_full[i], drop_base[i], drop_full[i]])
    print("[SAVE] results/robust_jointdrop.csv")

    # 画图：SRCC degradation对比
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p*100 for p in ps], drop_base, "o--",
            color="#d62728", label="Baseline ($\\lambda{=}0$)", linewidth=1.8, markersize=7)
    ax.plot([p*100 for p in ps], drop_full, "s-",
            color="#1f77b4", label="Ours ($\\lambda^*{=}0.05$)", linewidth=1.8, markersize=7)
    ax.fill_between([p*100 for p in ps], drop_base, drop_full,
                    alpha=0.10, color="#2ca02c",
                    label="Robustness gain")
    ax.set_xlabel("Missing joint ratio (%)", fontsize=12)
    ax.set_ylabel(r"SRCC degradation $\Delta$SRCC $\downarrow$", fontsize=12)
    ax.set_title("Robustness under missing joints (KIMORE ex5)", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-1, 42)
    plt.tight_layout()
    plt.savefig("results/robust_jointdrop_drop_srcc.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("results/robust_jointdrop_drop_srcc.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[SAVE] results/robust_jointdrop_drop_srcc.png")

if __name__ == "__main__":
    main()
