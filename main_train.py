import os

# Fix OpenMP runtime conflicts on Windows BEFORE importing numpy/torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset


from config import Config
from models.st_gcn import Model
from dataset import AQADataset


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False







def rankdata(a: np.ndarray) -> np.ndarray:
    """Tie-aware rankdata (average ranks for ties), 1-based ranks."""
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    a_sorted = a[sorter]
    n = a.shape[0]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1], True]
    idx = np.flatnonzero(obs)

    ranks = np.empty(n, dtype=np.float64)
    for start, end in zip(idx[:-1], idx[1:]):
        r = 0.5 * (start + end - 1) + 1.0
        ranks[sorter[start:end]] = r
    return ranks


def pearsonr_np(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    return float((x * y).sum() / (denom + 1e-12))


def spearmanr_np(x, y) -> float:
    rx = rankdata(np.asarray(x))
    ry = rankdata(np.asarray(y))
    return pearsonr_np(rx, ry)

def ranknet_loss(pred, target):
    """
    pred/target: shape (N,)
    RankNet-style pairwise loss; helps SRCC.
    """
    dp = pred.unsqueeze(1) - pred.unsqueeze(0)      # (N,N)
    dt = target.unsqueeze(1) - target.unsqueeze(0)  # (N,N)
    s = dt.sign()
    mask = s != 0
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return torch.log1p(torch.exp(-s[mask] * dp[mask])).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets = [], []
    for data, target in loader:
        data = data.to(device)
        out = model(data).view(-1).detach().cpu().numpy()
        preds.append(out)
        targets.append(target.view(-1).detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    srcc = spearmanr_np(preds, targets)
    plcc = pearsonr_np(preds, targets)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(math.sqrt(np.mean((preds - targets) ** 2)))
    return {"srcc": srcc, "plcc": plcc, "mae": mae, "rmse": rmse}


def get_ckpt_name() -> str:
    return "best_model_att.pth" if getattr(Config, "use_temporal_attention", False) else "best_model.pth"


def train_one_seed(seed: int):
    set_seed(seed)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 训练集：增强版（只给 train 用）
    full_train_aug = AQADataset(
        Config.TRAIN_DATA_PATH, Config.TRAIN_LABEL_PATH,
        train=True,
        target_frames=Config.max_frame,
        do_augment=bool(getattr(Config, "use_augment", True)),
        use_dummy=False
    )

    # 训练集：非增强版（只给 val 用）
    full_train_raw = AQADataset(
        Config.TRAIN_DATA_PATH, Config.TRAIN_LABEL_PATH,
        train=False,
        target_frames=Config.max_frame,
        do_augment=False,
        use_dummy=False
    )

    # 测试集：永远不增强
    test_set = AQADataset(
        Config.TEST_DATA_PATH, Config.TEST_LABEL_PATH,
        train=False,
        target_frames=Config.max_frame,
        do_augment=False,
        use_dummy=False
    )

    # recommended: split a small validation from training set
    val_ratio = float(getattr(Config, "val_ratio", 0.1))
    n = len(full_train_raw)
    g = torch.Generator().manual_seed(seed)

    if val_ratio > 0:
        n_val = max(1, int(n * val_ratio))
        perm = torch.randperm(n, generator=g).tolist()
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_set = Subset(full_train_aug, train_idx)  # 有增强
        val_set = Subset(full_train_raw, val_idx)  # 无增强
    else:
        train_set, val_set = full_train_aug, None  # 全部用增强训练

    loader_train = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False)
    loader_val = None if val_set is None else DataLoader(val_set, batch_size=Config.batch_size, shuffle=False)

    model = Model(
        in_channels=Config.coords_dim,
        num_class=1,
        graph_args={'layout': 'kinect25', 'strategy': 'spatial'},
        use_temporal_attention=getattr(Config, "use_temporal_attention", False),
    ).to(Config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.MSELoss()

    best_key = -1.0
    best_epoch = -1
    ckpt_name = get_ckpt_name()
    ckpt_path = os.path.join(Config.SAVE_DIR, ckpt_name)

    exp_prefix = os.path.basename(Config.TRAIN_DATA_PATH).replace("_train.pkl", "")
    tag = exp_prefix + ("_att" if getattr(Config, "use_temporal_attention", False) else "_base")
    log_path = os.path.join("results", f"{tag}_seed{seed}.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss",
                         "val_srcc", "val_plcc", "val_mae", "val_rmse",
                         "test_srcc", "test_plcc", "test_mae", "test_rmse"])

        for epoch in range(Config.epochs):
            model.train()
            epoch_loss = 0.0

            for data, target in loader_train:
                data = data.to(Config.DEVICE)
                target = target.to(Config.DEVICE)

                optimizer.zero_grad()
                output = model(data).view(-1)

                rnk = ranknet_loss(output, target.view(-1))

                mse = criterion(output, target.view(-1))
                lam = float(getattr(Config, "lambda_rank", 0.0))
                if lam > 0:
                    rnk = ranknet_loss(output, target.view(-1))
                    loss = mse + lam * rnk
                else:
                    loss = mse
                loss.backward()



                optimizer.step()

                epoch_loss += float(loss.item())

            scheduler.step()
            train_loss = epoch_loss / max(1, len(loader_train))

            test_metrics = evaluate(model, loader_test, Config.DEVICE)
            if loader_val is not None:
                val_metrics = evaluate(model, loader_val, Config.DEVICE)
                key = val_metrics["srcc"]  # select best by val SRCC
            else:
                val_metrics = {"srcc": float("nan"), "plcc": float("nan"), "mae": float("nan"), "rmse": float("nan")}
                key = test_metrics["srcc"]  # fallback

            print(
                f"Epoch {epoch+1:03d}: loss={train_loss:.4f} | "
                f"VAL SRCC={val_metrics['srcc']:.4f} PLCC={val_metrics['plcc']:.4f} | "
                f"TEST SRCC={test_metrics['srcc']:.4f} PLCC={test_metrics['plcc']:.4f}"
            )

            writer.writerow([
                epoch + 1, f"{train_loss:.6f}",
                f"{val_metrics['srcc']:.6f}", f"{val_metrics['plcc']:.6f}", f"{val_metrics['mae']:.6f}", f"{val_metrics['rmse']:.6f}",
                f"{test_metrics['srcc']:.6f}", f"{test_metrics['plcc']:.6f}", f"{test_metrics['mae']:.6f}", f"{test_metrics['rmse']:.6f}",
            ])

            # Save BEST only (no overwrite by last epoch)
            if key > best_key:
                best_key = key
                best_epoch = epoch + 1
                torch.save(model.state_dict(), ckpt_path)  # for explain.py convenience
                torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, f"best_{tag}_seed{seed}.pth"))
                print(f">>> Best updated at epoch {best_epoch} (key={best_key:.4f}) -> {ckpt_name}")

        # Save last for debugging
        torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, f"last_{tag}_seed{seed}.pth"))

    # Final test using best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    final_test = evaluate(model, loader_test, Config.DEVICE)
    summary = {"seed": seed, "best_epoch": best_epoch, "best_val_srcc": best_key, **final_test, "tag": tag}
    print("[SUMMARY]", summary)
    return summary


def main():
    seeds = getattr(Config, "seeds", [Config.seed])
    summaries = [train_one_seed(int(s)) for s in seeds]

    srccs = np.array([s["srcc"] for s in summaries], dtype=np.float64)
    plccs = np.array([s["plcc"] for s in summaries], dtype=np.float64)
    maes = np.array([s["mae"] for s in summaries], dtype=np.float64)
    rmses = np.array([s["rmse"] for s in summaries], dtype=np.float64)

    def mean_std(x):
        return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

    srcc_m, srcc_s = mean_std(srccs)
    plcc_m, plcc_s = mean_std(plccs)
    mae_m, mae_s = mean_std(maes)
    rmse_m, rmse_s = mean_std(rmses)

    print(f"[MEAN±STD] SRCC={srcc_m:.4f}±{srcc_s:.4f} | PLCC={plcc_m:.4f}±{plcc_s:.4f} | "
          f"MAE={mae_m:.4f}±{mae_s:.4f} | RMSE={rmse_m:.4f}±{rmse_s:.4f}")

    out_path = os.path.join("results", f"summary_{summaries[0]['tag']}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "seed", "best_epoch", "srcc", "plcc", "mae", "rmse"])
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s[k] for k in writer.fieldnames})
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
