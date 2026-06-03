import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

torch.backends.cudnn.benchmark = True

from config import Config
from dataset import AQADataset
from models.st_gcn import Model

KINECT25_EDGES = [
    (0, 1),
    (1, 20), (2, 20), (2, 3),
    (4, 20), (4, 5), (5, 6), (6, 7),
    (8, 20), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
    (21, 22), (22, 7),
    (23, 24), (24, 11),
]

def bone_length_loss(cf_data, org_data, edges):
    cf = cf_data.permute(0, 2, 3, 1)[..., :3]
    org = org_data.permute(0, 2, 3, 1)[..., :3]
    loss = 0.0
    for i, j in edges:
        cf_len = torch.sqrt(((cf[:, :, i, :] - cf[:, :, j, :]) ** 2).sum(dim=-1) + 1e-8)
        org_len = torch.sqrt(((org[:, :, i, :] - org[:, :, j, :]) ** 2).sum(dim=-1) + 1e-8)
        loss = loss + torch.mean(torch.abs(cf_len - org_len))
    return loss / len(edges)

def temporal_smooth_loss(x):
    return torch.mean((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2)

@torch.no_grad()
def predict(model, x):
    return float(model(x).view(-1)[0].item())

def generate_counterfactual(model, original_sample, use_l1=True, use_smooth=True, use_bone=True):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    org = original_sample.detach().to(Config.DEVICE)
    cf = org.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([cf], lr=Config.cf_lr)

    org_score = predict(model, org)
    y_max = float(getattr(Config, "cf_ymax", 50.0))
    margin = float(getattr(Config, "cf_margin", 5.0))
    target_score = min(org_score + margin, y_max)

    for _ in range(int(Config.cf_steps)):
        optimizer.zero_grad()
        score = model(cf).view(-1)[0]
        loss_score = (score - score.new_tensor(target_score)) ** 2
        loss = loss_score
        if use_l1:
            loss = loss + float(Config.lambda_l1) * torch.mean(torch.abs(cf - org))
        if use_smooth:
            loss = loss + float(Config.lambda_smooth) * temporal_smooth_loss(cf)
        if use_bone:
            loss = loss + float(Config.lambda_bone) * bone_length_loss(cf, org, KINECT25_EDGES)
        loss.backward()
        optimizer.step()

    cf_score = predict(model, cf)
    with torch.no_grad():
        l1 = torch.mean(torch.abs(cf - org)).item()
        smooth = temporal_smooth_loss(cf).item()
        bone = bone_length_loss(cf, org, KINECT25_EDGES).item()

    metrics = {
        "org_score": org_score,
        "target_score": float(target_score),
        "cf_score": cf_score,
        "delta_score": cf_score - org_score,
        "l1": float(l1),
        "smooth": float(smooth),
        "bone": float(bone),
    }
    return cf.detach().cpu(), metrics

def process_one_sample(args):
    i, x, methods, vis_set, ckpt_path = args
    results = []
    vis_data = {}
    x_input = x.unsqueeze(0)

    model = Model(
        in_channels=Config.coords_dim,
        num_class=1,
        graph_args={'layout': 'kinect25', 'strategy': 'spatial'},
        use_temporal_attention=getattr(Config, "use_temporal_attention", False),
    ).to(Config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    model.eval()

    for name, kw in methods:
        cf, m = generate_counterfactual(model, x_input, **kw)
        results.append([i, name, m["org_score"], m["cf_score"], m["delta_score"], m["l1"], m["smooth"], m["bone"]])
        if i in vis_set:
            vis_data[name] = cf[0].numpy()
        if abs(m["delta_score"]) > 20:
            print(f"[WARN] idx={i} {name} delta={m['delta_score']:.2f} too large.")

    return i, results, vis_data

def plot_skeleton_xy(ax, data_ctv, edges, frame_idx, title=None, alpha=1.0, lw=2.0):
    sk = data_ctv[:, frame_idx, :]
    x, y = sk[0], sk[1]
    for (i, j) in edges:
        ax.plot([x[i], x[j]], [y[i], y[j]], alpha=alpha, linewidth=lw)
    ax.scatter(x, y, s=15, alpha=alpha)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")

def visualize_methods(org_ctv, cf_dict, frames=(20, 50, 80), save_path="results/action_rectification_kinect25.png"):
    method_names = list(cf_dict.keys())
    n_rows = 1 + len(method_names)
    n_cols = len(frames)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.2 * n_rows))
    for k, f in enumerate(frames):
        plot_skeleton_xy(axs[0, k], org_ctv, KINECT25_EDGES, f, title=f"Original (t={f})", alpha=0.9, lw=2.0)
    for r, name in enumerate(method_names, start=1):
        cf_ctv = cf_dict[name]
        for k, f in enumerate(frames):
            ax = axs[r, k]
            plot_skeleton_xy(ax, org_ctv, KINECT25_EDGES, f, title=f"{name} (t={f})", alpha=0.25, lw=1.5)
            sk = cf_ctv[:, f, :]
            x, y = sk[0], sk[1]
            for (i, j) in KINECT25_EDGES:
                ax.plot([x[i], x[j]], [y[i], y[j]], alpha=0.9, linewidth=2.0)
            ax.scatter(x, y, s=15, alpha=0.9)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[VIS] saved: {save_path}")

def summarize(rows):
    keys = ["delta_score", "l1", "smooth", "bone"]
    out = {}
    for k in keys:
        arr = np.array([r[k] for r in rows], dtype=np.float64)
        out[k + "_mean"] = float(arr.mean())
        out[k + "_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return out

def main():
    os.makedirs("results", exist_ok=True)

    ckpt_name = "best_model_att.pth" if getattr(Config, "use_temporal_attention", False) else "best_model.pth"
    ckpt_path = os.path.join(Config.SAVE_DIR, ckpt_name)

    model = Model(
        in_channels=Config.coords_dim,
        num_class=1,
        graph_args={'layout': 'kinect25', 'strategy': 'spatial'},
        use_temporal_attention=getattr(Config, "use_temporal_attention", False),
    ).to(Config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    model.eval()
    print(f"[LOAD] checkpoint: {ckpt_path}")

    test_set = AQADataset(
        Config.TEST_DATA_PATH, Config.TEST_LABEL_PATH,
        train=False, target_frames=Config.max_frame, do_augment=False, use_dummy=False
    )

    max_samples = min(int(getattr(Config, "cf_eval_samples", 20)), len(test_set))

    scored = []
    with torch.no_grad():
        for i in range(max_samples):
            x, _ = test_set[i]
            s = float(model(x.unsqueeze(0).to(Config.DEVICE)).view(-1)[0].item())
            scored.append((s, i))
    scored.sort(key=lambda t: t[0])

    k_each = int(getattr(Config, "cf_vis_k_each", 2))
    low_idx = [i for _, i in scored[:k_each]]
    mid_start = max(0, len(scored) // 2 - k_each // 2)
    mid_idx = [i for _, i in scored[mid_start:mid_start + k_each]]
    high_idx = [i for _, i in scored[-k_each:]]
    vis_indices = list(dict.fromkeys(low_idx + mid_idx + high_idx))
    vis_set = set(vis_indices)
    print(f"[VIS] selected indices (low/mid/high): {vis_indices}")

    methods = [
        ("CF-ScoreOnly", dict(use_l1=False, use_smooth=False, use_bone=False)),
        ("CF-L1+Smooth", dict(use_l1=True, use_smooth=True, use_bone=False)),
        ("CF-Full(+Bone)", dict(use_l1=True, use_smooth=True, use_bone=True)),
    ]

    NUM_WORKERS = 4
    tasks = [(i, test_set[i][0], methods, vis_set, ckpt_path) for i in range(max_samples)]

    all_detail_rows = []
    per = {name: [] for name, _ in methods}
    vis_cache = {idx: {} for idx in vis_indices}

    print(f"[RUN] {max_samples} samples x 3 methods, {NUM_WORKERS} threads...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_one_sample, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            i, results, vis_data = future.result()
            done += 1
            if done % 10 == 0 or done == max_samples:
                print(f"  [{done}/{max_samples}] done")
            for row in results:
                all_detail_rows.append(row)
                name = row[1]
                per[name].append({
                    "delta_score": row[4],
                    "l1": row[5],
                    "smooth": row[6],
                    "bone": row[7],
                })
            if i in vis_set:
                vis_cache[i].update(vis_data)

    out_csv = os.path.join("results", "cf_metrics_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "n",
                    "delta_mean", "delta_std",
                    "l1_mean", "l1_std",
                    "smooth_mean", "smooth_std",
                    "bone_mean", "bone_std"])
        for name, _ in methods:
            s = summarize(per[name])
            w.writerow([name, len(per[name]),
                        f"{s['delta_score_mean']:.6f}", f"{s['delta_score_std']:.6f}",
                        f"{s['l1_mean']:.6f}", f"{s['l1_std']:.6f}",
                        f"{s['smooth_mean']:.6f}", f"{s['smooth_std']:.6f}",
                        f"{s['bone_mean']:.6f}", f"{s['bone_std']:.6f}"])
    print(f"[SAVE] {out_csv}")

    out_detail = os.path.join("results", "cf_metrics_detail.csv")
    all_detail_rows.sort(key=lambda r: r[0])
    with open(out_detail, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "method", "org_score", "cf_score", "delta", "l1", "smooth", "bone"])
        w.writerows(all_detail_rows)
    print(f"[SAVE] {out_detail}")

    frames = getattr(Config, "cf_vis_frames", (20, 50, 80))
    for idx in vis_indices:
        x_case, _ = test_set[idx]
        org_np = x_case.numpy()
        cf_dict = vis_cache[idx]
        save_path = os.path.join("results", f"action_rectification_kinect25_idx{idx}.png")
        visualize_methods(org_np, cf_dict, frames=frames, save_path=save_path)

if __name__ == "__main__":
    main()
