# posthoc_importance.py
import os, numpy as np, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from dataset import AQADataset, KINECT25_EDGES
from models.st_gcn import Model

def load_model(ckpt_path: str):
    model = Model(
        in_channels=Config.coords_dim,
        num_class=1,
        graph_args={'layout': 'kinect25', 'strategy': 'spatial'},
        use_temporal_attention=False,   # 方案1：永远关掉 learned attention
    ).to(Config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    model.eval()
    return model

@torch.no_grad()
def forward_score(model, x):
    return float(model(x).view(-1)[0].item())

def saliency_time_joint(model, x):
    """
    x: (1,3,T,V) on device
    return:
      I_tv: (T,V)  alpha_t:(T,)  beta_v:(V,)
    """
    x = x.clone().detach().requires_grad_(True)
    y = model(x).view(-1)[0]
    y.backward()

    g = x.grad.detach()[0]          # (3,T,V)
    I_tv = torch.sqrt((g ** 2).sum(dim=0) + 1e-12)  # (T,V)
    I_tv = I_tv / (I_tv.sum() + 1e-12)

    alpha_t = I_tv.sum(dim=1)       # (T,)
    beta_v  = I_tv.sum(dim=0)       # (V,)
    return I_tv.cpu().numpy(), alpha_t.cpu().numpy(), beta_v.cpu().numpy()

def plot_time(alpha_t, path):
    plt.figure(figsize=(6,3))
    plt.plot(np.arange(len(alpha_t)), alpha_t)
    plt.xlabel("Frame t"); plt.ylabel("Importance α(t)")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def plot_joint(beta_v, path, topk=10):
    idx = np.argsort(-beta_v)[:topk]
    plt.figure(figsize=(6,3))
    plt.bar([str(i) for i in idx], beta_v[idx])
    plt.xlabel("Joint id"); plt.ylabel("Importance β(v)")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def plot_heatmap(I_tv, path):
    plt.figure(figsize=(7,3))
    plt.imshow(I_tv.T, aspect="auto", origin="lower")
    plt.xlabel("Frame t"); plt.ylabel("Joint v")
    plt.colorbar(label="I(t,v)")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def occlude_frames(x, frames):
    # x: (1,3,T,V)
    x2 = x.clone()
    x2[:,:,frames,:] = 0.0
    return x2

def occlude_joints(x, joints):
    x2 = x.clone()
    x2[:,:,:,joints] = 0.0
    return x2

def faithfulness_eval(model, dataset, n=50, topk_t=10, topk_v=5, seed=0):
    rng = np.random.default_rng(seed)
    drops = {"top_t":[], "rand_t":[], "top_v":[], "rand_v":[]}
    n = min(n, len(dataset))
    for i in range(n):
        x, _ = dataset[i]
        x = x.unsqueeze(0).to(Config.DEVICE)

        I_tv, alpha_t, beta_v = saliency_time_joint(model, x)
        T, V = I_tv.shape

        top_frames = np.argsort(-alpha_t)[:topk_t]
        rand_frames = rng.choice(T, size=topk_t, replace=False)

        top_joints = np.argsort(-beta_v)[:topk_v]
        rand_joints = rng.choice(V, size=topk_v, replace=False)

        y0 = forward_score(model, x)
        y_top_t  = forward_score(model, occlude_frames(x, top_frames))
        y_rand_t = forward_score(model, occlude_frames(x, rand_frames))
        y_top_v  = forward_score(model, occlude_joints(x, top_joints))
        y_rand_v = forward_score(model, occlude_joints(x, rand_joints))

        drops["top_t"].append(y0 - y_top_t)
        drops["rand_t"].append(y0 - y_rand_t)
        drops["top_v"].append(y0 - y_top_v)
        drops["rand_v"].append(y0 - y_rand_v)

    return {k:(float(np.mean(v)), float(np.std(v, ddof=1)) if len(v)>1 else 0.0) for k,v in drops.items()}

def main():
    os.makedirs("results", exist_ok=True)

    ckpt_path = os.path.join(Config.SAVE_DIR, "best_model.pth")
    model = load_model(ckpt_path)

    test_set = AQADataset(
        Config.TEST_DATA_PATH, Config.TEST_LABEL_PATH,
        train=False, target_frames=Config.max_frame, do_augment=False, use_dummy=False
    )

    # 选几条样本出可视化：低/中/高预测
    scores = []
    with torch.no_grad():
        for i in range(min(50, len(test_set))):
            x,_ = test_set[i]
            s = forward_score(model, x.unsqueeze(0).to(Config.DEVICE))
            scores.append((s,i))
    scores.sort()
    pick = [scores[0][1], scores[len(scores)//2][1], scores[-1][1]]

    for idx in pick:
        x,_ = test_set[idx]
        x1 = x.unsqueeze(0).to(Config.DEVICE)
        I_tv, alpha_t, beta_v = saliency_time_joint(model, x1)
        plot_time(alpha_t, f"results/imp_time_idx{idx}.png")
        plot_joint(beta_v, f"results/imp_joint_idx{idx}.png", topk=10)
        plot_heatmap(I_tv, f"results/imp_heatmap_idx{idx}.png")
        print(f"[SAVE] importance figs for idx={idx}")

    # faithfulness（遮挡验证）
    stats = faithfulness_eval(model, test_set, n=50, topk_t=10, topk_v=5, seed=0)
    print("[FAITHFULNESS]", stats)

if __name__ == "__main__":
    main()