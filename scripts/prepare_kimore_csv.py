# scripts/prepare_kimore_csv.py
import os, argparse, pickle
import numpy as np

TARGET_LEN = 100

def load_csv(path):
    # 兼容纯数值CSV
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except ValueError:
        arr = np.genfromtxt(path, delimiter=",", dtype=np.float32, invalid_raise=False)
        if np.isnan(arr).any():
            keep_r = ~np.isnan(arr).all(axis=1)
            keep_c = ~np.isnan(arr).all(axis=0)
            arr = arr[keep_r][:, keep_c]
            arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float32)

def resample(seq, target_len=TARGET_LEN):
    # seq: (T,V,C)
    T, V, C = seq.shape
    if T == target_len:
        return seq.astype(np.float32)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(x_new, x_old, seq[:, v, c])
    return out

def normalize(seq):
    # 每帧去中心 + 全局std缩放（只对xy做去中心/缩放也行，这里对xyz整体做）
    seq = seq.astype(np.float32)
    center = seq[:, :, :3].mean(axis=1, keepdims=True)
    seq[:, :, :3] -= center
    s = float(np.std(seq[:, :, :3]))
    if s < 1e-6:
        s = 1.0
    seq[:, :, :3] /= s
    return seq

def decide_parse_mode(first_frame_vec):
    """
    输入: (D_frame,) 一帧的100维向量
    输出: (V, mode)
    mode:
      - 'xyz_only_25' : D=75 -> 25 joints xyz
      - 'interleaved4_25' : D=100 且每4维一组 (x,y,z,conf/state)
      - 'xyz_plus_block_25' : D=100 且前75是xyz，后25是conf/state
      - 'xyz_only_generic' / 'interleaved4_generic'
    """
    D = first_frame_vec.shape[0]

    def conf_score(conf):
        # conf/state 常见为 0/1/2/3 或 [0,1]；给一个“像离散值”的打分
        conf = np.asarray(conf)
        ok_range = (conf >= -0.5) & (conf <= 5.5)
        near_int = np.abs(conf - np.round(conf)) < 0.05
        return float(np.mean(ok_range & near_int))

    if D == 75:
        return 25, "xyz_only_25"

    if D == 100:
        # 两种常见排布： interleaved (25*4) 或 xyz(75)+conf(25)
        conf_inter = first_frame_vec[3::4]   # 25个
        conf_block = first_frame_vec[75:]    # 25个
        s_inter = conf_score(conf_inter)
        s_block = conf_score(conf_block)
        if s_inter >= s_block:
            return 25, "interleaved4_25"
        else:
            return 25, "xyz_plus_block_25"

    # 泛化兜底
    if D % 3 == 0:
        V = D // 3
        return V, "xyz_only_generic"
    if D % 4 == 0:
        V = D // 4
        return V, "interleaved4_generic"

    raise ValueError(f"Unknown frame dimension D={D}. Please print a row of Train_X.csv.")

def frames_to_seq(frames_2d, V, mode):
    """
    frames_2d: (T, D_frame)
    return: (T, V, 3)
    """
    T, D = frames_2d.shape

    if mode == "xyz_only_25":
        # D=75
        return frames_2d.reshape(T, 25, 3).astype(np.float32)

    if mode == "interleaved4_25":
        # D=100 -> (25,4) 取前3
        return frames_2d.reshape(T, 25, 4)[:, :, :3].astype(np.float32)

    if mode == "xyz_plus_block_25":
        # D=100 -> 前75是xyz
        return frames_2d[:, :75].reshape(T, 25, 3).astype(np.float32)

    if mode == "xyz_only_generic":
        return frames_2d.reshape(T, V, 3).astype(np.float32)

    if mode == "interleaved4_generic":
        return frames_2d.reshape(T, V, 4)[:, :, :3].astype(np.float32)

    raise ValueError(f"Unsupported mode {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="KIMORE_DATASET")
    ap.add_argument("--ex", type=str, default="Kimore ex5")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--target_len", type=int, default=100)
    args = ap.parse_args()

    ex_dir = os.path.join(args.root, args.ex)
    x_path = os.path.join(ex_dir, "Train_X.csv")
    y_path = os.path.join(ex_dir, "Train_Y.csv")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Missing Train_X/Y.csv in {ex_dir}")

    X = load_csv(x_path)     # (N_seq*T, D_frame)
    Y = load_csv(y_path)     # (N_seq, 1) or (N_seq, k)
    y = Y[:, 0].astype(np.float32)

    N_seq = y.shape[0]
    n_rows, D_frame = X.shape

    if n_rows == N_seq:
        # 兜底：如果真是每行一个样本（不太像你这份）
        raise ValueError("This X looks like per-sample rows, but your earlier run showed frames-stacked format.")

    if n_rows % N_seq != 0:
        raise ValueError(f"X rows {n_rows} not divisible by Y rows {N_seq}. Cannot align sequences.")

    T = n_rows // N_seq

    # 决定如何把一帧100维解析成 joints
    V, mode = decide_parse_mode(X[0])
    print(f"[INFO] {args.ex}: N_seq={N_seq}, frames_per_seq(T)={T}, frame_dim(D)={D_frame}")
    print(f"[INFO] parse: V={V}, mode={mode}  (we will use xyz, C=3)")
    print(f"[INFO] label range: {float(y.min()):.3f} .. {float(y.max()):.3f}")

    # 组回序列
    seqs = []
    for i in range(N_seq):
        frames = X[i*T:(i+1)*T]               # (T, D_frame)
        seq = frames_to_seq(frames, V, mode)  # (T, V, 3)
        seq = resample(seq, args.target_len)
        seq = normalize(seq)
        seqs.append(seq)

    # train/test split
    rng = np.random.RandomState(args.seed)
    idx = np.arange(N_seq)
    rng.shuffle(idx)
    cut = int(N_seq * (1.0 - args.test_ratio))
    tr, te = idx[:cut], idx[cut:]

    train_data = [seqs[i] for i in tr]
    train_y = [float(y[i]) for i in tr]
    test_data  = [seqs[i] for i in te]
    test_y  = [float(y[i]) for i in te]

    os.makedirs("data", exist_ok=True)
    tag = args.ex.replace(" ", "_").lower()

    with open(f"data/{tag}_train.pkl", "wb") as f: pickle.dump(train_data, f)
    with open(f"data/{tag}_train_y.pkl", "wb") as f: pickle.dump(train_y, f)
    with open(f"data/{tag}_test.pkl", "wb") as f: pickle.dump(test_data, f)
    with open(f"data/{tag}_test_y.pkl", "wb") as f: pickle.dump(test_y, f)

    print("[OK] saved:")
    print(f"  data/{tag}_train.pkl  ({len(train_data)})")
    print(f"  data/{tag}_test.pkl   ({len(test_data)})")

if __name__ == "__main__":
    main()
