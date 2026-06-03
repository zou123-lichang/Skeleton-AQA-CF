import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


# Kinect25 edges（用于骨长尺度归一化，可和graph.py保持一致）
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


def resize_sequence(seq, target_frames=100):
    """seq: (T,V,C) -> (target_frames,V,C)"""
    T, V, C = seq.shape
    if T == target_frames:
        return seq.astype(np.float32)

    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_frames)
    out = np.zeros((target_frames, V, C), dtype=np.float32)

    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(x_new, x_old, seq[:, v, c])

    return out.astype(np.float32)


def root_center(seq, root=20):
    """每帧减去root joint坐标，seq: (T,V,3)"""
    r = seq[:, root:root+1, :]  # (T,1,3)
    return seq - r


def scale_norm_by_bone(seq, edges=KINECT25_EDGES):
    """用平均骨长做尺度归一化，seq: (T,V,3)"""
    lens = []
    for i, j in edges:
        d = np.linalg.norm(seq[:, i, :] - seq[:, j, :], axis=-1)  # (T,)
        lens.append(d)
    mean_len = float(np.mean(np.stack(lens, axis=0)))
    return seq / (mean_len + 1e-6)


def rand_rotation_y(seq, deg=10.0):
    """绕y轴小角度旋转，seq: (T,V,3)"""
    a = np.deg2rad(np.random.uniform(-deg, deg))
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, 0, sa],
                  [0,  1, 0 ],
                  [-sa,0, ca]], dtype=np.float32)
    return seq @ R.T


def rand_time_crop_then_resize(seq, target_frames=100, min_ratio=0.8):
    """随机裁剪一段，再插值回 target_frames"""
    T = seq.shape[0]
    crop_len = np.random.randint(int(T * min_ratio), T + 1)
    start = np.random.randint(0, T - crop_len + 1)
    cropped = seq[start:start+crop_len]
    return resize_sequence(cropped, target_frames=target_frames)


class AQADataset(Dataset):
    """
    重要变化：
    1) 不再在 dataset 内部做 80/20 切分（因为你已经有 train.pkl / test.pkl）
    2) train=True 只用于控制是否做增强
    """
    def __init__(self, data_path, label_path=None,
                 train=False,
                 target_frames=100,
                 use_dummy=False,
                 do_root_center=True,
                 do_scale_norm=True,
                 do_augment=False):
        self.train = train
        self.target_frames = target_frames
        self.use_dummy = use_dummy

        self.do_root_center = do_root_center
        self.do_scale_norm = do_scale_norm
        self.do_augment = do_augment

        if self.use_dummy:
            self.data = [np.random.rand(np.random.randint(80, 120), 25, 3).astype(np.float32) for _ in range(200)]
            self.labels = (np.random.rand(200) * 50.0).astype(np.float32)
        else:
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)   # list of (T,V,C)
            with open(label_path, "rb") as f:
                self.labels = pickle.load(f) # list/array of float

            assert len(self.data) == len(self.labels), f"len mismatch {len(self.data)} vs {len(self.labels)}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]      # (T,V,C)
        y = float(self.labels[idx])

        # 安全：取xyz前三维
        if seq.shape[2] > 3:
            seq = seq[:, :, :3]

        # 时间对齐
        if self.train and self.do_augment:
            seq = rand_time_crop_then_resize(seq, target_frames=self.target_frames, min_ratio=0.8)
        else:
            seq = resize_sequence(seq, target_frames=self.target_frames)

        # 归一化
        if self.do_root_center:
            seq = root_center(seq, root=20)
        if self.do_scale_norm:
            seq = scale_norm_by_bone(seq)

        # 训练增强（轻量）
            # 训练增强（轻量）
        if self.train and self.do_augment:
            seq = rand_rotation_y(seq, deg=10.0)
            noise = np.random.normal(0.0, 0.005, size=seq.shape).astype(np.float32)
            seq = seq + noise

        # (T,V,3) -> (3,T,V)
        seq = np.transpose(seq, (2, 0, 1)).astype(np.float32)

        return torch.from_numpy(seq), torch.tensor(y).float()
