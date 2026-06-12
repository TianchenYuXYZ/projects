"""示范数据集: npz 目录 -> (image, proprio, action token) 监督样本。

切分按 demo 而不是按帧 (帧级切分会让同一条轨迹的相邻帧横跨 train/val,
验证指标虚高)。亮度抖动只作用于训练split。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vla_safety.vla.model import P_MEAN, P_STD
from vla_safety.vla.tokenizer import ActionTokenizer


def load_demo_arrays(demo_dir: str | Path):
    """返回 (images uint8 (N,H,W,3), proprios (N,4), actions (N,4), demo_ids (N,))。"""
    files = sorted(Path(demo_dir).glob("demo_*.npz"))
    if not files:
        raise FileNotFoundError(f"{demo_dir} 下没有 demo_*.npz; 先运行 01_gen_demos.py")
    imgs, props, acts, ids = [], [], [], []
    for i, f in enumerate(files):
        z = np.load(f)
        if not bool(z["success"]):
            continue
        n = len(z["actions"])
        imgs.append(z["images"])
        props.append(z["proprios"])
        acts.append(z["actions"])
        ids.append(np.full(n, i, dtype=np.int64))
    return (np.concatenate(imgs), np.concatenate(props),
            np.concatenate(acts), np.concatenate(ids))


class DemoDataset(Dataset):
    def __init__(self, images, proprios, actions, train: bool,
                 brightness_jitter: float = 0.08, seed: int = 0):
        self.images = images
        self.proprios = ((proprios - P_MEAN) / P_STD).astype(np.float32)
        self.tokens = ActionTokenizer().encode(actions)
        self.train = train
        self.jitter = brightness_jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = self.images[i].astype(np.float32) / 255.0
        if self.train and self.jitter > 0:
            img = np.clip(img * (1.0 + self.rng.uniform(-self.jitter, self.jitter)),
                          0.0, 1.0)
        img = (img - 0.5) / 0.5
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(self.proprios[i]),
                torch.from_numpy(self.tokens[i]))


def make_splits(demo_dir: str | Path, val_frac: float, seed: int):
    """按 demo 切分 -> (train_ds, val_ds, stats dict)。"""
    images, proprios, actions, demo_ids = load_demo_arrays(demo_dir)
    uniq = np.unique(demo_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_set = set(perm[:n_val].tolist())
    val_mask = np.isin(demo_ids, list(val_set))

    train_ds = DemoDataset(images[~val_mask], proprios[~val_mask],
                           actions[~val_mask], train=True, seed=seed)
    val_ds = DemoDataset(images[val_mask], proprios[val_mask],
                         actions[val_mask], train=False, seed=seed)
    stats = {"n_demos": int(len(uniq)), "n_val_demos": int(n_val),
             "n_train_frames": int((~val_mask).sum()),
             "n_val_frames": int(val_mask.sum())}
    return train_ds, val_ds, stats
