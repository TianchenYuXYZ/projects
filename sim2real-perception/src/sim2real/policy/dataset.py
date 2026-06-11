"""训练数据集: 内存中的 (image_u8, proprio, action) 三元组。"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class BCDataset(Dataset):
    def __init__(self, images_u8: np.ndarray, proprios: np.ndarray,
                 actions: np.ndarray):
        assert len(images_u8) == len(proprios) == len(actions)
        self.images = images_u8
        self.proprios = proprios.astype(np.float32)
        self.actions = actions.astype(np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = torch.from_numpy(
            self.images[i].transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        return img, torch.from_numpy(self.proprios[i]), torch.from_numpy(self.actions[i])
