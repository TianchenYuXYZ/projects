"""BC 策略头: feature(2048) + proprio(8) -> action(7), tanh 输出 [-1,1]。"""
from __future__ import annotations

import torch
import torch.nn as nn

from sim2real.perception.backbone import FEATURE_DIM


class BCPolicy(nn.Module):
    def __init__(self, proprio_dim: int = 8, action_dim: int = 7,
                 hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        dims = [FEATURE_DIM + proprio_dim, *hidden]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], action_dim))
        self.net = nn.Sequential(*layers)
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim

    def forward(self, feature: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([feature, proprio], dim=-1)))
