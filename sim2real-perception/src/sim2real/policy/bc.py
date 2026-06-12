"""BC 策略头: feature(2048) + proprio(8) -> action(7), 线性输出 + 消费端裁剪。

不用 tanh: 专家标签 37% dpos / 100% 夹爪是饱和 ±1, tanh 渐近线让 L1
永远差 ~0.15-0.2, 闭环里表现为 "比专家慢一拍", 状态逐步滑出数据流形
(0% 成功率的直接根因)。线性头 + clip 可精确表达饱和动作。
"""
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
        return self.net(torch.cat([feature, proprio], dim=-1))
