"""RT-2 风格动作离散化: 连续动作 <-> 256-bin token。

RT-2 把 6-DOF 速度/旋转离散成整数 bin 当 text token 生成; 这里对
4 维动作 (v_xyz/V_MAX, 夹爪) 做同样的均匀分箱。recovery arm 也走
这套编码 —— 安全层输出的是 VLA 原生 token, 不是外部 planner 的几何轨迹。
"""
from __future__ import annotations

import numpy as np

from vla_safety.common import ACTION_DIM, N_BINS


class ActionTokenizer:
    def __init__(self, n_bins: int = N_BINS, dim: int = ACTION_DIM):
        self.n_bins = n_bins
        self.dim = dim

    def encode(self, a: np.ndarray) -> np.ndarray:
        """[-1,1]^dim -> {0..n_bins-1}^dim。支持批量 (..., dim)。"""
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        bins = np.floor((a + 1.0) / 2.0 * self.n_bins).astype(np.int64)
        return np.clip(bins, 0, self.n_bins - 1)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """token -> bin 中心值, [-1,1]^dim。支持批量。"""
        t = np.asarray(tokens, dtype=np.float64)
        return ((t + 0.5) / self.n_bins) * 2.0 - 1.0

    def roundtrip_error_bound(self) -> float:
        """量化误差上界 = 半个 bin 宽。"""
        return 1.0 / self.n_bins
