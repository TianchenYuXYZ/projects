"""脚本化专家: rise -> cruise -> descend -> grasp -> lift 五相位状态机。

输出与 VLA 同一动作契约 (归一化速度 + 夹爪)。示范采集在无障碍训练
分布上进行 —— VLA 学到的是 "看见方块就直线过去" 的行为, 避障概念
被刻意排除在训练分布之外 (这正是安全层要填的缺口)。
"""
from __future__ import annotations

import numpy as np

from vla_safety.common import V_MAX


class ScriptedExpert:
    def __init__(self, travel_z: float, grasp_z: float,
                 kp: float = 6.0, noise: float = 0.06,
                 rng: np.random.Generator | None = None):
        self.travel_z = travel_z
        self.grasp_z = grasp_z
        self.kp = kp
        self.noise = noise
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self) -> None:
        self.phase = "rise"
        self._grasp_hold = 0

    def act(self, ee: np.ndarray, cube: np.ndarray, decision_dt: float) -> np.ndarray:
        """ee/cube 世界系位置 -> (4,) 归一化指令。decision_dt 用于 grasp 计时。"""
        grip = -1.0
        if self.phase == "rise":
            target = np.array([ee[0], ee[1], self.travel_z])
            if abs(ee[2] - self.travel_z) < 0.02:
                self.phase = "cruise"
        if self.phase == "cruise":
            target = np.array([cube[0], cube[1], self.travel_z])
            if np.linalg.norm(ee[:2] - cube[:2]) < 0.012:
                self.phase = "descend"
        if self.phase == "descend":
            target = np.array([cube[0], cube[1], self.grasp_z])
            if abs(ee[2] - self.grasp_z) < 0.008:
                self.phase = "grasp"
        if self.phase == "grasp":
            target = np.array([cube[0], cube[1], self.grasp_z])
            grip = 1.0
            self._grasp_hold += decision_dt
            if self._grasp_hold >= 0.35:
                self.phase = "lift"
        if self.phase == "lift":
            target = np.array([ee[0], ee[1], self.travel_z + 0.05])
            grip = 1.0

        v = self.kp * (target - ee) / V_MAX
        v = np.clip(v, -1.0, 1.0)
        v = np.clip(v * (1.0 + self.noise * self.rng.standard_normal(3)), -1.0, 1.0)
        return np.array([v[0], v[1], v[2], grip], dtype=np.float64)
