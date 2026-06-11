"""脚本化专家: 基于仿真状态的相位控制器, 用于生成"专家 demo"。

相位机: HOVER(移到方块正上方) -> DESCEND(下降到抓取高度) -> GRASP(闭爪保持)
        -> LIFT(抬升)。输出与 BC 策略同一动作空间, 旋转增量恒为 0 (保持
        初始 top-down 姿态)。
"""
from __future__ import annotations

import numpy as np

from sim2real.sim.env import DPOS_MAX, ManipEnv
from sim2real.sim import scene as scene_mod

# 末端参考点 = hand body (法兰)。指尖 pad 在 hand-0.094 ~ hand-0.111 之间,
# 指根碰撞网格最低点 hand-0.090 —— GRASP 高度需让 pads 包住方块
# 且指根高于方块顶 (25mm 方块 + hand=0.122: pads [0.011,0.028]✓ 指根 0.032>0.025✓)
HOVER_HEIGHT = 0.24
GRASP_HEIGHT = 0.122
LIFT_HEIGHT = 0.30
POS_TOL = 0.012
GRASP_HOLD_STEPS = 12


class ScriptedExpert:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.phase = "HOVER"
        self._hold = 0

    def act(self, env: ManipEnv) -> np.ndarray:
        ee = env.ee_pos
        cube = env.cube_pos
        z0 = scene_mod.TABLE_TOP_Z
        a = np.zeros(7, dtype=np.float32)
        a[6] = -1.0  # 默认张开

        if self.phase == "HOVER":
            target = np.array([cube[0], cube[1], z0 + HOVER_HEIGHT])
            if np.linalg.norm(target - ee) < POS_TOL:
                self.phase = "DESCEND"
        if self.phase == "DESCEND":
            target = np.array([cube[0], cube[1], z0 + GRASP_HEIGHT])
            if abs(ee[2] - target[2]) < 0.006 and np.linalg.norm(target[:2] - ee[:2]) < POS_TOL:
                self.phase = "GRASP"
        if self.phase == "GRASP":
            target = np.array([cube[0], cube[1], z0 + GRASP_HEIGHT])
            a[6] = 1.0
            self._hold += 1
            if self._hold >= GRASP_HOLD_STEPS:
                self.phase = "LIFT"
        if self.phase == "LIFT":
            target = np.array([cube[0], cube[1], z0 + LIFT_HEIGHT])
            a[6] = 1.0

        dpos = (target - ee) / DPOS_MAX
        a[:3] = np.clip(dpos, -1.0, 1.0)
        return a
