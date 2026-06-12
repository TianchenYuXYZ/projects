"""Recovery arm 库: 离散恢复动作类别, 编码为 VLA 原生 action token 序列。

文档定义的 arms (retreat-Z / lateral-shift / freeze / hand-off) 在本环境的实例化:
  retreat_up      竖直上撤 (越过低矮障碍)
  shift_y_plus    世界系 +y 横移 (绕开路径左/右侧障碍)
  shift_y_minus   世界系 -y 横移
  retreat_x_back  沿 -x 回撤 (拉开与障碍的距离, 让 VLA 重新规划)
  freeze          冻结 (保守兜底, 等价于文档的 hand-off 前置状态)

每个 arm = steps 个 motion token (3 维), 运行时由 Monitor 解码回速度指令。
夹爪通道运行时透传当前指令 —— 安全层永远不主动张开已闭合的夹爪
(避免 recovery 把已抓取的物体扔下去, 这本身就是一条安全性质)。
"""
from __future__ import annotations

import dataclasses

import numpy as np

from vla_safety.common import V_MAX
from vla_safety.vla.tokenizer import ActionTokenizer

ARM_NAMES = ["retreat_up", "shift_y_plus", "shift_y_minus", "retreat_x_back", "freeze"]

_DIRECTIONS = {
    "retreat_up":     np.array([0.0, 0.0, 1.0]),
    "shift_y_plus":   np.array([0.0, 1.0, 0.0]),
    "shift_y_minus":  np.array([0.0, -1.0, 0.0]),
    "retreat_x_back": np.array([-1.0, 0.0, 0.0]),
    "freeze":         np.array([0.0, 0.0, 0.0]),
}


@dataclasses.dataclass
class RecoveryArm:
    index: int
    name: str
    tokens: np.ndarray        # (steps, 3) int64 — motion token 序列
    velocities: np.ndarray    # (steps, 3) float — 解码后的归一化速度 (调试/测试用)


def build_arm_library(tokenizer: ActionTokenizer, arm_speed: float,
                      steps: int) -> list[RecoveryArm]:
    """arm_speed 单位 m/s; token 编码的是归一化速度 arm_speed/V_MAX。"""
    v_norm = arm_speed / V_MAX
    assert 0.0 < v_norm <= 1.0, f"arm_speed {arm_speed} 超出 V_MAX {V_MAX}"
    arms = []
    for idx, name in enumerate(ARM_NAMES):
        v = _DIRECTIONS[name] * v_norm
        seq = np.tile(v, (steps, 1))                       # (steps, 3)
        tokens = tokenizer.encode(seq)[:, :3]              # encode 是逐元素的
        decoded = tokenizer.decode(tokens)
        arms.append(RecoveryArm(index=idx, name=name,
                                tokens=tokens, velocities=decoded))
    return arms
