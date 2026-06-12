"""CBF 风格几何投影 fallback (对照组, 对应文档里的 VLSA / CBF-QP 路线)。

把 nominal 速度指令投影到 "远离冲突质心" 的半空间:
    v' = v - max(0, v·n) n - beta * n,   n = (centroid - ee) / |centroid - ee|

闭式解 (无 QP 求解器), 是 CBF-QP 单约束情形的解析特例。求解延迟为零,
所以这条对照不比延迟 —— 它对照的是文档强调的另一件事:
safety-induced distribution shift。投影产生的是连续的几何修正轨迹,
不在 VLA 的 action token 分布里; recovery arm 则是离散 token 序列,
执行完把控制权干净地交还 VLA。
"""
from __future__ import annotations

import numpy as np

from vla_safety.perception.depth_safety import ConflictReport


class QPProjectionFallback:
    def __init__(self, beta: float = 0.4):
        self.beta = beta            # 归一化速度单位的外推强度
        self.interventions = 0

    def project(self, cmd: np.ndarray, report: ConflictReport,
                ee: np.ndarray) -> np.ndarray:
        """cmd: (4,) 归一化指令; 仅修改速度分量, 夹爪透传。"""
        if not report.conflict or report.centroid is None:
            return cmd
        n = report.centroid - np.asarray(ee, dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            return cmd
        n = n / norm
        v = cmd[:3].astype(np.float64)
        v_proj = v - max(0.0, float(v @ n)) * n - self.beta * n
        out = cmd.copy()
        out[:3] = np.clip(v_proj, -1.0, 1.0)
        self.interventions += 1
        return out
