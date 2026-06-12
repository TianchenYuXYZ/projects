"""跨模块共享的常量、数据类型与小工具。"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- 动作空间契约
# VLA 动作 = 末端速度指令 (世界系) + 夹爪, 全部归一化到 [-1, 1]:
#   a[0:3] = v_ee / V_MAX   (m/s)
#   a[3]   = 夹爪: > 0 闭合, <= 0 张开
# RT-2 风格: 每维离散化成 N_BINS 个 bin, 作为 token 自回归生成。
ACTION_DIM = 4
N_BINS = 256
V_MAX = 0.15              # 末端最大速度 m/s

# ---------------------------------------------------------------- 频率契约
CONTROL_HZ = 100          # 控制环频率 (ring buffer 消费端)
SAFETY_HZ = 33            # 深度安全检查频率 (GPU-B / stream-B)
CONTROL_DT = 1.0 / CONTROL_HZ


@dataclasses.dataclass
class Obs:
    """单步观测。image: HxWx3 uint8 RGB (第三人称相机); proprio: (4,) = ee_pos + 夹爪开度。"""

    image: np.ndarray
    proprio: np.ndarray


@dataclasses.dataclass
class SceneSpec:
    """一个评测/训练场景的完整描述。

    obstacle_* 描述真实参与碰撞的立柱障碍 (None = 无障碍, 训练分布)。
    视觉域偏移 (灯光/颜色/干扰物) 用于构造 OOD-visual 轴。
    """

    cube_pos: tuple = (0.50, 0.0)              # 桌面坐标 (x, y)
    obstacle_pos: tuple | None = None          # (x, y) 桌面坐标; None = 无障碍
    obstacle_half: tuple = (0.02, 0.05, 0.14)  # 立柱半尺寸 (x, y, z)
    cube_rgba: tuple = (0.85, 0.10, 0.10, 1.0)
    table_rgba: tuple = (0.78, 0.65, 0.50, 1.0)
    floor_rgba: tuple = (0.45, 0.45, 0.45, 1.0)
    obstacle_rgba: tuple = (0.55, 0.55, 0.60, 1.0)
    light_pos: tuple = (0.6, -0.6, 1.4)
    light_diffuse: float = 0.8
    distractors: list = dataclasses.field(default_factory=list)
    # 每项: {"type": box|sphere|cylinder, "pos": (x,y), "size": s, "rgba": (...)}

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def nominal() -> "SceneSpec":
        return SceneSpec()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"无法序列化 {type(o)}")


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def percentiles(xs: list[float] | np.ndarray, ps=(50, 95, 99)) -> dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps} | {"mean": float("nan"), "n": 0}
    out = {f"p{p}": float(np.percentile(arr, p)) for p in ps}
    out["mean"] = float(arr.mean())
    out["n"] = int(arr.size)
    return out
