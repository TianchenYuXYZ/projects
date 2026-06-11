"""跨模块共享的数据类型与小工具。"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass
class Obs:
    """单步观测。image: HxWx3 uint8 RGB; proprio: (8,) = 7 关节角 + 夹爪开度。"""

    image: np.ndarray
    proprio: np.ndarray


@dataclasses.dataclass
class Trajectory:
    """一条轨迹。actions[t] 是在 obs[t] 下执行的动作; qpos[t] 用于增广时的运动学回放。"""

    images: np.ndarray        # (T, H, W, 3) uint8
    proprios: np.ndarray      # (T, 8) float32
    actions: np.ndarray       # (T, 7) float32, 已归一化到 [-1, 1]
    qpos: np.ndarray          # (T, nq) float32
    success: bool

    def __len__(self) -> int:
        return len(self.actions)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, images=self.images, proprios=self.proprios,
            actions=self.actions, qpos=self.qpos, success=self.success,
        )

    @staticmethod
    def load(path: Path) -> "Trajectory":
        z = np.load(path)
        return Trajectory(
            images=z["images"], proprios=z["proprios"],
            actions=z["actions"], qpos=z["qpos"], success=bool(z["success"]),
        )


@dataclasses.dataclass
class SceneConfig:
    """一个视觉场景的完整描述, 由 DomainRandomizer 采样、scene.build_xml 消费。

    所有字段都有标称值 (nominal), demo 采集与 baseline 评测用标称场景。
    """

    table_texture: str | None = None          # 纹理 PNG 路径, None = 纯色
    floor_texture: str | None = None
    table_rgba: tuple = (0.78, 0.65, 0.50, 1.0)
    floor_rgba: tuple = (0.45, 0.45, 0.45, 1.0)
    cube_rgba: tuple = (0.85, 0.10, 0.10, 1.0)
    cube_pos: tuple = (0.50, 0.0)              # 桌面坐标 (x, y), 世界系
    lights: list = dataclasses.field(default_factory=lambda: [
        {"pos": (0.6, -0.6, 1.4), "diffuse": 0.8, "ambient": 0.2},
    ])
    camera_pos: tuple = (1.05, -0.45, 0.55)
    camera_lookat: tuple = (0.45, 0.0, 0.08)
    camera_fovy: float = 50.0
    distractors: list = dataclasses.field(default_factory=list)
    # 每项: {"type": box|sphere|cylinder, "pos": (x,y), "size": s, "rgba": (...)}

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def nominal() -> "SceneConfig":
        return SceneConfig()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
