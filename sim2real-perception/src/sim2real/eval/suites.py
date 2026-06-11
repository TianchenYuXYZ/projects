"""三档 unseen 评测套件: 用独立 seed + 独立测试纹理库生成测试场景。"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from sim2real.common import PROJECT_ROOT, SceneConfig
from sim2real.datagen.randomizer import DomainRandomizer


def make_suite_scenes(dr_cfg: dict, eval_cfg: dict, suite_name: str,
                      n_episodes: int) -> list[SceneConfig]:
    """每个 episode 一个独立场景; 纹理来自测试库 (与训练库不同源)。"""
    suite = eval_cfg["suites"][suite_name]
    test_dir = PROJECT_ROOT / eval_cfg["test_texture_dir"]
    test_pool = sorted(test_dir.glob("*.png"))
    pool = {s: list(test_pool) for s in dr_cfg["texture"]["surfaces"]}

    randomizer = DomainRandomizer(dr_cfg, texture_pool=pool)
    # zlib.crc32 稳定可复现 (内置 hash 每进程随机化)
    rng = np.random.default_rng(
        int(eval_cfg["seed"]) + zlib.crc32(suite_name.encode()) % 10000)
    return [randomizer.sample_scene(rng, enable=dict(suite)) for _ in range(n_episodes)]
