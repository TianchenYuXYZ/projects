"""DomainRandomizer: 从 DR 配置采样 SceneConfig。

干扰物位置用拒绝采样, 保证离目标方块和末端运动路径足够远 ——
配合纯视觉 (无碰撞) 干扰物, 回放的 demo 轨迹物理上严格不变。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sim2real.common import PROJECT_ROOT, SceneConfig
from sim2real.sim import scene as scene_mod


class DomainRandomizer:
    def __init__(self, cfg: dict, texture_pool: dict[str, list[Path]] | None = None):
        """texture_pool: surface 名 -> 可用纹理路径列表 (已经过 CLIP 筛选)。"""
        self.cfg = cfg
        self.texture_pool = texture_pool or {}

    def sample_scene(self, rng: np.random.Generator,
                     enable: dict | None = None) -> SceneConfig:
        """enable: 各维度开关覆盖 (评测套件用), None = 全按配置。"""
        c = self.cfg
        en = enable or {}
        scene = SceneConfig.nominal()

        def on(dim: str) -> bool:
            if dim in en:
                return bool(en[dim])
            return bool(c.get(dim, {}).get("enabled", False))

        if on("texture"):
            tcfg = c["texture"]
            lo, hi = tcfg["tint_range"]
            for surf in tcfg["surfaces"]:
                pool = self.texture_pool.get(surf, [])
                if pool:
                    tex = str(pool[rng.integers(len(pool))])
                    setattr(scene, f"{surf}_texture", tex)
                tint = tuple(np.clip(rng.uniform(lo, hi, 3), 0, 2)) + (1.0,)
                setattr(scene, f"{surf}_rgba", tint)

        jit = float(c.get("cube", {}).get("rgba_jitter", 0.0))
        if jit > 0:
            base = np.array(scene.cube_rgba[:3])
            scene.cube_rgba = tuple(np.clip(base + rng.uniform(-jit, jit, 3), 0, 1)) + (1.0,)

        if on("light"):
            lcfg = c["light"]
            n = int(rng.integers(lcfg["n_lights"][0], lcfg["n_lights"][1] + 1))
            scene.lights = [
                {
                    "pos": (
                        float(rng.uniform(*lcfg["pos_x"]) + scene_mod.TABLE_CENTER[0]),
                        float(rng.uniform(*lcfg["pos_y"])),
                        float(rng.uniform(*lcfg["pos_z"])),
                    ),
                    "diffuse": float(rng.uniform(*lcfg["diffuse"])),
                    "ambient": float(rng.uniform(*lcfg["ambient"])),
                }
                for _ in range(n)
            ]

        if on("camera"):
            ccfg = c["camera"]
            pos = np.array(scene.camera_pos) + rng.uniform(
                -ccfg["pos_jitter"], ccfg["pos_jitter"], 3)
            lookat = np.array(scene.camera_lookat) + rng.uniform(
                -ccfg["lookat_jitter"], ccfg["lookat_jitter"], 3)
            scene.camera_pos = tuple(pos)
            scene.camera_lookat = tuple(lookat)
            scene.camera_fovy = float(
                scene.camera_fovy + rng.uniform(-ccfg["fovy_jitter"], ccfg["fovy_jitter"]))

        if on("distractor"):
            scene.distractors = self._sample_distractors(rng, scene)

        return scene

    def _sample_distractors(self, rng: np.random.Generator,
                            scene: SceneConfig) -> list[dict]:
        dcfg = self.cfg["distractor"]
        n = int(rng.integers(dcfg["n_range"][0], dcfg["n_range"][1] + 1))
        cube_xy = np.array(scene.cube_pos)
        # 末端路径在俯视图上近似为 基座->方块 的线段
        path_a = np.array([0.0, 0.0])
        path_b = cube_xy
        cx, cy = scene_mod.TABLE_CENTER

        out: list[dict] = []
        tries = 0
        while len(out) < n and tries < 200:
            tries += 1
            xy = np.array([
                rng.uniform(*dcfg["region_x"]) + cx,
                rng.uniform(*dcfg["region_y"]) + cy,
            ])
            if np.linalg.norm(xy - cube_xy) < dcfg["min_dist_to_cube"]:
                continue
            if _dist_to_segment(xy, path_a, path_b) < dcfg["min_dist_to_path"]:
                continue
            out.append({
                "type": str(rng.choice(dcfg["types"])),
                "pos": (float(xy[0]), float(xy[1])),
                "size": float(rng.uniform(*dcfg["size_range"])),
                "rgba": tuple(rng.uniform(0.1, 0.95, 3)) + (1.0,),
            })
        return out


def _dist_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def build_texture_pool(cfg: dict, clip_guide=None) -> dict[str, list[Path]]:
    """加载纹理库并 (可选) 用 CLIP 按 surface prompt 筛选。"""
    tex_dir = PROJECT_ROOT / cfg["texture"]["library_dir"]
    all_tex = sorted(tex_dir.glob("*.png"))
    pool: dict[str, list[Path]] = {}
    gcfg = cfg.get("clip_guide", {})
    for surf in cfg["texture"]["surfaces"]:
        if clip_guide is not None and gcfg.get("enabled", False):
            prompt = gcfg["prompts"][surf]
            pool[surf] = clip_guide.curate(all_tex, prompt, gcfg["keep_ratio"])
        else:
            pool[surf] = list(all_tex)
    return pool
