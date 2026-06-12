"""DR 增广: 单条 demo -> N 个随机化场景的回放渲染数据集。

CLIP 引导 (可选): 先用 CLIP 按 surface prompt 筛纹理池, 再交给 randomizer。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim2real.common import PROJECT_ROOT, load_yaml
from sim2real.datagen.builder import build_dataset
from sim2real.datagen.randomizer import DomainRandomizer, build_texture_pool


def main() -> None:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "dr.yaml")

    clip_guide = None
    if cfg.get("clip_guide", {}).get("enabled", False):
        from sim2real.datagen.clip_guide import CLIPGuide

        gcfg = cfg["clip_guide"]
        print("[clip] 加载 CLIP 并筛选纹理池 ...")
        clip_guide = CLIPGuide(gcfg["model"], gcfg["pretrained"])
    pool = build_texture_pool(cfg, clip_guide)
    for surf, texs in pool.items():
        print(f"[pool] {surf}: {len(texs)} textures")

    randomizer = DomainRandomizer(cfg, texture_pool=pool)
    ncfg = cfg.get("expert_noise", {})
    out_dir = PROJECT_ROOT / "data" / "dr_dataset"
    build_dataset(
        randomizer, out_dir,
        n_scenes=int(cfg["n_train_scenes"]),
        frames_per_scene=int(cfg["frames_per_scene"]),
        seed=int(cfg["seed"]),
        noise_std=float(ncfg.get("std", 0.0)),
        noise_phases=tuple(ncfg.get("phases", [])),
    )
    print(f"dataset -> {out_dir}")


if __name__ == "__main__":
    main()
