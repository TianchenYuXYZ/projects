"""数据集构建: 把单条 demo 回放到 N 个随机化场景, 产出增广训练集。

核心技巧: 视觉 DR 不影响物理 -> 直接按 demo 记录的 qpos 序列做运动学
回放渲染, (proprio, action) 标签天然保持有效。每个场景沿轨迹均匀抽
frames_per_scene 帧, 存成 npz 分片。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sim2real.common import SceneConfig, Trajectory
from sim2real.datagen.randomizer import DomainRandomizer
from sim2real.sim.env import ManipEnv

SHARD_SIZE = 64  # 每个 npz 分片的场景数


def build_dataset(
    demo: Trajectory,
    randomizer: DomainRandomizer,
    out_dir: Path,
    n_scenes: int,
    frames_per_scene: int,
    seed: int,
    render_size: int = 224,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    T = len(demo)
    indices = np.linspace(0, T - 1, frames_per_scene).round().astype(int)

    env = ManipEnv(SceneConfig.nominal(), render_size=render_size)
    shard_imgs, shard_props, shard_acts, shard_scene_ids = [], [], [], []
    scenes_meta = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx, shard_imgs, shard_props, shard_acts, shard_scene_ids
        if not shard_imgs:
            return
        np.savez_compressed(
            out_dir / f"shard_{shard_idx:04d}.npz",
            images=np.concatenate(shard_imgs),
            proprios=np.concatenate(shard_props),
            actions=np.concatenate(shard_acts),
            scene_ids=np.concatenate(shard_scene_ids),
        )
        shard_idx += 1
        shard_imgs, shard_props, shard_acts, shard_scene_ids = [], [], [], []

    for sid in tqdm(range(n_scenes), desc="DR scenes"):
        scene = randomizer.sample_scene(rng)
        env.reset(scene)
        frames = env.replay_render(demo.qpos, indices)

        shard_imgs.append(frames)
        shard_props.append(demo.proprios[indices])
        shard_acts.append(demo.actions[indices])
        shard_scene_ids.append(np.full(len(indices), sid, dtype=np.int32))
        scenes_meta.append(json.loads(scene.to_json()))

        if (sid + 1) % SHARD_SIZE == 0:
            flush()
    flush()
    env.close()

    meta = {
        "n_scenes": n_scenes,
        "frames_per_scene": frames_per_scene,
        "frame_indices": indices.tolist(),
        "demo_len": T,
        "scenes": scenes_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_shards(data_dir: Path) -> dict[str, np.ndarray]:
    """合并加载所有分片 (数据量在内存可承受范围: ~70k 帧 224^2 uint8 ≈ 10GB 之内需注意,
    默认配置 1500x16=24k 帧 ≈ 3.6GB, 可整载)。"""
    shards = sorted(data_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"{data_dir} 下没有 shard_*.npz")
    imgs, props, acts, sids = [], [], [], []
    for s in shards:
        z = np.load(s)
        imgs.append(z["images"])
        props.append(z["proprios"])
        acts.append(z["actions"])
        sids.append(z["scene_ids"])
    return {
        "images": np.concatenate(imgs),
        "proprios": np.concatenate(props),
        "actions": np.concatenate(acts),
        "scene_ids": np.concatenate(sids),
    }
