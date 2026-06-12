"""数据集构建: 脚本专家在 N 个随机化场景闭环执行, 产出增广训练集。

与纯回放渲染的区别 (闭环 0% 教训):
  * 方块初始位置随机 -> 状态空间覆盖, BC 才有 funnel 可学
  * DART 噪声注入: 执行 a_label + noise, 标签存干净的 a_label,
    数据天然包含 "偏离 -> 修正" 的纠错样本
  * 视觉 DR (纹理/光照/相机/干扰物) 同时生效, 保留 perception 故事

渲染优化: rollout 时只记 qpos/proprio/action (无渲染), 成功后在同一场景
对抽样帧做运动学回放渲染 —— 每场景只渲染 frames_per_scene 帧。
专家失败的场景重采样 (最多 3 次), 仍失败则跳过。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sim2real.common import SceneConfig
from sim2real.datagen.randomizer import DomainRandomizer
from sim2real.sim.env import ManipEnv
from sim2real.sim.expert import ScriptedExpert

SHARD_SIZE = 64  # 每个 npz 分片的场景数
MAX_EPISODE_STEPS = 160
MAX_SCENE_RETRIES = 3


def _rollout_expert(env: ManipEnv, expert: ScriptedExpert,
                    rng: np.random.Generator, noise_std: float,
                    noise_phases: set[str]) -> dict | None:
    """单场景闭环 rollout。返回 None 表示专家失败。"""
    env.reset()  # 场景已在调用方 _load, 此处仅复位状态
    expert.reset()
    proprios, labels, qpos_hist = [], [], []
    success = False
    for _ in range(MAX_EPISODE_STEPS):
        a_label = expert.act(env)
        proprios.append(env.proprio())
        labels.append(a_label.copy())
        qpos_hist.append(env.data.qpos.copy())

        a_exec = a_label.copy()
        if noise_std > 0 and expert.phase in noise_phases:
            a_exec[:3] = np.clip(
                a_exec[:3] + rng.normal(0, noise_std, 3), -1, 1)
        _, success, _ = env.step(a_exec)
        if success and expert.phase == "LIFT":
            break
    if not success:
        return None
    return {
        "proprios": np.stack(proprios).astype(np.float32),
        "actions": np.stack(labels).astype(np.float32),
        "qpos": np.stack(qpos_hist).astype(np.float32),
    }


def build_dataset(
    randomizer: DomainRandomizer,
    out_dir: Path,
    n_scenes: int,
    frames_per_scene: int,
    seed: int,
    noise_std: float = 0.15,
    noise_phases: tuple[str, ...] = ("HOVER", "DESCEND"),
    render_size: int = 224,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    phases = set(noise_phases)

    env = ManipEnv(SceneConfig.nominal(), render_size=render_size)
    expert = ScriptedExpert()
    shard_imgs, shard_props, shard_acts, shard_scene_ids = [], [], [], []
    scenes_meta = []
    shard_idx = 0
    n_expert_fail = 0

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
        traj = None
        for _ in range(MAX_SCENE_RETRIES):
            scene = randomizer.sample_scene(rng)
            env.reset(scene)
            traj = _rollout_expert(env, expert, rng, noise_std, phases)
            if traj is not None:
                break
            n_expert_fail += 1
        if traj is None:
            continue

        T = len(traj["actions"])
        indices = np.linspace(0, T - 1, min(frames_per_scene, T)).round().astype(int)
        frames = env.replay_render(traj["qpos"], indices)

        shard_imgs.append(frames)
        shard_props.append(traj["proprios"][indices])
        shard_acts.append(traj["actions"][indices])
        shard_scene_ids.append(np.full(len(indices), sid, dtype=np.int32))
        scenes_meta.append(json.loads(scene.to_json()))

        if (sid + 1) % SHARD_SIZE == 0:
            flush()
    flush()
    env.close()

    meta = {
        "n_scenes_requested": n_scenes,
        "n_scenes_built": len(scenes_meta),
        "n_expert_failures": n_expert_fail,
        "frames_per_scene": frames_per_scene,
        "noise_std": noise_std,
        "noise_phases": list(noise_phases),
        "scenes": scenes_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_shards(data_dir: Path) -> dict[str, np.ndarray]:
    """合并加载所有分片 (默认配置 ~24k 帧 224^2 uint8 ≈ 3.6GB, 可整载)。"""
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
