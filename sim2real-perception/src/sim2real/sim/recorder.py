"""demo 采集: 在标称场景跑脚本化专家, 记录 (image, proprio, action, qpos)。"""
from __future__ import annotations

import numpy as np

from sim2real.common import SceneConfig, Trajectory
from sim2real.sim.env import ManipEnv
from sim2real.sim.expert import ScriptedExpert


def collect_demo(
    env: ManipEnv,
    expert: ScriptedExpert,
    scene: SceneConfig | None = None,
    max_steps: int = 160,
) -> Trajectory:
    obs = env.reset(scene)
    expert.reset()

    images, proprios, actions, qpos_hist = [], [], [], []
    success = False
    for _ in range(max_steps):
        a = expert.act(env)
        images.append(obs.image)
        proprios.append(obs.proprio)
        actions.append(a)
        qpos_hist.append(env.data.qpos.copy())
        obs, success, _ = env.step(a)
        if success and expert.phase == "LIFT":
            break

    return Trajectory(
        images=np.stack(images).astype(np.uint8),
        proprios=np.stack(proprios).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        qpos=np.stack(qpos_hist).astype(np.float32),
        success=bool(success),
    )
