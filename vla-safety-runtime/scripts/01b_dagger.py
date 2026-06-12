"""DAgger 数据聚合 (Ross et al. 2011): 治 BC 闭环复合误差的标准药。

当前策略按 *部署节奏* (15 tick 决策 + 150ms 延迟) 闭环走访状态,
脚本专家在每个走访状态上重标注动作; 执行的是策略动作 (走访分布),
学习的是专家动作 (监督信号)。输出 dagger_*.npz, 02 重训时自动聚合。

用法: python scripts/01b_dagger.py [n_rollouts] [round_tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

import vla_safety  # noqa: F401
from vla_safety.common import CONTROL_HZ, PROJECT_ROOT, load_yaml, save_json
from vla_safety.env import ManipSafetyEnv, SceneSampler
from vla_safety.env.expert import ScriptedExpert
from vla_safety.vla.policy import VLAPolicy


def dagger_rollout(env, policy, expert, cfg, rng) -> dict:
    """策略驱动 + 专家标注。返回 (frames, 元信息)。"""
    think = int(cfg["vla"]["decision_period_ticks"])
    horizon = int(cfg["env"]["episode_ticks"])
    env.reset()
    expert.reset()
    images, wrist_images, proprios, actions = [], [], [], []
    pending = None
    success = False
    for t in range(horizon):
        if pending is not None and t >= pending["ready"]:
            env.set_command(pending["cmd"])
            pending = None
        if pending is None:
            img = env.render_rgb()
            wimg = env.render_wrist_rgb()
            prop = env.proprio()
            # 专家在该状态的标注 (监督信号)
            a_exp = expert.act(env.ee_pos, env.cube_pos, think / CONTROL_HZ)
            images.append(img)
            wrist_images.append(wimg)
            proprios.append(prop)
            actions.append(a_exp.astype(np.float32))
            # 执行的是 *策略* 动作 (走访策略自己会到达的状态分布)
            a_pi = policy.act(img, wimg, prop)
            pending = {"ready": t + think, "cmd": a_pi}
        env.tick()
        if env.is_success():
            success = True
            break
    return {
        "images": np.stack(images), "wrist_images": np.stack(wrist_images),
        "proprios": np.stack(proprios),
        "actions": np.stack(actions), "success": True,   # 标注帧全部有效
        "rollout_success": success,
    }


def main() -> None:
    n_rollouts = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    tag = sys.argv[2] if len(sys.argv) > 2 else "r1"
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = VLAPolicy.load(PROJECT_ROOT / cfg["paths"]["weights"],
                            cfg["vla"], cfg["env"]["render_size"], device=device)
    sampler = SceneSampler(cube_x=tuple(cfg["demos"]["cube_x"]),
                           cube_y=tuple(cfg["demos"]["cube_y"]))
    out_dir = PROJECT_ROOT / cfg["paths"]["demos_dir"]
    env = ManipSafetyEnv(render_size=cfg["env"]["render_size"],
                         depth_size=cfg["env"]["depth_size"],
                         workspace=cfg["env"]["workspace"])
    n_succ, frames = 0, 0
    for i in range(n_rollouts):
        rng = np.random.default_rng(50_000 + i)
        env.reset(sampler.sample_train(rng))
        expert = ScriptedExpert(travel_z=float(cfg["env"]["travel_ee_z"]),
                                grasp_z=float(cfg["env"]["grasp_ee_z"]),
                                noise=0.0, rng=rng)
        d = dagger_rollout(env, policy, expert, cfg, rng)
        n_succ += d.pop("rollout_success")
        frames += len(d["actions"])
        np.savez_compressed(out_dir / f"dagger_{tag}_{i:04d}.npz", **d)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_rollouts} (策略 rollout 成功率 "
                  f"{n_succ / (i + 1):.0%}, 累计 {frames} 帧)")
    env.close()
    save_json({"round": tag, "n_rollouts": n_rollouts,
               "policy_success_during_collection": n_succ / n_rollouts,
               "frames": frames}, out_dir / f"dagger_{tag}_stats.json")
    print(f"DAgger {tag}: {n_rollouts} 条, {frames} 帧, "
          f"采集时策略成功率 {n_succ / n_rollouts:.0%}")


if __name__ == "__main__":
    main()
