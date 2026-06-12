"""评测执行器: variant x suite 的批量 episode 运行与聚合。

variant 定义:
  baseline   VLA 单独运行 (无安全层)
  ucb1       monitor + UCB1 上下文 bandit
  thompson   monitor + Thompson Sampling
  random     monitor + 均匀随机选 arm   (bandit 价值的下界对照)
  fixed      monitor + 固定 retreat_up  (无上下文适配的对照)
  qp         CBF 风格解析投影 fallback  (外部 planner 路线的对照)

同一 suite 下各 variant 共用同一场景 seed 序列 (seed_base + i),
保证逐集可比。bandit 状态在 suite 内跨集在线累计 (training-free 在线学习)。
"""
from __future__ import annotations

import time

import numpy as np
import torch

from vla_safety.common import save_json
from vla_safety.env import ManipSafetyEnv, SceneSampler
from vla_safety.perception.depth_safety import DepthSafetyChecker
from vla_safety.runtime.episode import run_episode
from vla_safety.safety.monitor import OracleMonitor
from vla_safety.baselines.qp_fallback import QPProjectionFallback
from vla_safety.vla.policy import VLAPolicy
from vla_safety.vla.tokenizer import ActionTokenizer

MONITOR_VARIANTS = {"ucb1", "thompson", "random", "fixed"}


def build_variant(variant: str, cfg: dict, fovy: float, device: str):
    """返回 (monitor, qp, qp_checker)。"""
    if variant in MONITOR_VARIANTS:
        bandit_cfg = dict(cfg["bandit"])
        bandit_cfg["algo"] = variant
        monitor = OracleMonitor(
            cfg["safety"], cfg["recovery"], bandit_cfg, ActionTokenizer(),
            depth_size=int(cfg["env"]["depth_size"]), fovy_deg=fovy,
            device=device,
        )
        return monitor, None, None
    if variant == "qp":
        checker = DepthSafetyChecker(cfg["safety"], int(cfg["env"]["depth_size"]),
                                     fovy, device=device)
        return None, QPProjectionFallback(), checker
    if variant == "baseline":
        return None, None, None
    raise ValueError(f"未知 variant: {variant}")


def run_suite(variant: str, suite: str, cfg: dict, policy: VLAPolicy,
              n_episodes: int, seed_base: int, device: str,
              record_traj_first: int = 3, verbose: bool = True) -> dict:
    sampler = SceneSampler(cube_x=tuple(cfg["demos"]["cube_x"]),
                           cube_y=tuple(cfg["demos"]["cube_y"]))
    env = ManipSafetyEnv(render_size=int(cfg["env"]["render_size"]),
                         depth_size=int(cfg["env"]["depth_size"]),
                         workspace=cfg["env"]["workspace"])
    env.reset()
    monitor, qp, qp_checker = build_variant(variant, cfg, env.depth_fovy, device)

    episodes = []
    t0 = time.time()
    for i in range(n_episodes):
        rng = np.random.default_rng(seed_base + i)
        spec = sampler.sample_suite(suite, rng)
        env.reset(spec)
        res = run_episode(env, policy, cfg, monitor=monitor, qp=qp,
                          qp_checker=qp_checker,
                          record_traj=(i < record_traj_first))
        rec = res.to_dict()
        rec["episode"] = i
        rec["scene"] = {"cube_pos": spec.cube_pos,
                        "obstacle_pos": spec.obstacle_pos}
        episodes.append(rec)
        if verbose and (i + 1) % 10 == 0:
            vfs = sum(e["violation_free_success"] for e in episodes)
            print(f"    [{variant}/{suite}] {i + 1}/{n_episodes} "
                  f"violation-free {vfs}/{i + 1} ({time.time() - t0:.0f}s)")
    env.close()

    agg = aggregate(episodes)
    out = {"variant": variant, "suite": suite, "n_episodes": n_episodes,
           "seed_base": seed_base, "aggregate": agg, "episodes": episodes,
           "wall_s": round(time.time() - t0, 1)}
    if monitor is not None:
        out["bandit"] = monitor.bandit.snapshot()
    return out


def aggregate(episodes: list[dict]) -> dict:
    n = len(episodes)
    vio_eps = [e for e in episodes if e["violation_ticks"] > 0]
    trig = [e["n_triggers"] for e in episodes]
    rewards = [t["reward"] for e in episodes for t in e["triggers"]]
    return {
        "success_rate": sum(e["success"] for e in episodes) / n,
        "violation_free_success_rate":
            sum(e["violation_free_success"] for e in episodes) / n,
        "violation_episode_rate": len(vio_eps) / n,
        "mean_violation_ticks": float(np.mean([e["violation_ticks"]
                                               for e in episodes])),
        "mean_triggers": float(np.mean(trig)),
        "n_recovery_total": int(np.sum(trig)),
        "recovery_reward_rate": (float(np.mean(rewards)) if rewards else None),
        "mean_qp_interventions": float(np.mean([e["qp_interventions"]
                                                for e in episodes])),
        "mean_decisions": float(np.mean([e["decisions"] for e in episodes])),
    }


def run_and_save(variant: str, suites: list[str], cfg: dict, policy: VLAPolicy,
                 results_dir, device: str) -> dict:
    n = int(cfg["eval"]["n_episodes"])
    seed_base = int(cfg["eval"]["seed_base"])
    all_out = {}
    for suite in suites:
        print(f"  suite={suite}")
        out = run_suite(variant, suite, cfg, policy, n, seed_base, device)
        all_out[suite] = out
        a = out["aggregate"]
        print(f"  => success {a['success_rate']:.1%}, "
              f"violation-free {a['violation_free_success_rate']:.1%}, "
              f"violation eps {a['violation_episode_rate']:.1%}")
    save_json(all_out, results_dir / f"eval_{variant}.json")
    return all_out
