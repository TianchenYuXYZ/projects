"""专家示范采集: 训练分布 (无障碍 + 轻度视觉随机化)。

两组示范:
  nominal    标准起点 (ready 位姿)
  perturbed  随机漂移序幕后再执行专家 —— 覆盖 recovery 之后 VLA 需要
             从离轨位姿重新接管的状态分布 (横移/抬升后的中途位姿)

记录频率 = 专家决策频率 (10Hz); 只保留成功示范。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import vla_safety  # noqa: F401
from vla_safety.common import CONTROL_HZ, PROJECT_ROOT, load_yaml, save_json
from vla_safety.env import ManipSafetyEnv, SceneSampler
from vla_safety.env.expert import ScriptedExpert


def collect_demo(env: ManipSafetyEnv, expert: ScriptedExpert,
                 ticks_per_decision: int, max_decisions: int,
                 prologue_rng: np.random.Generator | None) -> dict | None:
    """跑一条示范; 失败返回 None。"""
    env.reset()
    expert.reset()

    if prologue_rng is not None:
        # 漂移序幕: 随机方向速度指令, 把 EE 带离标称起点 (不记录)
        d = prologue_rng.normal(size=3)
        d[2] *= 0.5
        d /= np.linalg.norm(d)
        mag = prologue_rng.uniform(0.5, 1.0)
        n_ticks = int(prologue_rng.integers(30, 80))
        env.set_command(np.array([d[0] * mag, d[1] * mag, d[2] * mag, -1.0]))
        for _ in range(n_ticks):
            env.tick()

    images, proprios, actions = [], [], []
    dt = ticks_per_decision / CONTROL_HZ
    success = False
    for _ in range(max_decisions):
        img = env.render_rgb()
        prop = env.proprio()
        cmd = expert.act(env.ee_pos, env.cube_pos, dt)
        images.append(img)
        proprios.append(prop)
        actions.append(cmd.astype(np.float32))
        env.set_command(cmd)
        for _ in range(ticks_per_decision):
            env.tick()
        if env.is_success():
            success = True
            break
    if not success:
        return None
    return {
        "images": np.stack(images),
        "proprios": np.stack(proprios),
        "actions": np.stack(actions).astype(np.float32),
        "success": True,
    }


def main() -> None:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    dcfg = cfg["demos"]
    out_dir = PROJECT_ROOT / cfg["paths"]["demos_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = SceneSampler(cube_x=tuple(dcfg["cube_x"]), cube_y=tuple(dcfg["cube_y"]))
    ticks_per_decision = CONTROL_HZ // int(dcfg["decision_hz"])
    max_decisions = int(cfg["env"]["episode_ticks"] / ticks_per_decision)

    env = ManipSafetyEnv(render_size=cfg["env"]["render_size"],
                         depth_size=cfg["env"]["depth_size"],
                         workspace=cfg["env"]["workspace"])
    groups = [("nominal", int(dcfg["n_nominal"]), False),
              ("perturbed", int(dcfg["n_perturbed"]), True)]
    t0 = time.time()
    saved, attempts = 0, 0
    stats = {g: {"ok": 0, "fail": 0} for g, _, _ in groups}
    for group, n_target, perturbed in groups:
        got = 0
        while got < n_target:
            seed = 10_000 + attempts
            rng = np.random.default_rng(seed)
            attempts += 1
            spec = sampler.sample_train(rng)
            env.reset(spec)
            expert = ScriptedExpert(
                travel_z=float(cfg["env"]["travel_ee_z"]),
                grasp_z=float(cfg["env"]["grasp_ee_z"]),
                noise=float(dcfg["noise_v"]), rng=rng,
            )
            demo = collect_demo(env, expert, ticks_per_decision, max_decisions,
                                prologue_rng=rng if perturbed else None)
            if demo is None:
                stats[group]["fail"] += 1
                continue
            np.savez_compressed(out_dir / f"demo_{saved:04d}.npz", **demo)
            stats[group]["ok"] += 1
            got += 1
            saved += 1
            if saved % 40 == 0:
                print(f"  {saved} 条已保存 ({time.time() - t0:.0f}s), "
                      f"当前组 {group} {got}/{n_target}")
    env.close()

    lengths = [len(np.load(f)["actions"]) for f in sorted(out_dir.glob("demo_*.npz"))]
    summary = {
        "n_demos": saved,
        "attempts": attempts,
        "groups": stats,
        "mean_len": float(np.mean(lengths)),
        "total_frames": int(np.sum(lengths)),
        "wall_s": round(time.time() - t0, 1),
    }
    save_json(summary, out_dir / "demo_stats.json")
    print(f"完成: {saved} 条示范, {summary['total_frames']} 帧, "
          f"均长 {summary['mean_len']:.1f}, 耗时 {summary['wall_s']}s")
    print(f"组统计: {stats}")


if __name__ == "__main__":
    main()
