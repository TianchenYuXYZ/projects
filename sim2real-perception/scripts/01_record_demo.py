"""在标称场景采集单条专家 demo -> data/demo.npz (+ 预览帧)。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from sim2real.common import PROJECT_ROOT, SceneConfig
from sim2real.sim.env import ManipEnv
from sim2real.sim.expert import ScriptedExpert
from sim2real.sim.recorder import collect_demo


def main() -> None:
    env = ManipEnv(SceneConfig.nominal())
    expert = ScriptedExpert()
    traj = collect_demo(env, expert)
    env.close()

    print(f"demo: T={len(traj)}, success={traj.success}, "
          f"phase_end={expert.phase}")
    if not traj.success:
        raise SystemExit("专家 demo 未成功, 检查 expert 参数 (GRASP_HEIGHT 等)")

    out = PROJECT_ROOT / "data" / "demo.npz"
    traj.save(out)
    print(f"saved -> {out}")

    prev_dir = PROJECT_ROOT / "data" / "demo_preview"
    prev_dir.mkdir(parents=True, exist_ok=True)
    for t in np.linspace(0, len(traj) - 1, 6).round().astype(int):
        Image.fromarray(traj.images[t]).save(prev_dir / f"frame_{t:03d}.png")
    print(f"preview -> {prev_dir}")


if __name__ == "__main__":
    main()
