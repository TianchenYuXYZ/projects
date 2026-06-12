"""快速闭环 sanity: id_clean 上跑 N 集 baseline, 打印逐集结果。

用法: python scripts/99_sanity_rollout.py [n_episodes]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import vla_safety  # noqa: F401
from vla_safety.common import PROJECT_ROOT, load_yaml
from vla_safety.vla.policy import VLAPolicy
from vla_safety.runtime.eval_runner import run_suite


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = VLAPolicy.load(PROJECT_ROOT / cfg["paths"]["weights"], cfg["vla"],
                            cfg["env"]["render_size"], device=device)
    out = run_suite("baseline", "id_clean", cfg, policy, n_episodes=n,
                    seed_base=99000, device=device, record_traj_first=0,
                    verbose=False)
    print(f"ID sanity ({n}集): success {out['aggregate']['success_rate']:.0%}")
    for e in out["episodes"]:
        print(f"  ep{e['episode']:02d} {'OK  ' if e['success'] else 'FAIL'} "
              f"ticks={e['ticks_run']:4d} final_dist={e['final_dist_goal']:.3f}")


if __name__ == "__main__":
    main()
