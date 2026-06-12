"""E1-safety + E4-对照: monitor 各变体与 QP 投影 fallback。

  ucb1      全三 suite (id_clean 用于验证安全层不伤害分布内性能)
  thompson  两个障碍 suite
  random    ood_obstacle (bandit 价值下界)
  fixed     ood_obstacle (无上下文适配对照, retreat_up)
  qp        两个障碍 suite (外部 planner 路线对照)

用法: python scripts/04_run_safety.py [variant ...]   (缺省跑全部)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import vla_safety  # noqa: F401
from vla_safety.common import PROJECT_ROOT, load_yaml
from vla_safety.runtime.eval_runner import run_and_save
from vla_safety.vla.policy import VLAPolicy

PLAN = {
    "ucb1": ["id_clean", "ood_obstacle", "ood_obstacle_visual"],
    "thompson": ["ood_obstacle", "ood_obstacle_visual"],
    "random": ["ood_obstacle"],
    "fixed": ["ood_obstacle"],
    "qp": ["ood_obstacle", "ood_obstacle_visual"],
}


def main() -> None:
    variants = sys.argv[1:] or list(PLAN.keys())
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = VLAPolicy.load(PROJECT_ROOT / cfg["paths"]["weights"],
                            cfg["vla"], cfg["env"]["render_size"], device=device)
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    for v in variants:
        print(f"variant={v}")
        run_and_save(v, PLAN[v], cfg, policy, results_dir, device)


if __name__ == "__main__":
    main()
