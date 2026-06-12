"""E1-baseline: VLA 单独运行, 三个 suite。

预期复现文档的失败模式: id_clean 上正常工作 (sanity), ood_obstacle
上 violation-free completion 断崖 (VLA 没有避障概念, 直线撞柱)。
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


def main() -> None:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = VLAPolicy.load(PROJECT_ROOT / cfg["paths"]["weights"],
                            cfg["vla"], cfg["env"]["render_size"], device=device)
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    print("variant=baseline")
    run_and_save("baseline", list(cfg["eval"]["suites"]), cfg, policy,
                 results_dir, device)


if __name__ == "__main__":
    main()
