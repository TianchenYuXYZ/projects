"""zero-shot 闭环评测: nominal + 三档 unseen 套件, baseline vs ours。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from sim2real.common import PROJECT_ROOT, load_yaml
from sim2real.datagen.randomizer import DomainRandomizer
from sim2real.eval.runner import PolicyAgent, run_suite
from sim2real.eval.suites import make_suite_scenes
from sim2real.perception.lora import inject_lora
from sim2real.policy.trainer import Trainer


def load_agent(variant: str, run_dir: Path, cfg: dict) -> PolicyAgent:
    cfg = dict(cfg)
    if variant.startswith("baseline"):
        cfg["lora"] = dict(cfg["lora"], enabled=False)
    trainer = Trainer(cfg)
    backbone, policy, _ = trainer.build()
    if (run_dir / "lora.pth").exists():
        if not any("lora" in n for n, _ in backbone.named_parameters()):
            inject_lora(backbone, int(cfg["lora"]["rank"]),
                        float(cfg["lora"]["alpha"]),
                        list(cfg["lora"]["target_stages"]))
        backbone.load_state_dict(
            torch.load(run_dir / "lora.pth", map_location="cpu",
                       weights_only=True), strict=False)
    policy.load_state_dict(
        torch.load(run_dir / "policy.pth", map_location="cpu", weights_only=True))
    return PolicyAgent(backbone, policy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["baseline", "ours"])
    ap.add_argument("--episodes", type=int, default=None)
    args = ap.parse_args()

    tcfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")
    dcfg = load_yaml(PROJECT_ROOT / "configs" / "dr.yaml")
    ecfg = load_yaml(PROJECT_ROOT / "configs" / "eval.yaml")
    n_ep = args.episodes or int(ecfg["episodes_per_suite"])
    max_steps = int(ecfg["max_steps"])

    results: dict = {}
    for variant in args.variants:
        run_dir = PROJECT_ROOT / "runs" / variant
        agent = load_agent(variant, run_dir, tcfg)
        results[variant] = {}

        # nominal 套件: 视觉标称 + 方块位置抖动 (与训练同分布的状态随机,
        # 全部视觉维度关闭) —— 单点确定性场景只会得到 0% 或 100%
        nom_rand = DomainRandomizer(dcfg, texture_pool={})
        nom_rng = np.random.default_rng(int(ecfg["seed"]))
        all_off = {k: False for k in ("texture", "light", "camera", "distractor")}
        nominal = [nom_rand.sample_scene(nom_rng, enable=all_off)
                   for _ in range(min(n_ep, 20))]
        r = run_suite(agent, nominal, max_steps, desc=f"{variant}/nominal")
        results[variant]["nominal"] = r
        print(f"[{variant}] nominal: {r['success_rate']:.2%}")

        for suite_name in ecfg["suites"]:
            scenes = make_suite_scenes(dcfg, ecfg, suite_name, n_ep)
            r = run_suite(agent, scenes, max_steps,
                          desc=f"{variant}/{suite_name}")
            results[variant][suite_name] = r
            print(f"[{variant}] {suite_name}: {r['success_rate']:.2%}")

    out = PROJECT_ROOT / "runs" / "eval_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
