"""训练: --variant ours (DR+过滤+LoRA) | baseline (单 demo, frozen, 无 LoRA)。

baseline 对应文档 3.4 的 67% 故事: "R3M frozen + 直接在单条 demo 上 BC"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sim2real.common import PROJECT_ROOT, Trajectory, load_yaml
from sim2real.datagen.builder import load_shards
from sim2real.policy.dataset import BCDataset
from sim2real.policy.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["ours", "baseline"], default="ours")
    ap.add_argument("--rank", type=int, default=None, help="覆盖 LoRA rank (扫描实验用)")
    args = ap.parse_args()

    cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")
    if args.variant == "baseline":
        cfg["lora"]["enabled"] = False
        if "lr" in cfg.get("baseline", {}):
            cfg["optim"]["lr"] = cfg["baseline"]["lr"]
    if args.rank is not None:
        cfg["lora"]["rank"] = args.rank

    if args.variant == "ours":
        data = load_shards(PROJECT_ROOT / "data" / "dr_dataset")
        keep = np.load(PROJECT_ROOT / "data" / "filter" / "keep_mask.npy")
        ds = BCDataset(data["images"][keep], data["proprios"][keep],
                       data["actions"][keep])
        epochs = None
    else:
        demo = Trajectory.load(PROJECT_ROOT / "data" / "demo.npz")
        ds = BCDataset(demo.images, demo.proprios, demo.actions)
        epochs = int(cfg.get("baseline", {}).get("epochs", 200))

    trainer = Trainer(cfg)
    backbone, policy, lora_params = trainer.build()
    suffix = f"_r{cfg['lora']['rank']}" if (args.variant == "ours" and args.rank) else ""
    out_dir = PROJECT_ROOT / "runs" / (args.variant + suffix)
    stats = trainer.train(ds, backbone, policy, lora_params, out_dir, epochs=epochs)
    print(f"[{args.variant}] n={stats['n_samples']}, "
          f"trainable={stats['n_trainable']:,}, "
          f"final_loss={stats['final_loss']:.5f}, wall={stats['wall_sec']:.0f}s")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
