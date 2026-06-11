"""联合训练: frozen backbone (+LoRA) + BC 头。

只有 LoRA 矩阵和策略头参与梯度更新 —— 对应文档 3.3:
"R3M backbone 完全冻住, 只更新 low-rank adapter, 保住 generalization prior"。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sim2real.perception.backbone import PerceptionBackbone
from sim2real.perception.lora import inject_lora, lora_state_dict
from sim2real.policy.bc import BCPolicy
from sim2real.policy.dataset import BCDataset


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    def build(self) -> tuple[PerceptionBackbone, BCPolicy, list[nn.Parameter]]:
        bcfg = self.cfg["backbone"]
        backbone = PerceptionBackbone(bcfg["name"], bcfg.get("weights_dir", "weights"))
        backbone.freeze()

        lora_params: list[nn.Parameter] = []
        lcfg = self.cfg.get("lora", {})
        if lcfg.get("enabled", False):
            lora_params = inject_lora(
                backbone, rank=int(lcfg["rank"]), alpha=float(lcfg["alpha"]),
                target_stages=list(lcfg["target_stages"]),
            )

        pcfg = self.cfg["policy"]
        policy = BCPolicy(
            proprio_dim=int(pcfg["proprio_dim"]),
            action_dim=int(pcfg["action_dim"]),
            hidden=tuple(pcfg["hidden"]),
        )
        return backbone.to(self.device), policy.to(self.device), lora_params

    def train(self, dataset: BCDataset, backbone: PerceptionBackbone,
              policy: BCPolicy, lora_params: list[nn.Parameter],
              out_dir: Path, epochs: int | None = None) -> dict:
        ocfg = self.cfg["optim"]
        epochs = epochs or int(ocfg["epochs"])
        loader = DataLoader(
            dataset, batch_size=int(ocfg["batch_size"]), shuffle=True,
            num_workers=int(ocfg.get("num_workers", 0)), pin_memory=True,
            drop_last=len(dataset) > int(ocfg["batch_size"]),
        )
        trainable = list(policy.parameters()) + lora_params
        opt = torch.optim.AdamW(
            trainable, lr=float(ocfg["lr"]), weight_decay=float(ocfg["weight_decay"]))
        loss_fn = nn.L1Loss() if ocfg.get("loss", "l1") == "l1" else nn.MSELoss()

        backbone.train(False)  # BN 统计保持冻结; LoRA 分支仍可训练
        policy.train(True)
        history = []
        t0 = time.time()
        for ep in range(epochs):
            ep_loss, n = 0.0, 0
            for img, prop, act in loader:
                img = img.to(self.device, non_blocking=True)
                prop = prop.to(self.device, non_blocking=True)
                act = act.to(self.device, non_blocking=True)
                feat = backbone(img)
                pred = policy(feat, prop)
                loss = loss_fn(pred, act)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(img)
                n += len(img)
            history.append(ep_loss / max(n, 1))
        wall = time.time() - t0

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), out_dir / "policy.pth")
        if lora_params:
            torch.save(lora_state_dict(backbone), out_dir / "lora.pth")
        stats = {
            "epochs": epochs, "final_loss": history[-1] if history else None,
            "loss_history": history, "wall_sec": wall,
            "n_samples": len(dataset),
            "n_trainable": int(sum(p.numel() for p in trainable)),
        }
        (out_dir / "train_stats.json").write_text(json.dumps(stats), encoding="utf-8")
        return stats


def make_action_stats(actions: np.ndarray) -> dict:
    """动作分布统计, 写进 manifest 供排查 (动作本身已在 [-1,1], 不再缩放)。"""
    return {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
    }
