"""mini-VLA 行为克隆训练: 逐 token 交叉熵 (RT-2 风格)。

指标:
  - token_acc: 每维 token 严格命中率
  - val_mae:   反 token 化后的动作 MAE (物理可解释: 归一化速度单位)
按 val loss 保存最优权重 -> weights/vla_bc.pt; 训练曲线 -> weights/train_curve.png
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import vla_safety  # noqa: F401
from vla_safety.common import PROJECT_ROOT, load_yaml, save_json
from vla_safety.vla.dataset import make_splits
from vla_safety.vla.model import MiniVLA
from vla_safety.vla.tokenizer import ActionTokenizer


@torch.no_grad()
def evaluate(model, loader, device, tokenizer) -> dict:
    model.eval()
    tot_loss, tot_correct, tot_tok, tot_mae, n = 0.0, 0, 0, 0.0, 0
    for img, wimg, prop, tok in loader:
        img, wimg, prop, tok = (img.to(device), wimg.to(device),
                                prop.to(device), tok.to(device))
        logits = model(img, wimg, prop, tok)
        loss = F.cross_entropy(logits.flatten(0, 1), tok.flatten())
        pred = logits.argmax(-1)
        tot_loss += loss.item() * tok.numel()
        tot_correct += (pred == tok).sum().item()
        tot_tok += tok.numel()
        a_pred = tokenizer.decode(pred.cpu().numpy())
        a_true = tokenizer.decode(tok.cpu().numpy())
        tot_mae += float(np.abs(a_pred - a_true).sum())
        n += tok.numel()
    return {"loss": tot_loss / tot_tok, "token_acc": tot_correct / tot_tok,
            "mae": tot_mae / n}


def main() -> None:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    tcfg = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(tcfg["seed"]))

    train_ds, val_ds, stats = make_splits(
        PROJECT_ROOT / cfg["paths"]["demos_dir"],
        float(tcfg["val_frac"]), int(tcfg["seed"]))
    print(f"数据: {stats}")
    train_loader = DataLoader(train_ds, batch_size=int(tcfg["batch_size"]),
                              shuffle=True, num_workers=int(tcfg["num_workers"]),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = MiniVLA.from_config(cfg["vla"], cfg["env"]["render_size"]).to(device)
    print(f"模型参数量: {model.n_params() / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))
    epochs = int(tcfg["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup = max(1, int(0.03 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    tokenizer = ActionTokenizer()

    history = []
    best_val = float("inf")
    ckpt_path = PROJECT_ROOT / cfg["paths"]["weights"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss, ep_tok = 0.0, 0
        for img, wimg, prop, tok in train_loader:
            img, wimg, prop, tok = (img.to(device), wimg.to(device),
                                    prop.to(device), tok.to(device))
            logits = model(img, wimg, prop, tok)
            loss = F.cross_entropy(logits.flatten(0, 1), tok.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item() * tok.numel()
            ep_tok += tok.numel()
        val = evaluate(model, val_loader, device, tokenizer)
        history.append({"epoch": ep, "train_loss": ep_loss / ep_tok, **{
            f"val_{k}": v for k, v in val.items()}})
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({"model": model.state_dict(),
                        "cfg": cfg["vla"], "epoch": ep}, ckpt_path)
        print(f"epoch {ep:02d}  train {ep_loss / ep_tok:.4f}  "
              f"val {val['loss']:.4f}  acc {val['token_acc']:.3f}  "
              f"mae {val['mae']:.4f}  ({time.time() - t0:.0f}s)")

    # 曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([h["train_loss"] for h in history], label="train")
    ax[0].plot([h["val_loss"] for h in history], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("CE loss"); ax[0].legend()
    ax[1].plot([h["val_token_acc"] for h in history], label="token acc")
    ax[1].plot([h["val_mae"] for h in history], label="action MAE")
    ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(ckpt_path.parent / "train_curve.png", dpi=120)

    save_json({"history": history, "best_val_loss": best_val, "data": stats,
               "n_params": model.n_params(),
               "wall_s": round(time.time() - t0, 1)},
              ckpt_path.parent / "train_report.json")
    print(f"最优 val loss {best_val:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
