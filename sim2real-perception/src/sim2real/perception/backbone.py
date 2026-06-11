"""感知 backbone: R3M (ResNet50 trunk) 优先, ImageNet ResNet50 回退。

统一输入契约 (Python / ONNX / C++ 三端一致):
    float32, RGB, CHW, [0, 1], 224x224
ImageNet mean/std 归一化作为模块第一层 **烘焙进网络** (随 ONNX 导出),
C++ 端因此无需关心 backbone 具体来源。输出 2048 维特征。

R3M 权重获取: scripts/00_setup_assets.py 会尝试经官方 r3m 包下载并把
convnet trunk 的 state_dict 落到 weights/r3m_resnet50.pth; 失败则训练配置
回退到 imagenet_resnet50 (接口完全一致, 文档化即可)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from sim2real.common import PROJECT_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 2048


class _Normalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class PerceptionBackbone(nn.Module):
    """encode: (B,3,224,224) float[0,1] -> (B,2048)。"""

    def __init__(self, name: str = "r3m_resnet50",
                 weights_dir: str | Path = "weights"):
        super().__init__()
        self.name = name
        trunk = torchvision.models.resnet50(weights=None)
        trunk.fc = nn.Identity()

        wdir = PROJECT_ROOT / weights_dir
        if name == "r3m_resnet50":
            ckpt = wdir / "r3m_resnet50.pth"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"{ckpt} 不存在; 先运行 scripts/00_setup_assets.py, "
                    f"或将 backbone.name 改为 imagenet_resnet50"
                )
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = trunk.load_state_dict(state, strict=False)
            assert not missing, f"R3M 权重缺 key: {missing[:5]}"
        elif name == "imagenet_resnet50":
            trunk = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )
            trunk.fc = nn.Identity()
        else:
            raise ValueError(f"未知 backbone: {name}")

        self.normalize = _Normalize()
        self.trunk = trunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.normalize(x))

    # 训练管线里语义化别名
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)


@torch.no_grad()
def embed_images(backbone: PerceptionBackbone, images_u8: np.ndarray,
                 device: str = "cuda", batch_size: int = 128,
                 l2_normalize: bool = True) -> np.ndarray:
    """uint8 (N,H,W,3) -> (N,2048) float32; 供过滤器与评测复用。"""
    backbone = backbone.to(device).eval()
    out = []
    for i in range(0, len(images_u8), batch_size):
        x = torch.from_numpy(images_u8[i: i + batch_size]).to(device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        f = backbone(x)
        if l2_normalize:
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(f.cpu().numpy())
    return np.concatenate(out).astype(np.float32)
