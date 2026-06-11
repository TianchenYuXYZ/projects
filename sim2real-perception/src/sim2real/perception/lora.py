"""ResNet 版 LoRA: 对 Bottleneck 的 1x1 conv 注入低秩旁路。

y = W x + (alpha/r) * B(A(x)),  A: (r, in, 1, 1) kaiming 初始化,
B: (out, r, 1, 1) 零初始化 -> 注入瞬间网络输出不变。
merge_lora 把 BA 折叠回 W, 导出后推理零额外开销。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        assert base.kernel_size == (1, 1) and base.stride == (1, 1), \
            "只对 stride-1 的 1x1 conv 注入 LoRA"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Conv2d(base.in_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

    def merged_conv(self) -> nn.Conv2d:
        """返回折叠了 LoRA 增量的普通 Conv2d。"""
        merged = nn.Conv2d(
            self.base.in_channels, self.base.out_channels, 1,
            bias=self.base.bias is not None,
        )
        with torch.no_grad():
            delta = (self.lora_B.weight.squeeze(-1).squeeze(-1)
                     @ self.lora_A.weight.squeeze(-1).squeeze(-1))
            merged.weight.copy_(self.base.weight + self.scale * delta[..., None, None])
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


def inject_lora(backbone: nn.Module, rank: int, alpha: float,
                target_stages: list[str]) -> list[nn.Parameter]:
    """对 trunk 中指定 stage 的 Bottleneck conv1/conv3 (1x1, stride1) 注入。

    返回所有可训练 LoRA 参数 (backbone 其余参数保持冻结)。
    """
    trunk = backbone.trunk if hasattr(backbone, "trunk") else backbone
    params: list[nn.Parameter] = []
    for stage_name in target_stages:
        stage = getattr(trunk, stage_name)
        for block in stage:
            for conv_name in ("conv1", "conv3"):
                conv = getattr(block, conv_name)
                if isinstance(conv, LoRAConv2d):
                    continue
                if conv.kernel_size == (1, 1) and conv.stride == (1, 1):
                    wrapped = LoRAConv2d(conv, rank, alpha)
                    setattr(block, conv_name, wrapped)
                    params += [wrapped.lora_A.weight, wrapped.lora_B.weight]
    return params


def merge_lora(backbone: nn.Module) -> int:
    """把所有 LoRAConv2d 原地替换为折叠后的 Conv2d, 返回合并层数。"""
    n = 0
    for module in list(backbone.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRAConv2d):
                setattr(module, name, child.merged_conv())
                n += 1
    return n


def lora_state_dict(backbone: nn.Module) -> dict[str, torch.Tensor]:
    """只导出 LoRA 参数 (模块化: 一个 task 一份 adapter)。"""
    return {
        k: v for k, v in backbone.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
