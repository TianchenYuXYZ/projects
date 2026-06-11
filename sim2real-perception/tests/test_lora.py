"""LoRA 注入/合并的数学正确性。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torchvision

from sim2real.perception.lora import LoRAConv2d, inject_lora, lora_state_dict, merge_lora


def _toy_backbone():
    trunk = torchvision.models.resnet50(weights=None)
    trunk.fc = torch.nn.Identity()
    return trunk


def test_injection_is_identity_at_init():
    """B 零初始化 -> 注入后输出与原网络逐位一致。"""
    torch.manual_seed(0)
    net = _toy_backbone().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y0 = net(x)
    inject_lora(net, rank=4, alpha=8, target_stages=["layer3", "layer4"])
    with torch.no_grad():
        y1 = net(x)
    assert torch.allclose(y0, y1, atol=0), "注入瞬间必须严格等价"


def test_merge_matches_lora_forward():
    """扰动 LoRA 参数后, merge 出的普通 conv 与旁路 forward 等价。"""
    torch.manual_seed(1)
    net = _toy_backbone().eval()
    params = inject_lora(net, rank=4, alpha=8, target_stages=["layer4"])
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p) * 0.02)  # 模拟训练后的非零 LoRA
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y_lora = net(x)
    n = merge_lora(net)
    assert n > 0
    assert not any(isinstance(m, LoRAConv2d) for m in net.modules())
    with torch.no_grad():
        y_merged = net(x)
    assert torch.allclose(y_lora, y_merged, atol=1e-4), \
        f"merge 偏差 {(y_lora - y_merged).abs().max().item()}"


def test_trainable_param_budget():
    """LoRA 参数量必须远小于 backbone (文档: 几千万 -> 百万级)。"""
    net = _toy_backbone()
    total = sum(p.numel() for p in net.parameters())
    params = inject_lora(net, rank=8, alpha=16, target_stages=["layer3", "layer4"])
    trainable = sum(p.numel() for p in params)
    assert trainable < total * 0.05, f"LoRA 参数占比过高: {trainable}/{total}"
    sd = lora_state_dict(net)
    assert all(("lora_A" in k or "lora_B" in k) for k in sd)
    assert sum(v.numel() for v in sd.values()) == trainable
