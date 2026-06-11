"""部署 bundle 导出: Python 训练侧与 C++ 部署侧的唯一契约。

deploy_bundle/
├── manifest.json     元信息: 输入规约 / 维度 / tau / 文件清单
├── perception.onnx   image  f32[1,3,224,224] (RGB, [0,1]) -> feature f32[1,2048]
├── policy.onnx       feature f32[1,2048] + proprio f32[1,8] -> action f32[1,7]
└── anchors.bin       K x 2048 float32 row-major, 已 L2 归一化

约定: perception 输出的是 *原始* 特征 (供 policy 用);
cosine 门控在消费端先做 L2 归一化再与 anchors 点积。
LoRA 在导出前合并进权重, C++ 端对 backbone 来源完全无感。
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from sim2real.perception.backbone import FEATURE_DIM, PerceptionBackbone
from sim2real.perception.filter import CosineFilter
from sim2real.perception.lora import merge_lora
from sim2real.policy.bc import BCPolicy
from sim2real.sim.env import DPOS_MAX, DROT_MAX

BUNDLE_VERSION = 1


def export_bundle(backbone: PerceptionBackbone, policy: BCPolicy,
                  cos_filter: CosineFilter | None, out_dir: Path,
                  image_size: int = 224, opset: int = 17) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    bb = copy.deepcopy(backbone).to(device).eval()
    n_merged = merge_lora(bb)
    pol = copy.deepcopy(policy).to(device).eval()

    img = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        bb, (img,), str(out_dir / "perception.onnx"),
        input_names=["image"], output_names=["feature"],
        opset_version=opset, dynamo=False,
    )
    feat = torch.zeros(1, FEATURE_DIM)
    prop = torch.zeros(1, pol.proprio_dim)
    torch.onnx.export(
        pol, (feat, prop), str(out_dir / "policy.onnx"),
        input_names=["feature", "proprio"], output_names=["action"],
        opset_version=opset, dynamo=False,
    )

    n_anchors, tau = 0, None
    if cos_filter is not None:
        anchors = np.ascontiguousarray(cos_filter.anchors, dtype=np.float32)
        anchors.tofile(out_dir / "anchors.bin")
        n_anchors, tau = anchors.shape[0], cos_filter.tau

    manifest = {
        "version": BUNDLE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image": {"size": image_size, "layout": "NCHW", "color": "RGB",
                  "range": [0.0, 1.0]},
        "feature_dim": FEATURE_DIM,
        "proprio_dim": pol.proprio_dim,
        "action_dim": pol.action_dim,
        "action_scale": {"dpos_max": DPOS_MAX, "drot_max": DROT_MAX},
        "backbone": backbone.name,
        "lora_merged_layers": n_merged,
        "gate": {"n_anchors": n_anchors, "tau": tau,
                 "note": "consumer must L2-normalize feature before dot(anchors)"},
        "files": {"perception": "perception.onnx", "policy": "policy.onnx",
                  "anchors": "anchors.bin" if n_anchors else None},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_bundle(out_dir: Path, backbone: PerceptionBackbone,
                  policy: BCPolicy, n: int = 8, image_size: int = 224,
                  seed: int = 0) -> dict:
    """ONNX (CPU EP) vs PyTorch 数值一致性, 作为导出自检。"""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    imgs = rng.random((n, 3, image_size, image_size), dtype=np.float32)
    props = rng.random((n, policy.proprio_dim), dtype=np.float32)

    bb = copy.deepcopy(backbone).to("cpu").eval()
    merge_lora(bb)
    pol = copy.deepcopy(policy).to("cpu").eval()

    sess_p = ort.InferenceSession(str(out_dir / "perception.onnx"),
                                  providers=["CPUExecutionProvider"])
    sess_a = ort.InferenceSession(str(out_dir / "policy.onnx"),
                                  providers=["CPUExecutionProvider"])
    feat_diff, act_diff = 0.0, 0.0
    with torch.no_grad():
        for i in range(n):
            t_feat = bb(torch.from_numpy(imgs[i: i + 1])).numpy()
            o_feat = sess_p.run(None, {"image": imgs[i: i + 1]})[0]
            feat_diff = max(feat_diff, float(np.abs(t_feat - o_feat).max()))
            t_act = pol(torch.from_numpy(t_feat),
                        torch.from_numpy(props[i: i + 1])).numpy()
            o_act = sess_a.run(None, {"feature": o_feat,
                                      "proprio": props[i: i + 1]})[0]
            act_diff = max(act_diff, float(np.abs(t_act - o_act).max()))
    return {"feature_max_abs_diff": feat_diff, "action_max_abs_diff": act_diff}
