"""导出部署 bundle: merge LoRA -> ONNX x2 + anchors.bin + manifest.json + 自检。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib
import json

import numpy as np

from sim2real.common import PROJECT_ROOT, load_yaml
from sim2real.export.bundle import export_bundle, verify_bundle
from sim2real.perception.filter import CosineFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_agent = importlib.import_module("05_eval").load_agent  # 复用模型装载逻辑


def main() -> None:
    tcfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")
    dpcfg = load_yaml(PROJECT_ROOT / "configs" / "deploy.yaml")

    agent = load_agent("ours", PROJECT_ROOT / "runs" / "ours", tcfg)

    fdir = PROJECT_ROOT / "data" / "filter"
    anchors = np.load(fdir / "anchors.npy")
    tau = json.loads((fdir / "tau.json").read_text(encoding="utf-8"))["tau"]
    cos_filter = CosineFilter(anchors, tau)

    out_dir = PROJECT_ROOT / dpcfg["bundle_dir"]
    manifest = export_bundle(
        agent.backbone, agent.policy, cos_filter, out_dir,
        image_size=int(dpcfg["image_size"]), opset=int(dpcfg["opset"]))
    print(json.dumps(manifest, indent=2))

    diffs = verify_bundle(out_dir, agent.backbone, agent.policy)
    print(f"[verify] torch vs onnxruntime: {diffs}")
    assert diffs["action_max_abs_diff"] < 1e-4, "ONNX 导出数值不一致"
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
