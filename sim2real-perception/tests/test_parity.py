"""跨语言一致性 (集成测试): pybind11 模块 vs Python onnxruntime。

需要先完成 06_export (deploy_bundle) 与 C++ 构建, 否则跳过。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

BUNDLE = ROOT / "deploy_bundle"

s2r_cpp = pytest.importorskip("s2r_cpp", reason="C++ 绑定未构建")
pytestmark = pytest.mark.skipif(
    not (BUNDLE / "manifest.json").exists(), reason="deploy_bundle 未导出")


def test_pipeline_preprocess_matches_numpy():
    pipe = s2r_cpp.ImagePipeline(224)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    chw_cpp = pipe.preprocess_rgb(img)
    chw_np = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    assert np.abs(chw_cpp - chw_np).max() < 1e-6


def test_runtime_step_matches_onnxruntime():
    import onnxruntime as ort

    rt = s2r_cpp.DeployRuntime(str(BUNDLE))
    sess_p = ort.InferenceSession(str(BUNDLE / "perception.onnx"),
                                  providers=["CPUExecutionProvider"])
    sess_a = ort.InferenceSession(str(BUNDLE / "policy.onnx"),
                                  providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    for _ in range(4):
        img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        prop = rng.random(8, dtype=np.float32)
        a_cpp, accepted, score, t_us = rt.step(img, prop)

        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        feat = sess_p.run(None, {"image": x})[0]
        a_py = sess_a.run(None, {"feature": feat, "proprio": prop[None]})[0][0]
        assert np.abs(a_cpp - a_py).max() < 1e-4
        assert isinstance(accepted, bool) and t_us > 0
