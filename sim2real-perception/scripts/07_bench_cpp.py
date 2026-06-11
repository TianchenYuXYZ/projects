"""C++ runtime 验收: parity (Python vs C++ 动作一致性) + 延迟报告。

流程:
1. 生成 parity 集: 评测场景渲染 N 帧 (PNG) + proprio.csv -> data/parity/
2. Python 端用 onnxruntime 跑 bundle -> actions_py.csv
3. 调 C++ deploy_replay -> actions_cpp.csv + latency.json
4. 对比并打印报告
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from sim2real.common import PROJECT_ROOT, load_yaml
from sim2real.datagen.randomizer import DomainRandomizer, build_texture_pool
from sim2real.sim.env import ManipEnv

N_FRAMES = 64
EXE_CANDIDATES = [
    PROJECT_ROOT / "cpp" / "build" / "bin" / "Release" / "deploy_replay.exe",
    PROJECT_ROOT / "cpp" / "build" / "apps" / "Release" / "deploy_replay.exe",
]


def make_parity_set(out_dir: Path) -> None:
    """随机化场景下沿专家轨迹渲染帧, 覆盖分布内外样本。"""
    from sim2real.common import Trajectory

    dcfg = load_yaml(PROJECT_ROOT / "configs" / "dr.yaml")
    demo = Trajectory.load(PROJECT_ROOT / "data" / "demo.npz")
    pool = build_texture_pool(dcfg, None)
    rng = np.random.default_rng(777)
    randomizer = DomainRandomizer(dcfg, texture_pool=pool)

    out_dir.mkdir(parents=True, exist_ok=True)
    env = ManipEnv(randomizer.sample_scene(rng))
    rows = []
    per_scene = 8
    fid = 0
    indices = np.linspace(0, len(demo) - 1, per_scene).round().astype(int)
    while fid < N_FRAMES:
        env.reset(randomizer.sample_scene(rng))
        frames = env.replay_render(demo.qpos, indices)
        for k, t in enumerate(indices):
            if fid >= N_FRAMES:
                break
            Image.fromarray(frames[k]).save(out_dir / f"frame_{fid:04d}.png")
            rows.append([f"frame_{fid:04d}.png"] + demo.proprios[t].tolist())
            fid += 1
    env.close()
    with open(out_dir / "proprio.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[parity] {fid} frames -> {out_dir}")


def run_python_reference(parity_dir: Path, bundle_dir: Path) -> np.ndarray:
    """与 C++ 完全相同的输入路径: PNG -> [0,1] CHW -> onnxruntime CPU。"""
    import onnxruntime as ort

    sess_p = ort.InferenceSession(str(bundle_dir / "perception.onnx"),
                                  providers=["CPUExecutionProvider"])
    sess_a = ort.InferenceSession(str(bundle_dir / "policy.onnx"),
                                  providers=["CPUExecutionProvider"])
    rows = list(csv.reader(open(parity_dir / "proprio.csv", encoding="utf-8")))
    actions = []
    for name, *prop in rows:
        img = np.asarray(Image.open(parity_dir / name), dtype=np.float32) / 255.0
        x = img.transpose(2, 0, 1)[None]
        feat = sess_p.run(None, {"image": x})[0]
        a = sess_a.run(None, {"feature": feat,
                              "proprio": np.array([prop], dtype=np.float32)})[0]
        actions.append(a[0])
    out = np.stack(actions)
    np.savetxt(parity_dir / "actions_py.csv", out, delimiter=",", fmt="%.8f")
    return out


def main() -> None:
    dpcfg = load_yaml(PROJECT_ROOT / "configs" / "deploy.yaml")
    bundle_dir = PROJECT_ROOT / dpcfg["bundle_dir"]
    parity_dir = PROJECT_ROOT / "data" / "parity"

    if not (parity_dir / "proprio.csv").exists():
        make_parity_set(parity_dir)
    ref = run_python_reference(parity_dir, bundle_dir)

    exe = next((p for p in EXE_CANDIDATES if p.exists()), None)
    if exe is None:
        raise SystemExit(f"未找到 deploy_replay.exe, 先构建 cpp/ (见 cpp/README)")
    subprocess.run([str(exe), str(bundle_dir), str(parity_dir)], check=True)

    cpp = np.loadtxt(parity_dir / "actions_cpp.csv", delimiter=",")
    diff = float(np.abs(ref - cpp).max())
    tol = float(dpcfg["parity_tol"])
    lat = json.loads((parity_dir / "latency.json").read_text(encoding="utf-8"))
    print(f"[parity] max |a_py - a_cpp| = {diff:.2e}  (tol {tol:.0e}) "
          f"{'PASS' if diff < tol else 'FAIL'}")
    print(f"[latency] {json.dumps(lat, indent=2)}")
    if diff >= tol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
