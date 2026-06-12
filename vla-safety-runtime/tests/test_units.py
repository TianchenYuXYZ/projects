"""单元测试: tokenizer / arms / bandit / 反投影 / seqlock(Python端)。

运行: python tests/test_units.py   (无 mujoco 依赖, 全部毫秒级)
"""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

import vla_safety  # noqa: F401
from vla_safety.perception.depth_safety import DepthSafetyChecker
from vla_safety.runtime.ring import SeqlockReader, SeqlockWriter
from vla_safety.safety.arms import ARM_NAMES, build_arm_library
from vla_safety.safety.bandit import ContextualBandit
from vla_safety.vla.tokenizer import ActionTokenizer


def test_tokenizer_roundtrip():
    tk = ActionTokenizer()
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=(1000, 4))
    err = np.abs(tk.decode(tk.encode(a)) - a)
    assert err.max() <= tk.roundtrip_error_bound() + 1e-12, err.max()
    # 边界值
    assert (tk.encode(np.array([-1.0, 1.0, 0.0, 0.999])) ==
            np.array([0, 255, 128, 255])).all()
    print("  tokenizer roundtrip OK (max err "
          f"{err.max():.5f} <= {tk.roundtrip_error_bound():.5f})")


def test_arms():
    tk = ActionTokenizer()
    arms = build_arm_library(tk, arm_speed=0.12, steps=6)
    assert [a.name for a in arms] == ARM_NAMES
    v_norm = 0.12 / 0.15
    for a in arms:
        assert a.tokens.shape == (6, 3)
        dec = tk.decode(a.tokens)
        if a.name == "freeze":
            assert np.abs(dec).max() < 1.0 / 256 + 1e-9
        elif a.name == "retreat_up":
            assert abs(dec[0, 2] - v_norm) < 1.0 / 256
        elif a.name == "shift_y_minus":
            assert abs(dec[0, 1] + v_norm) < 1.0 / 256
    print("  recovery arms 编码/解码 OK")


def test_bandit_convergence():
    # 合成 Bernoulli 环境: arm 概率 [0.2, 0.8, 0.5, 0.3, 0.1], 最优 = 1
    probs = np.array([0.2, 0.8, 0.5, 0.3, 0.1])
    for algo in ("ucb1", "thompson"):
        rng = np.random.default_rng(7)
        b = ContextualBandit(algo, 5, ["c"], rng, ucb_c=1.2)
        env_rng = np.random.default_rng(11)
        pulls_best = 0
        for t in range(600):
            arm = b.select("c")
            r = float(env_rng.random() < probs[arm])
            b.update("c", arm, r)
            if t >= 300 and arm == 1:
                pulls_best += 1
        frac = pulls_best / 300
        assert frac > 0.6, f"{algo}: 后半程最优臂占比仅 {frac:.2f}"
        print(f"  bandit {algo}: 后半程最优臂占比 {frac:.2f} OK")
    # fixed / random 行为
    b = ContextualBandit("fixed", 5, ["c"], np.random.default_rng(0), fixed_arm=3)
    assert all(b.select("c") == 3 for _ in range(10))
    print("  bandit fixed/random 行为 OK")


def test_backprojection_synthetic():
    """合成相机: 位于原点, R=I (x右 y上 -z视线)。在 (0, 0, -0.5) 放一个
    虚拟障碍块 (深度图中央 8x8 像素 d=0.5, 其余 d=10 远裁剪)。"""
    h = w = 64
    fovy = 90.0
    cfg = {"d_max": 2.0, "table_z_filter": -100.0, "self_mask_radius": 1e-6,
           "self_mask_extend": 0.0, "goal_mask_radius": 1e-6,
           "corridor_l_min": 0.8, "lookahead_s": 0.1, "corridor_radius": 0.1,
           "conflict_min_points": 4, "near_far_split": 0.4,
           "front_half_angle_deg": 20.0, "min_speed_trigger": 0.0}
    ck = DepthSafetyChecker(cfg, h, fovy, device="cpu")
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[28:36, 28:36] = 0.5
    cam_pos = np.zeros(3)
    cam_mat = np.eye(3)
    ee = np.array([0.0, 0.0, 0.3])          # EE 在相机上方 0.3
    tcp = ee + np.array([0, -1e-6, 0])
    goal = np.array([100.0, 100.0, 100.0])
    v = np.array([0.0, 0.0, -0.12])         # 向 -z 运动, 走廊指向障碍
    rep = ck.check(depth, cam_pos, cam_mat, ee, tcp, goal, v)
    assert rep.conflict, "合成障碍未检出"
    expect = np.array([0.0, 0.0, -0.5])
    err = np.linalg.norm(rep.centroid - expect)
    assert err < 0.02, f"质心误差 {err:.4f} (centroid={rep.centroid})"
    # 走廊偏向一边 -> 无冲突
    v2 = np.array([0.12, 0.0, 0.0])
    rep2 = ck.check(depth, cam_pos, cam_mat, ee, tcp, goal, v2)
    assert not rep2.conflict, "侧向走廊误报"
    print(f"  反投影合成场景 OK (质心误差 {err * 100:.2f} cm)")


def test_bearing_context():
    """方位角符号约定: 运动 +x, 障碍在 +y (左) -> bearing > 0 -> left。"""
    from vla_safety.perception.depth_safety import _signed_bearing_deg
    b = _signed_bearing_deg(np.array([1.0, 0, 0]), np.array([0.5, 0.5, 0]))
    assert abs(b - 45.0) < 1e-6, b
    b2 = _signed_bearing_deg(np.array([1.0, 0, 0]), np.array([0.5, -0.5, 0]))
    assert abs(b2 + 45.0) < 1e-6, b2
    print("  bearing 符号约定 OK (左正右负)")


def test_seqlock_python_threads():
    """Python 写端 + Python 读端并发: payload 由 cmd_id 推导, 校验一致性。"""
    tag = "Local\\vla_test_ring"
    w = SeqlockWriter(tag)
    r = SeqlockReader(tag)
    stop = threading.Event()
    torn = [0]
    reads = [0]

    def reader():
        while not stop.is_set():
            rec = r.read()
            if rec is None or rec["cmd_id"] == 0:
                continue
            reads[0] += 1
            base = float(rec["cmd_id"] % 1000)
            for i in range(4):
                if abs(rec["v"][i] - base * (i + 1)) > 1e-6:
                    torn[0] += 1
                    break

    th = threading.Thread(target=reader)
    th.start()
    import time
    for k in range(1, 30001):
        base = float(k % 1000)
        w.write([base, base * 2, base * 3, base * 4], source=k % 3)
        if k % 200 == 0:
            time.sleep(0)            # 让出 GIL, 读线程才有时间片
    stop.set()
    th.join()
    assert torn[0] == 0, f"检测到 {torn[0]} 次撕裂读"
    assert reads[0] > 20, f"读端有效读数过少 ({reads[0]})"
    w.close()
    r.close()
    print(f"  seqlock Python 端 OK ({reads[0]} 次一致读, 0 撕裂)")


if __name__ == "__main__":
    print("test_units:")
    test_tokenizer_roundtrip()
    test_arms()
    test_bandit_convergence()
    test_backprojection_synthetic()
    test_bearing_context()
    test_seqlock_python_threads()
    print("全部通过")
