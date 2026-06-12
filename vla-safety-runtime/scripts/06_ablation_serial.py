"""E6: 感知-策略解耦 vs 串行耦合 —— 控制环频率实验。

复现文档 3.1 的论证: VLA thinking 低频 (1-10Hz) 但控制必须高频,
绑在同一执行线程是反模式。三个条件下用墙钟跑 100Hz 控制环:

  floor    无策略负载                       (本机控制环上限)
  disagg   VLA 解码在后台线程 + 独立 stream  (本项目架构)
  serial   VLA 解码内联阻塞控制环            (耦合反模式)

指标: 实际达成频率 / deadline miss 率 / tick 间隔 p99。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

import vla_safety  # noqa: F401
from vla_safety.common import PROJECT_ROOT, SceneSpec, load_yaml, percentiles, save_json
from vla_safety.env import ManipSafetyEnv
from vla_safety.runtime.streams import DecodeLoadGenerator, StreamRig
from vla_safety.vla.model import MiniVLA

TARGET_HZ = 100
DURATION_S = 15.0
THINK_PERIOD_S = 0.15           # serial 模式: 每 150ms 内联一次解码


def run_loop(env: ManipSafetyEnv, mode: str, model, device, rig,
             repeats: int, img_size: int = 128) -> dict:
    period = 1.0 / TARGET_HZ
    img = torch.randn(1, 3, img_size, img_size, device=device)
    wimg = torch.randn(1, 3, img_size, img_size, device=device)
    prop = torch.randn(1, 4, device=device)

    intervals = []
    misses = 0
    env.set_command(np.array([0.3, 0.0, -0.1, -1.0]))

    load_ctx = (DecodeLoadGenerator(model, device, rig.stream_main, repeats)
                if mode == "disagg" else _Null())
    with load_ctx:
        if mode == "disagg":
            time.sleep(0.5)
        t_start = time.perf_counter()
        next_tick = t_start
        last = t_start
        next_think = t_start
        ticks = 0
        while True:
            now = time.perf_counter()
            if now - t_start >= DURATION_S:
                break
            if mode == "serial" and now >= next_think:
                # 反模式: 解码同步阻塞控制线程
                with torch.cuda.stream(rig.stream_main):
                    model.decode_load(img, wimg, prop, repeats)
                rig.stream_main.synchronize()
                next_think = now + THINK_PERIOD_S
            env.tick()
            ticks += 1
            now2 = time.perf_counter()
            intervals.append((now2 - last) * 1e3)
            if (now2 - last) > 1.5 * period:
                misses += 1
            last = now2
            next_tick += period
            # 混合等待: 余量大时 sleep 让出 GIL (后台解码线程需要时间片),
            # 临近节拍自旋保精度
            while True:
                rem = next_tick - time.perf_counter()
                if rem <= 0:
                    break
                if rem > 0.002:
                    time.sleep(0.001)
                else:
                    pass
            if next_tick < time.perf_counter() - period:
                next_tick = time.perf_counter()        # 落拍重相位
    wall = time.perf_counter() - t_start
    return {
        "achieved_hz": ticks / wall,
        "deadline_miss_rate": misses / max(1, ticks),
        "interval_ms": percentiles(intervals, ps=(50, 95, 99)),
        "ticks": ticks,
    }


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> None:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    device = torch.device("cuda")
    repeats = int(cfg["latency_bench"]["decode_load_repeats"])
    model = MiniVLA.from_config(cfg["vla"], cfg["env"]["render_size"]).to(device).eval()
    rig = StreamRig()

    env = ManipSafetyEnv(SceneSpec.nominal(),
                         render_size=int(cfg["env"]["render_size"]),
                         depth_size=int(cfg["env"]["depth_size"]),
                         workspace=cfg["env"]["workspace"])
    results = {}
    for mode in ["floor", "disagg", "serial"]:
        env.reset()
        print(f"mode={mode} ({DURATION_S:.0f}s @ 目标 {TARGET_HZ}Hz)")
        r = run_loop(env, mode, model, device, rig, repeats)
        results[mode] = r
        print(f"  达成 {r['achieved_hz']:.1f} Hz, miss {r['deadline_miss_rate']:.2%}, "
              f"间隔 p99 {r['interval_ms']['p99']:.1f} ms")
    env.close()

    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    save_json(results, results_dir / "ablation_serial.json")
    print(f"-> {results_dir / 'ablation_serial.json'}")


if __name__ == "__main__":
    main()
