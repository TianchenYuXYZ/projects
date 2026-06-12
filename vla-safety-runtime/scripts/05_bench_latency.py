"""E2: 恢复通道墙钟延迟分解。

测量边界: 深度帧已在主机内存 (相当于传感器帧 DMA 完成) -> recovery
指令对控制环可见。链路分解:

  t_gpu    深度上传 + 反投影/走廊核函数 + 归约回传 (CUDA events, 含排队)
  t_plan   bandit select + recovery plan (CPU)
  t_tok    arm token 解码 -> 速度指令 (CPU)
  t_write  seqlock 共享内存写 (CPU)
  e2e_py   check() 入口 -> SHM 写完成 (perf_counter)
  shm_vis  跨进程可见延迟 (Python QPC 写戳 -> C++ 自旋读 QPC, 同源时钟)

GPU 争用条件 (复现文档 3.3 的 stream priority 论证):
  prio_idle    高优先级 stream, GPU 空闲          (下界)
  prio_load    高优先级 stream, VLA 解码满载      (设计点)
  noprio_load  同优先级独立 stream, 解码满载      (无优先级提示)
  shared_load  与解码共用 stream, 满载            (串行反模式)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

import vla_safety  # noqa: F401
from vla_safety.common import (PROJECT_ROOT, SceneSpec, V_MAX, load_yaml,
                               percentiles, save_json)
from vla_safety.env import ManipSafetyEnv
from vla_safety.runtime.ring import SeqlockReader, SeqlockWriter, qpc_freq, qpc_now
from vla_safety.runtime.streams import DecodeLoadGenerator, StreamRig
from vla_safety.safety.monitor import OracleMonitor
from vla_safety.vla.model import MiniVLA
from vla_safety.vla.tokenizer import ActionTokenizer

EXE_CANDIDATES = [
    PROJECT_ROOT / "cpp" / "build" / "Release" / "control_loop.exe",
    PROJECT_ROOT / "cpp" / "build" / "Debug" / "control_loop.exe",
]


def setup_env_near_obstacle(cfg: dict) -> tuple[ManipSafetyEnv, np.ndarray]:
    """把 EE 开到障碍前 ~16cm 的巡航位姿, 之后保持静止重复渲染。"""
    obst = (0.46, 0.01)
    env = ManipSafetyEnv(SceneSpec(cube_pos=(0.56, 0.0), obstacle_pos=obst),
                         render_size=int(cfg["env"]["render_size"]),
                         depth_size=int(cfg["env"]["depth_size"]),
                         workspace=cfg["env"]["workspace"])
    env.reset()
    travel_z = float(cfg["env"]["travel_ee_z"])
    env.set_command(np.array([0, 0, -1.0, -1.0]))
    for _ in range(400):
        env.tick()
        if abs(env.ee_pos[2] - travel_z) < 0.005:
            break
    while np.linalg.norm(np.array(obst) - env.ee_pos[:2]) > 0.16:
        to = np.array(obst) - env.ee_pos[:2]
        u = to / np.linalg.norm(to)
        env.set_command(np.array([u[0] * 0.8, u[1] * 0.8, 0, -1.0]))
        env.tick()
    env.set_command(np.zeros(4))
    to = np.array(obst) - env.ee_pos[:2]
    u = to / np.linalg.norm(to)
    v_world = np.array([u[0], u[1], 0.0]) * 0.8 * V_MAX
    return env, v_world


def bench_condition(name: str, cfg: dict, env: ManipSafetyEnv,
                    v_world: np.ndarray, model: MiniVLA,
                    device: torch.device, writer: SeqlockWriter,
                    n_trials: int, warmup: int) -> dict:
    use_priority = name.startswith("prio")
    with_load = name.endswith("load")
    shared = name.startswith("shared")

    rig = StreamRig(safety_priority=use_priority)
    safety_stream = rig.stream_main if shared else rig.stream_safety
    bandit_cfg = dict(cfg["bandit"]); bandit_cfg["algo"] = "ucb1"
    monitor = OracleMonitor(cfg["safety"], cfg["recovery"], bandit_cfg,
                            ActionTokenizer(),
                            depth_size=int(cfg["env"]["depth_size"]),
                            fovy_deg=env.depth_fovy, device=str(device),
                            stream=safety_stream, timing=True)

    depth, cpos, cmat = env.render_depth()
    ee, tcp, goal = env.ee_pos, env.tcp_pos, env.cube_pos

    rows = {"t_gpu_ms": [], "t_plan_us": [], "t_tok_us": [],
            "t_write_us": [], "e2e_ms": []}
    n_conflict = 0

    def one_trial():
        nonlocal n_conflict
        t0 = time.perf_counter_ns()
        rep = monitor.check(depth, cpos, cmat, ee, tcp, goal, v_world)
        t1 = time.perf_counter_ns()
        if not rep.conflict:
            return None
        n_conflict += 1
        plan = monitor.plan_recovery(rep, 0, 0.3)
        t2 = time.perf_counter_ns()
        v = monitor.tokenizer.decode(plan.arm.tokens[0])
        cmd = np.array([v[0], v[1], v[2], -1.0])
        t3 = time.perf_counter_ns()
        writer.write(cmd, source=1)
        t4 = time.perf_counter_ns()
        # 维持 bandit 干净状态 (本脚本只测延迟, 不学习)
        monitor.bandit.counts[plan.context][plan.arm.index] += 1
        monitor.bandit.succ[plan.context][plan.arm.index] += 1
        return {"t_gpu_ms": rep.t_gpu_ms, "t_plan_us": (t2 - t1) / 1e3,
                "t_tok_us": (t3 - t2) / 1e3, "t_write_us": (t4 - t3) / 1e3,
                "e2e_ms": (t4 - t0) / 1e6}

    load_ctx = (DecodeLoadGenerator(model, device, rig.stream_main,
                                    int(cfg["latency_bench"]["decode_load_repeats"]))
                if with_load else _Null())
    with load_ctx:
        if with_load:
            time.sleep(0.5)                       # 等负载稳态
        for _ in range(warmup):
            one_trial()
        for _ in range(n_trials):
            r = one_trial()
            if r is not None:
                for k, val in r.items():
                    rows[k].append(val)

    assert n_conflict > 0, f"{name}: 没有任何冲突命中, 场景搭建失败"
    out = {k: percentiles(vs) for k, vs in rows.items()}
    out["priority_used"] = rig.priority_used if use_priority else 0
    out["n_samples"] = len(rows["e2e_ms"])
    return out


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def bench_shm_cross_process(cfg: dict, writer: SeqlockWriter,
                            results_dir: Path) -> dict:
    """C++ 读端实测跨进程可见延迟 (自旋) 与 200Hz 控制节拍拾取延迟。"""
    exe = next((p for p in EXE_CANDIDATES if p.exists()), None)
    if exe is None:
        raise FileNotFoundError("control_loop.exe 未构建; 先运行 06 的构建步骤")
    out = {}
    for mode, hz, dur in [("spin", 0, 6), ("hz200", 200, 6)]:
        csv_path = results_dir / f"shm_{mode}.csv"
        proc = subprocess.Popen(
            [str(exe), "--shm", cfg["latency_bench"]["shm_name"],
             "--hz", str(hz), "--duration", str(dur),
             "--out", str(csv_path), "--wait", "10"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.8)                            # 读端就绪
        t_end = time.time() + dur - 1.0
        i = 0
        while time.time() < t_end:
            writer.write(np.array([0.1, 0.2, 0.3, -1.0]), source=1)
            i += 1
            time.sleep(0.005)                      # ~200Hz 写
        stdout, stderr = proc.communicate(timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"control_loop 退出码 {proc.returncode}: {stderr}")
        import json
        out[mode] = json.loads(stdout.strip().splitlines()[-1])
        out[mode]["n_written"] = i
    return out


def main() -> None:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    bcfg = cfg["latency_bench"]
    device = torch.device("cuda")
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(exist_ok=True)

    print("[1/4] 搭建障碍前位姿场景")
    env, v_world = setup_env_near_obstacle(cfg)
    dist = float(np.linalg.norm(np.array([0.46, 0.01]) - env.ee_pos[:2]))
    print(f"      EE {env.ee_pos.round(3)}, 距障碍 {dist:.3f} m")

    rsz = int(cfg["env"]["render_size"])
    model = MiniVLA.from_config(cfg["vla"], rsz).to(device).eval()
    # 负载规模标定: 单次 decode_load 墙钟
    img = torch.randn(1, 3, rsz, rsz, device=device)
    wimg = torch.randn(1, 3, rsz, rsz, device=device)
    prop = torch.randn(1, 4, device=device)
    model.decode_load(img, wimg, prop, 4)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.decode_load(img, wimg, prop, int(bcfg["decode_load_repeats"]))
    torch.cuda.synchronize()
    load_iter_ms = (time.perf_counter() - t0) * 1e3
    print(f"      解码负载单轮 {load_iter_ms:.1f} ms (repeats="
          f"{bcfg['decode_load_repeats']}, 持续循环占满 stream_main)")

    writer = SeqlockWriter(bcfg["shm_name"])
    # 写端自检
    rd = SeqlockReader(bcfg["shm_name"])
    writer.write(np.array([1, 2, 3, 4.0]), source=0)
    chk = rd.read()
    assert chk is not None and abs(chk["v"][2] - 3.0) < 1e-6, "SHM 回读自检失败"
    rd.close()

    results = {"load_iter_ms": load_iter_ms, "conditions": {}}
    n_trials, warmup = int(bcfg["n_trials"]), int(bcfg["warmup"])
    print("[2/4] GPU 链路四条件")
    for cond in ["prio_idle", "prio_load", "noprio_load", "shared_load"]:
        print(f"      {cond} ...")
        r = bench_condition(cond, cfg, env, v_world, model, device,
                            writer, n_trials, warmup)
        results["conditions"][cond] = r
        print(f"        t_gpu p50/p99 = {r['t_gpu_ms']['p50']:.2f}/"
              f"{r['t_gpu_ms']['p99']:.2f} ms, e2e p99 = "
              f"{r['e2e_ms']['p99']:.2f} ms (n={r['n_samples']})")

    print("[3/4] 跨进程 SHM 可见延迟 (C++ 读端)")
    shm = bench_shm_cross_process(cfg, writer, results_dir)
    results["shm"] = shm
    print(f"      spin: p50 {shm['spin']['latency_us']['p50']:.1f} us, "
          f"p99 {shm['spin']['latency_us']['p99']:.1f} us; "
          f"200Hz 拾取: p99 {shm['hz200']['latency_us']['p99'] / 1000:.2f} ms")

    e2e99 = results["conditions"]["prio_load"]["e2e_ms"]["p99"]
    vis99 = shm["spin"]["latency_us"]["p99"] / 1000.0
    results["headline"] = {
        "recovery_p99_ms_under_load": e2e99 + vis99,
        "claim": "< 5 ms",
        "met": bool(e2e99 + vis99 < 5.0),
    }
    print(f"[4/4] 满载恢复链路 p99 = {e2e99:.2f} + {vis99:.3f} = "
          f"{e2e99 + vis99:.2f} ms (声明 < 5 ms: "
          f"{'PASS' if results['headline']['met'] else 'FAIL'})")

    writer.close()
    env.close()
    save_json(results, results_dir / "latency.json")
    print(f"-> {results_dir / 'latency.json'}")


if __name__ == "__main__":
    main()
