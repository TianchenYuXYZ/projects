"""深度图 -> 扫掠走廊冲突检测 (GPU-B / stream-B 上的安全感知核)。

输入只有深度图 + 相机位姿 + 末端/指令状态 —— 监控器对障碍物没有任何
特权信息 (障碍 ground truth 只在评测统计里使用)。

管线 (全部张量运算, 跑在独立 CUDA stream 上):
  1. 深度图经 pinned staging 异步上传
  2. 针孔反投影到相机系, 再变换到世界系
  3. 过滤: 远裁剪 / 桌面点 / 自体 (attachment->TCP 胶囊) / 任务目标屏蔽
  4. 扫掠胶囊走廊: 线段 [ee, ee + v_hat * L], L = max(L_min, |v|*lookahead)
     点到线段距离 < radius 即走廊内
  5. 归约 (点数 + 质心) 异步拷回 pinned host

上下文离散化 (CPU, 微秒级): 障碍方位 {front,left,right} x 距离 {near,far}。
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import torch

CONTEXTS = ["front_near", "front_far", "left_near", "left_far",
            "right_near", "right_far"]


@dataclasses.dataclass
class ConflictReport:
    conflict: bool
    n_points: int
    centroid: np.ndarray | None      # 世界系冲突点质心
    dist: float                      # ee -> 质心距离 (m)
    bearing_deg: float               # 相对指令方向的带符号方位角 (左正右负)
    context: str | None              # CONTEXTS 之一
    t_gpu_ms: float = float("nan")   # GPU 段耗时 (上传+核函数+回传, 含排队)
    t_cpu_us: float = float("nan")   # CPU 段耗时 (上下文离散化)


def _no_conflict(t_gpu_ms: float = float("nan")) -> ConflictReport:
    return ConflictReport(False, 0, None, math.inf, 0.0, None, t_gpu_ms, 0.0)


class DepthSafetyChecker:
    def __init__(self, safety_cfg: dict, depth_size: int, fovy_deg: float,
                 device: str = "cuda",
                 stream: "torch.cuda.Stream | None" = None,
                 timing: bool = False):
        self.cfg = safety_cfg
        self.device = torch.device(device)
        self.stream = stream
        self.timing = timing and self.device.type == "cuda"
        h = w = depth_size

        # 像素中心射线因子: x 右, y 上, 看向 -z (MuJoCo 相机约定)
        f = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        us = np.arange(w, dtype=np.float32) + 0.5
        vs = np.arange(h, dtype=np.float32) + 0.5
        uu, vv = np.meshgrid(us, vs)                     # vv 沿行 (图像向下)
        x_factor = (uu - w / 2.0) / f
        y_factor = (h / 2.0 - vv) / f
        self._xf = torch.from_numpy(x_factor).to(self.device).reshape(-1)
        self._yf = torch.from_numpy(y_factor).to(self.device).reshape(-1)

        pin = self.device.type == "cuda"
        self._staging = torch.empty(h * w, dtype=torch.float32, pin_memory=pin)
        self._depth_dev = torch.empty(h * w, dtype=torch.float32, device=self.device)
        # 归约结果 [n_inside, sum_x, sum_y, sum_z]
        self._res_host = torch.empty(4, dtype=torch.float32, pin_memory=pin)

        if self.timing:
            self._ev0 = torch.cuda.Event(enable_timing=True)
            self._ev1 = torch.cuda.Event(enable_timing=True)

    # ------------------------------------------------------------------ check
    def check(self, depth: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
              ee: np.ndarray, tcp: np.ndarray, goal: np.ndarray,
              v_world: np.ndarray) -> ConflictReport:
        """v_world: 当前指令的末端速度 (m/s, 世界系)。|v|≈0 时无走廊, 直接放行。"""
        cfg = self.cfg
        speed = float(np.linalg.norm(v_world))
        if speed < 1e-6:
            return _no_conflict()
        direction = v_world / speed
        length = max(float(cfg["corridor_l_min"]), speed * float(cfg["lookahead_s"]))

        stream_ctx = (torch.cuda.stream(self.stream)
                      if (self.stream is not None) else _NullCtx())
        with torch.no_grad(), stream_ctx:
            if self.timing:
                self._ev0.record()
            self._staging.copy_(torch.from_numpy(depth.reshape(-1)))
            self._depth_dev.copy_(self._staging, non_blocking=True)
            d = self._depth_dev

            R = torch.as_tensor(cam_mat, dtype=torch.float32, device=self.device)
            cpos = torch.as_tensor(cam_pos, dtype=torch.float32, device=self.device)
            a = torch.as_tensor(ee, dtype=torch.float32, device=self.device)
            tcp_t = torch.as_tensor(tcp, dtype=torch.float32, device=self.device)
            goal_t = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
            dir_t = torch.as_tensor(direction, dtype=torch.float32, device=self.device)

            valid = (d > 0.02) & (d < float(cfg["d_max"]))
            p_cam = torch.stack([self._xf * d, self._yf * d, -d], dim=1)  # (N,3)
            p = p_cam @ R.T + cpos                                        # 世界系

            valid &= p[:, 2] > float(cfg["table_z_filter"])
            # 自体屏蔽: 沿手轴从 TCP 一直盖到腕部上方 (link7/法兰也在深度图里)
            approach = tcp_t - a
            approach = approach / torch.linalg.norm(approach).clamp_min(1e-9)
            wrist_top = a - approach * float(cfg.get("self_mask_extend", 0.12))
            valid &= _dist_to_segment(p, wrist_top, tcp_t) > float(cfg["self_mask_radius"])
            valid &= torch.linalg.norm(p - goal_t, dim=1) > float(cfg["goal_mask_radius"])

            b = a + dir_t * length
            inside = (_dist_to_segment(p, a, b) < float(cfg["corridor_radius"])) & valid

            n = inside.sum()
            sums = (p * inside.unsqueeze(1)).sum(dim=0)
            res = torch.cat([n.reshape(1).to(torch.float32), sums])
            self._res_host.copy_(res, non_blocking=True)
            if self.timing:
                self._ev1.record()
            if self.stream is not None:
                self.stream.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        t_gpu = self._ev0.elapsed_time(self._ev1) if self.timing else float("nan")

        t0 = time.perf_counter_ns()
        n_pts = int(self._res_host[0].item())
        if n_pts < int(cfg["conflict_min_points"]):
            rep = _no_conflict(t_gpu)
            rep.t_cpu_us = (time.perf_counter_ns() - t0) / 1e3
            return rep

        centroid = (self._res_host[1:].numpy() / n_pts).astype(np.float64)
        delta = centroid - np.asarray(ee, dtype=np.float64)
        dist = float(np.linalg.norm(delta))
        bearing = _signed_bearing_deg(direction, delta)
        ctx = self._context(bearing, dist)
        t_cpu = (time.perf_counter_ns() - t0) / 1e3
        return ConflictReport(True, n_pts, centroid, dist, bearing, ctx, t_gpu, t_cpu)

    # -------------------------------------------------------------- internals
    def _context(self, bearing_deg: float, dist: float) -> str:
        half = float(self.cfg["front_half_angle_deg"])
        if abs(bearing_deg) <= half:
            d = "front"
        elif bearing_deg > 0:
            d = "left"
        else:
            d = "right"
        bucket = "near" if dist < float(self.cfg["near_far_split"]) else "far"
        return f"{d}_{bucket}"


def _dist_to_segment(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """点集 p (N,3) 到线段 [a,b] 的距离。a==b 时退化为点距。"""
    ab = b - a
    denom = torch.dot(ab, ab).clamp_min(1e-12)
    t = ((p - a) @ ab / denom).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(1) * ab
    return torch.linalg.norm(p - closest, dim=1)


def _signed_bearing_deg(direction: np.ndarray, delta: np.ndarray) -> float:
    """指令方向 -> 冲突质心的带符号水平方位角; 左侧为正。

    指令几乎纯竖直 (下降/上撤) 时水平方位无意义, 视为 front (0 度)。
    """
    v_xy = np.asarray(direction[:2], dtype=np.float64)
    d_xy = np.asarray(delta[:2], dtype=np.float64)
    if np.linalg.norm(v_xy) < 1e-3 or np.linalg.norm(d_xy) < 1e-6:
        return 0.0
    ang = math.atan2(v_xy[0] * d_xy[1] - v_xy[1] * d_xy[0],
                     float(v_xy @ d_xy))
    return math.degrees(ang)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
