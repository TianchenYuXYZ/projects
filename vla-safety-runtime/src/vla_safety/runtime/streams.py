"""CUDA stream 装备: 感知-策略解耦的单卡双 stream 实现。

文档的部署形态是 dual-T4 (GPU-A 跑 VLA, GPU-B 跑感知+监控)。本机只有
一张 RTX 3060, 采用同一思路的单卡变体 (对应文档引用的 VPEngine 路线):

  stream_main     VLA 解码 (GPU-A 语义, 默认优先级)
  stream_safety   深度反投影 + 走廊检测 (GPU-B 语义, 高优先级)
  stream_recovery 恢复通道传输 (高优先级, cudaStreamCreateWithPriority)

CUDA 的 stream priority 保证: 即便 stream_main 塞满 VLA decode kernel,
高优先级 stream 的 kernel 会被 scheduler 优先调度 —— 这是 <5ms 恢复
延迟在共享 GPU 上成立的硬件机制。device_a / device_b 参数保留双卡
形态: 真有第二张卡时把 safety/recovery 放到 cuda:1 即可。
"""
from __future__ import annotations

import threading

import torch


class StreamRig:
    def __init__(self, device_a: str = "cuda:0", device_b: str | None = None,
                 safety_priority: bool = True):
        self.device_a = torch.device(device_a)
        self.device_b = torch.device(device_b) if device_b else self.device_a
        lo, hi = torch.cuda.get_stream_priority_range() \
            if hasattr(torch.cuda, "get_stream_priority_range") else (-1, 0)
        prio = lo if safety_priority else 0          # lo = 最高优先级 (数值最小)
        self.priority_used = prio
        self.stream_main = torch.cuda.Stream(device=self.device_a, priority=0)
        self.stream_safety = torch.cuda.Stream(device=self.device_b, priority=prio)
        self.stream_recovery = torch.cuda.Stream(device=self.device_b, priority=prio)


class DecodeLoadGenerator:
    """后台线程在 stream_main 上持续提交 VLA 解码负载 (GPU 争用源)。

    每轮提交后同步一次, 保持恒定的在飞深度 (不同步会让 kernel 队列
    无限增长, 把所有测量都变成排队时间)。
    """

    def __init__(self, model, device: torch.device, stream: torch.cuda.Stream,
                 repeats: int, img_size: int = 96):
        self.model = model
        self.stream = stream
        self.repeats = repeats
        self._img = torch.randn(1, 3, img_size, img_size, device=device)
        self._prop = torch.randn(1, 4, device=device)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.iterations = 0

    def __enter__(self) -> "DecodeLoadGenerator":
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return False

    def _worker(self) -> None:
        while not self._stop.is_set():
            with torch.cuda.stream(self.stream):
                self.model.decode_load(self._img, self._prop, self.repeats)
            self.stream.synchronize()
            self.iterations += 1
