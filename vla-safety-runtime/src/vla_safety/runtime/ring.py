"""seqlock 最新值槽 (Python 写端) —— 跨进程恢复通道的最后一跳。

为什么是 seqlock 而不是队列: 控制环要的是"最新指令", 不是积压历史。
单写者 + 任意读者, 写端无等待, 读端重试直到读到一致快照。

共享内存布局 (64 字节, 与 cpp/seqlock_ring.hpp 严格一致):
  offset 0   uint32  seq        奇数=写入中, 偶数=稳定
  offset 4   uint32  magic      0x564C4131 'VLA1'
  offset 8   int64   qpc_write  写入时刻的 QueryPerformanceCounter 原始 tick
  offset 16  uint64  cmd_id     单调递增指令号
  offset 24  float32 v[4]       归一化指令 (v_xyz, grip)
  offset 40  uint32  source     0=vla, 1=recovery, 2=qp
  offset 44  uint32  flags
  offset 48  16B     pad        (凑满一个 cache line)

时戳用两进程都能读的原始 QPC tick (同一 boot 内同源), 跨进程延迟
= (qpc_read - qpc_write) / qpc_freq, 不依赖任何解释器内部的时钟基准。

内存序: 仅支持 x86-64 (TSO)。CPython 的 mmap 写不提供原子性保证,
但单写者 + x86 强序 + C++ 读端的 acquire fence 足以保证本协议正确;
读端 (C++) 通过 seq 奇偶 + 前后一致校验剔除撕裂读。
"""
from __future__ import annotations

import ctypes
import mmap
import struct

MAGIC = 0x564C4131
SLOT_SIZE = 64
_PAYLOAD_FMT = "<qQ4fII"          # qpc_write, cmd_id, v[4], source, flags
_PAYLOAD_OFFSET = 8

_kernel32 = ctypes.windll.kernel32


def qpc_now() -> int:
    v = ctypes.c_longlong(0)
    _kernel32.QueryPerformanceCounter(ctypes.byref(v))
    return v.value


def qpc_freq() -> int:
    v = ctypes.c_longlong(0)
    _kernel32.QueryPerformanceFrequency(ctypes.byref(v))
    return v.value


class SeqlockWriter:
    """单写者。tagname 形如 'Local\\\\vla_safety_ring' (Windows 命名共享内存)。"""

    def __init__(self, tagname: str):
        self.mm = mmap.mmap(-1, SLOT_SIZE, tagname=tagname)
        self._cmd_id = 0
        self._seq = 0
        # 初始化: seq=0 (稳定), magic 就位
        self.mm[0:4] = struct.pack("<I", 0)
        self.mm[4:8] = struct.pack("<I", MAGIC)

    def write(self, v4, source: int, flags: int = 0) -> int:
        """写入一条指令, 返回 cmd_id。"""
        self._cmd_id += 1
        self._seq += 1
        self.mm[0:4] = struct.pack("<I", self._seq)          # 奇数: 写入中
        payload = struct.pack(
            _PAYLOAD_FMT, qpc_now(), self._cmd_id,
            float(v4[0]), float(v4[1]), float(v4[2]), float(v4[3]),
            source, flags,
        )
        self.mm[_PAYLOAD_OFFSET:_PAYLOAD_OFFSET + len(payload)] = payload
        self._seq += 1
        self.mm[0:4] = struct.pack("<I", self._seq)          # 偶数: 稳定
        return self._cmd_id

    def close(self) -> None:
        self.mm.close()


class SeqlockReader:
    """Python 读端 (测试/自检用; 生产读端是 C++ 控制环)。"""

    def __init__(self, tagname: str):
        self.mm = mmap.mmap(-1, SLOT_SIZE, tagname=tagname)

    def read(self, max_retries: int = 64) -> dict | None:
        """读取一致快照; 写入风暴下重试耗尽返回 None。"""
        for _ in range(max_retries):
            s0 = struct.unpack("<I", self.mm[0:4])[0]
            if s0 & 1:
                continue
            payload = bytes(self.mm[_PAYLOAD_OFFSET:_PAYLOAD_OFFSET + 40])
            s1 = struct.unpack("<I", self.mm[0:4])[0]
            if s0 == s1:
                qpc_w, cmd_id, v0, v1, v2, v3, src, flags = struct.unpack(
                    _PAYLOAD_FMT, payload)
                return {"qpc_write": qpc_w, "cmd_id": cmd_id,
                        "v": (v0, v1, v2, v3), "source": src,
                        "flags": flags, "seq": s0}
        return None

    def close(self) -> None:
        self.mm.close()
