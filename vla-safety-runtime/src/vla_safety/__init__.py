"""VLA Runtime Safety: 感知-策略解耦 + MAB Oracle Monitor + 低延迟恢复通道。

复现 research_experience_breakdown2.docx 描述的系统:
  - RT-2 风格 mini-VLA (动作离散化为 256-bin token, 自回归解码)
  - 深度感知几何安全检查 (独立 CUDA stream, 模拟 GPU-B)
  - training-free MAB Oracle Monitor (UCB1 / Thompson, arms = VLA 原生 token 序列)
  - 高优先级 CUDA stream + pinned memory + seqlock 共享内存的 <5ms 恢复通道
"""
import sys

# Windows 控制台默认 cp1252, 中文日志会崩; 统一强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

__version__ = "0.1.0"
