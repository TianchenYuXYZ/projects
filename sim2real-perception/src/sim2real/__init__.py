"""Sim-to-Real robust perception pipeline.

Python 侧负责训练链路: 仿真采 demo -> DR+CLIP 增广 -> R3M cosine 过滤
-> frozen backbone + LoRA + BC 训练 -> 评测 -> 导出 ONNX 部署包。
C++ 侧 (cpp/) 负责部署 runtime, 通过 deploy_bundle 契约对接。
"""

import sys

__version__ = "0.1.0"

# Windows 控制台默认 cp1252, 中文日志会崩; 统一强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
