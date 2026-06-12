"""VLAPolicy: 推理封装 —— obs 进, 归一化速度指令出。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vla_safety.vla.model import MiniVLA, preprocess
from vla_safety.vla.tokenizer import ActionTokenizer


class VLAPolicy:
    def __init__(self, model: MiniVLA, tokenizer: ActionTokenizer,
                 device: str = "cuda",
                 stream: "torch.cuda.Stream | None" = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.stream = stream            # 主推理 stream (GPU-A 语义)

    @torch.no_grad()
    def act(self, image_u8: np.ndarray, wrist_u8: np.ndarray,
            proprio: np.ndarray) -> np.ndarray:
        """贪心解码 -> (4,) 归一化指令。"""
        img, wimg, prop = preprocess(image_u8, wrist_u8, proprio, self.device)
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                tokens = self.model.generate(img, wimg, prop)
            self.stream.synchronize()
        else:
            tokens = self.model.generate(img, wimg, prop)
        return self.tokenizer.decode(tokens[0].cpu().numpy()).astype(np.float64)

    @staticmethod
    def load(ckpt_path: str | Path, vla_cfg: dict, img_size: int,
             device: str = "cuda",
             stream: "torch.cuda.Stream | None" = None) -> "VLAPolicy":
        model = MiniVLA.from_config(vla_cfg, img_size)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        return VLAPolicy(model, ActionTokenizer(), device=device, stream=stream)
