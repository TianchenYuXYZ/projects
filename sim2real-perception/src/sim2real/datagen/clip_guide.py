"""CLIP 引导的纹理筛选: 用 image-text 相似度剔除语义离谱的随机纹理。

对每个表面 (table / floor) 给定文本 prompt, 对候选纹理打分, 保留前
keep_ratio 比例 —— 即文档中的 "CLIP-guided texture/scene generation":
随机化要广, 但不能离谱到把桌面随机成不可能出现的东西。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class CLIPGuide:
    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k", device: str = "cuda"):
        import open_clip

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def score(self, image_paths: list[Path], prompt: str,
              batch_size: int = 64) -> np.ndarray:
        """返回每张纹理与 prompt 的余弦相似度。"""
        text = self.tokenizer([prompt]).to(self.device)
        tfeat = self.model.encode_text(text)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

        sims = []
        for i in range(0, len(image_paths), batch_size):
            batch = torch.stack([
                self.preprocess(Image.open(p).convert("RGB"))
                for p in image_paths[i: i + batch_size]
            ]).to(self.device)
            ifeat = self.model.encode_image(batch)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            sims.append((ifeat @ tfeat.T).squeeze(-1).cpu().numpy())
        return np.concatenate(sims)

    def curate(self, image_paths: list[Path], prompt: str,
               keep_ratio: float) -> list[Path]:
        """按 CLIP 得分保留前 keep_ratio 的纹理。"""
        if not image_paths:
            return []
        sims = self.score(image_paths, prompt)
        k = max(1, int(len(image_paths) * keep_ratio))
        order = np.argsort(-sims)[:k]
        return [image_paths[i] for i in order]
