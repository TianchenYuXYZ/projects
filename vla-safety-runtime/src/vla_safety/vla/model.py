"""mini-VLA: RT-2 范式的最小可复现实例。

RT-2 把控制建模成 multimodal transformer 内的 translation 任务 ——
图像 (+指令) 进, 离散 action token 出, 自回归生成。这里保留范式骨架,
把 backbone 缩到 6GB 显卡可训的量级:

  CNN encoder (96x96 -> 36 patch token) + proprio token  ->  memory
  causal transformer decoder 自回归生成 4 个 action token (256-bin)

decode_load() 提供解码负载仿真: 重复跑 decoder 堆叠, 把单步推理拉长到
RT-2 量级 (数百 ms), 用于 GPU 争用 / stream 优先级实验 —— 报告中明确
标注这是负载仿真, 不是真实 55B 模型。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from vla_safety.common import ACTION_DIM, N_BINS

# 观测归一化常量 (proprio = ee_pos + 夹爪开度)
P_MEAN = np.array([0.45, 0.0, 0.20, 0.04], dtype=np.float32)
P_STD = np.array([0.20, 0.20, 0.15, 0.05], dtype=np.float32)


def preprocess(image_u8: np.ndarray, proprio: np.ndarray,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 HWC RGB + (4,) proprio -> 归一化 batch=1 张量。"""
    img = torch.from_numpy(np.ascontiguousarray(image_u8)).to(device)
    img = img.permute(2, 0, 1).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0)
    p = (proprio.astype(np.float32) - P_MEAN) / P_STD
    return img, torch.from_numpy(p).to(device).unsqueeze(0)


class ConvEncoder(nn.Module):
    def __init__(self, channels: list[int], d_model: int):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 3
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1),
                       nn.GroupNorm(8, c), nn.SiLU()]
            c_in = c
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(c_in, d_model, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.body(img))                  # (B, d, H', W')
        return h.flatten(2).transpose(1, 2)            # (B, H'*W', d)


class MiniVLA(nn.Module):
    BOS = N_BINS                                       # token 词表末位作 BOS

    def __init__(self, d_model=192, n_layers=4, n_heads=4, d_ff=384,
                 cnn_channels=(32, 64, 96, 128), dropout=0.1,
                 n_bins=N_BINS, action_dim=ACTION_DIM, img_size=96):
        super().__init__()
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.encoder = ConvEncoder(list(cnn_channels), d_model)
        n_patches = (img_size // (2 ** len(cnn_channels))) ** 2
        self.n_mem = n_patches + 1
        self.proprio_mlp = nn.Sequential(
            nn.Linear(4, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.mem_pos = nn.Parameter(torch.zeros(1, self.n_mem, d_model))
        self.tok_emb = nn.Embedding(n_bins + 1, d_model)
        self.act_pos = nn.Parameter(torch.zeros(1, action_dim, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.head = nn.Linear(d_model, n_bins)
        nn.init.trunc_normal_(self.mem_pos, std=0.02)
        nn.init.trunc_normal_(self.act_pos, std=0.02)
        mask = torch.triu(torch.full((action_dim, action_dim), float("-inf")), 1)
        self.register_buffer("causal_mask", mask, persistent=False)

    # ----------------------------------------------------------------- train
    def encode_memory(self, img: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        vis = self.encoder(img)                                   # (B, P, d)
        prop = self.proprio_mlp(proprio).unsqueeze(1)             # (B, 1, d)
        return torch.cat([vis, prop], dim=1) + self.mem_pos

    def forward(self, img: torch.Tensor, proprio: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        """teacher forcing。tokens: (B, action_dim) int64 -> logits (B, A, n_bins)。"""
        mem = self.encode_memory(img, proprio)
        bos = torch.full((tokens.shape[0], 1), self.BOS,
                         dtype=torch.long, device=tokens.device)
        tgt_in = torch.cat([bos, tokens[:, :-1]], dim=1)
        tgt = self.tok_emb(tgt_in) + self.act_pos
        h = self.decoder(tgt, mem, tgt_mask=self.causal_mask)
        return self.head(h)

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, img: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """贪心自回归解码 (确定性评测)。返回 (B, action_dim) int64。"""
        mem = self.encode_memory(img, proprio)
        b = img.shape[0]
        out = torch.full((b, self.action_dim), self.BOS,
                         dtype=torch.long, device=img.device)
        seq = torch.full((b, 1), self.BOS, dtype=torch.long, device=img.device)
        for t in range(self.action_dim):
            tgt = self.tok_emb(seq) + self.act_pos[:, : seq.shape[1]]
            mask = self.causal_mask[: seq.shape[1], : seq.shape[1]]
            h = self.decoder(tgt, mem, tgt_mask=mask)
            nxt = self.head(h[:, -1]).argmax(dim=-1)
            out[:, t] = nxt
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        return out

    @torch.no_grad()
    def decode_load(self, img: torch.Tensor, proprio: torch.Tensor,
                    repeats: int) -> None:
        """解码负载仿真: 重复跑 decoder, 占满当前 stream (模拟大 VLA 推理)。"""
        mem = self.encode_memory(img, proprio)
        b = img.shape[0]
        tgt_tok = torch.zeros((b, self.action_dim), dtype=torch.long,
                              device=img.device)
        tgt = self.tok_emb(tgt_tok) + self.act_pos
        for _ in range(repeats):
            h = self.decoder(tgt, mem, tgt_mask=self.causal_mask)
            tgt = tgt + 0.0 * h                       # 保持依赖链, 防止图优化剔除

    # ---------------------------------------------------------------- config
    @staticmethod
    def from_config(vla_cfg: dict, img_size: int) -> "MiniVLA":
        return MiniVLA(
            d_model=int(vla_cfg["d_model"]), n_layers=int(vla_cfg["n_layers"]),
            n_heads=int(vla_cfg["n_heads"]), d_ff=int(vla_cfg["d_ff"]),
            cnn_channels=tuple(vla_cfg["cnn_channels"]),
            dropout=float(vla_cfg["dropout"]), img_size=img_size,
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
