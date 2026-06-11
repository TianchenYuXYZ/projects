"""R3M cosine similarity 数据过滤: 语义把关人。

anchors = demo 关键帧的 L2 归一化 R3M 特征。
帧得分  = 与所有 anchor 的最大余弦相似度。
场景保留条件 = 该场景抽样帧得分的均值 >= tau。

tau 标定: recall 必须在 *视觉扰动分布* 上度量 —— 同一批关键帧位姿在
K 个随机化场景下重渲染 (语义不变, 只换视觉), 这些 "已知语义一致" 样本
对 anchors 的得分分布取 (1-recall) 分位点。直接在原 demo 关键帧上做留一
会因同场景外观把 tau 抬到 ~0.999, 把所有增广样本全杀掉。
"""
from __future__ import annotations

import numpy as np


class CosineFilter:
    def __init__(self, anchors: np.ndarray, tau: float):
        """anchors: (K, D), 已 L2 归一化。"""
        self.anchors = anchors.astype(np.float32)
        self.tau = float(tau)

    def frame_scores(self, feats: np.ndarray) -> np.ndarray:
        """(N, D) 归一化特征 -> (N,) 每帧与最近 anchor 的余弦相似度。"""
        return (feats @ self.anchors.T).max(axis=1)

    def keep_mask_by_scene(self, feats: np.ndarray,
                           scene_ids: np.ndarray) -> np.ndarray:
        """场景级过滤: 场景内帧得分均值低于 tau 则整个场景剔除。返回帧级布尔掩码。

        用均值而非最小值: 单帧噪声不应一票否决, 语义漂移体现在整条轨迹
        得分整体偏低。
        """
        scores = self.frame_scores(feats)
        keep = np.ones(len(feats), dtype=bool)
        for sid in np.unique(scene_ids):
            m = scene_ids == sid
            if scores[m].mean() < self.tau:
                keep[m] = False
        return keep

    @staticmethod
    def calibrate(anchor_feats: np.ndarray, nuisance_feats: np.ndarray,
                  recall: float = 0.95) -> float:
        """nuisance_feats: 与 anchors 同语义但视觉随机化的标定样本特征
        (K 场景 x n 关键帧)。tau = 其对 anchors 得分的 (1-recall) 分位点,
        即放过 95% 的 "已知语义一致" 样本。"""
        assert len(nuisance_feats) >= 10, "标定样本太少"
        scores = (nuisance_feats @ anchor_feats.T).max(axis=1)
        return float(np.quantile(scores, 1.0 - recall))


def select_keyframes(n_frames: int, n_keyframes: int) -> np.ndarray:
    """沿轨迹均匀取关键帧索引 (含首尾)。"""
    return np.linspace(0, n_frames - 1, n_keyframes).round().astype(int)
