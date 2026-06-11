"""CosineFilter 阈值标定与场景级过滤逻辑。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sim2real.perception.filter import CosineFilter, select_keyframes


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_calibrate_recall():
    """tau 必须让 >= recall 比例的扰动标定样本通过。"""
    rng = np.random.default_rng(0)
    base = _unit(rng.normal(size=(1, 64)))
    anchors = _unit(base + 0.05 * rng.normal(size=(12, 64))).astype(np.float32)
    # 标定集 = 同语义 + 视觉扰动 (偏离更大)
    nuisance = _unit(base + 0.4 * rng.normal(size=(100, 64))).astype(np.float32)
    tau = CosineFilter.calibrate(anchors, nuisance, recall=0.95)
    scores = (nuisance @ anchors.T).max(axis=1)
    assert (scores >= tau).mean() >= 0.95
    # 正交样本 (语义漂移) 必须被拒
    drift = _unit(rng.normal(size=(50, 64))).astype(np.float32)
    f = CosineFilter(anchors, tau)
    assert (f.frame_scores(drift) < tau).mean() > 0.5


def test_scene_level_filtering():
    rng = np.random.default_rng(1)
    anchors = _unit(rng.normal(size=(5, 32))).astype(np.float32)
    f = CosineFilter(anchors, tau=0.99)
    # 场景 0: 全部帧 = anchor[0] (相似度 1) -> 保留
    good = np.tile(anchors[0], (4, 1))
    # 场景 1: 有一帧正交 (相似度 ~0) -> 整场景剔除
    ortho = _unit(rng.normal(size=(1, 32))).astype(np.float32)
    ortho -= (ortho @ anchors.T) @ anchors  # 去除 anchor 分量
    ortho = _unit(ortho)
    bad = np.vstack([np.tile(anchors[1], (3, 1)), ortho])
    feats = np.vstack([good, bad]).astype(np.float32)
    sids = np.array([0] * 4 + [1] * 4)
    keep = f.keep_mask_by_scene(feats, sids)
    assert keep[:4].all() and not keep[4:].any()


def test_select_keyframes():
    kf = select_keyframes(100, 12)
    assert len(kf) == 12 and kf[0] == 0 and kf[-1] == 99
    assert (np.diff(kf) > 0).all()
