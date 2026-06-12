"""R3M cosine 过滤: demo 关键帧做 anchors, 视觉扰动标定集定 tau, 场景级筛数据。

标定集 = 同关键帧位姿 x K 个随机化场景重渲染 (语义不变, 视觉随机),
其对 anchors 的得分分布取 (1-recall) 分位点作为 tau。
产物 data/filter/: anchors.npy (L2 归一化), tau.json, keep_mask.npy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sim2real.common import PROJECT_ROOT, Trajectory, load_yaml
from sim2real.datagen.builder import load_shards
from sim2real.datagen.randomizer import DomainRandomizer, build_texture_pool
from sim2real.perception.backbone import PerceptionBackbone, embed_images
from sim2real.perception.filter import CosineFilter, select_keyframes
from sim2real.sim.env import ManipEnv

N_CALIB_SCENES = 20          # 标定集场景数 (独立于训练增广 seed)
CALIB_SEED_OFFSET = 99991


def render_calibration_set(demo: Trajectory, kf: np.ndarray,
                           dr_cfg: dict) -> np.ndarray:
    """同关键帧位姿在 K 个随机场景下重渲染。

    标定场景的方块必须固定在 demo 的位置: 回放的是 demo 的 qpos,
    若方块被位置抖动挪走, 渲染出的 "抓空气" 帧语义已错, 会把 tau 拉低。
    """
    from sim2real.common import SceneConfig

    pool = build_texture_pool(dr_cfg, None)
    randomizer = DomainRandomizer(dr_cfg, texture_pool=pool)
    rng = np.random.default_rng(int(dr_cfg["seed"]) + CALIB_SEED_OFFSET)
    nominal_cube = SceneConfig.nominal().cube_pos

    def sample() -> SceneConfig:
        s = randomizer.sample_scene(rng)
        s.cube_pos = nominal_cube
        return s

    env = ManipEnv(sample())
    frames = []
    for _ in range(N_CALIB_SCENES):
        env.reset(sample())
        frames.append(env.replay_render(demo.qpos, kf))
    env.close()
    return np.concatenate(frames)


def main() -> None:
    tcfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")
    dr_cfg = load_yaml(PROJECT_ROOT / "configs" / "dr.yaml")
    fcfg = tcfg["filter"]
    demo = Trajectory.load(PROJECT_ROOT / "data" / "demo.npz")
    data = load_shards(PROJECT_ROOT / "data" / "dr_dataset")

    backbone = PerceptionBackbone(tcfg["backbone"]["name"],
                                  tcfg["backbone"].get("weights_dir", "weights"))

    kf = select_keyframes(len(demo), int(fcfg["n_keyframes"]))
    anchors = embed_images(backbone, demo.images[kf])

    print(f"[filter] 渲染标定集 ({N_CALIB_SCENES} 场景 x {len(kf)} 关键帧) ...")
    calib_frames = render_calibration_set(demo, kf, dr_cfg)
    calib_feats = embed_images(backbone, calib_frames)
    tau = CosineFilter.calibrate(anchors, calib_feats,
                                 recall=float(fcfg["recall"]))
    cos_filter = CosineFilter(anchors, tau)
    calib_scores = cos_filter.frame_scores(calib_feats)
    print(f"[filter] anchors={len(anchors)}, tau={tau:.4f} "
          f"(标定得分 p5={np.percentile(calib_scores, 5):.4f} "
          f"p50={np.percentile(calib_scores, 50):.4f})")

    print("[filter] 计算 DR 数据集特征 ...")
    feats = embed_images(backbone, data["images"])
    keep = cos_filter.keep_mask_by_scene(feats, data["scene_ids"])

    n_scenes = len(np.unique(data["scene_ids"]))
    n_kept_scenes = len(np.unique(data["scene_ids"][keep]))
    print(f"[filter] 场景保留 {n_kept_scenes}/{n_scenes}, "
          f"帧保留 {keep.sum()}/{len(keep)}")

    out = PROJECT_ROOT / "data" / "filter"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "anchors.npy", anchors)
    np.save(out / "keep_mask.npy", keep)
    scores = cos_filter.frame_scores(feats)
    (out / "tau.json").write_text(json.dumps({
        "tau": tau, "recall": fcfg["recall"], "n_keyframes": len(anchors),
        "n_calib_scenes": N_CALIB_SCENES,
        "kept_scenes": int(n_kept_scenes), "total_scenes": int(n_scenes),
        "kept_frames": int(keep.sum()), "total_frames": int(len(keep)),
        "score_pcts": {p: float(np.percentile(scores, p))
                       for p in (1, 5, 25, 50, 75, 95, 99)},
    }), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
