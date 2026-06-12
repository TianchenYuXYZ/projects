"""资产搭建 + 腕部深度相机标定注入 + 几何自检。

流程:
1. 从 sim2real-perception 复制 menagerie Panda 资产 (本地已有, 不走网络)
2. 两遍标定法把 wrist_depth 相机注入 hand body:
   pass-1 用占位位姿编译, 读出 ready 位姿下 hand 的世界位姿;
   解算 "前下视 50°、世界系悬停于手后上方" 对应的 hand 局部位姿;
   pass-2 重写 panda_safety.xml 并复检 (世界位姿误差 < 1e-5)
3. 反投影自检: 有障碍场景必须报冲突且质心落在障碍表面附近;
   无障碍场景前向运动必须无冲突 (验证自体/桌面/目标三重过滤)
4. 输出 RGB / 深度 sanity 渲染图与 calib 摘要
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np
from PIL import Image

import vla_safety  # noqa: F401  (UTF-8 bootstrap)
from vla_safety.common import PROJECT_ROOT, SceneSpec, load_yaml, save_json

CAM_TILT_DEG = 50.0                 # 前下视角
# 挂点在腕部正上方稍前: 夹爪落在画面底缘 (~85-90° 俯角, fovy 90 的边缘),
# 前向走廊 (20-60° 俯角) 视线无遮挡。挂在手后上方会让 hand 正好占住视野中心。
CAM_OFFSET_WORLD = np.array([0.02, 0.0, 0.15])    # 相对 hand 原点的世界系偏移
CAM_FOVY = 90.0
CAM_LINE_TAG = '<camera name="wrist_depth"'


def desired_cam_world() -> np.ndarray:
    """期望的相机世界旋转 (列 = x右, y上, z后; 视线沿 -z)。"""
    tilt = np.radians(CAM_TILT_DEG)
    look = np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
    up_world = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(look, up_world)
    x_cam /= np.linalg.norm(x_cam)
    z_cam = -look
    y_cam = np.cross(z_cam, x_cam)
    return np.column_stack([x_cam, y_cam, z_cam])


def patch_panda(panda_dir: Path, pos: np.ndarray, quat: np.ndarray) -> None:
    src = (panda_dir / "panda.xml").read_text(encoding="utf-8")
    lines = src.splitlines()
    out = []
    injected = False
    for line in lines:
        out.append(line)
        if '<body name="hand"' in line and not injected:
            indent = " " * (len(line) - len(line.lstrip()) + 2)
            # menagerie panda.xml 没有任何 site; 注入 EE 参考 site (hand 原点)
            out.append(f'{indent}<site name="attachment_site" pos="0 0 0" '
                       f'size="0.001" rgba="0 0 0 0"/>')
            out.append(
                f'{indent}<camera name="wrist_depth" '
                f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
                f'quat="{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}" '
                f'fovy="{CAM_FOVY}"/>'
            )
            injected = True
    if not injected:
        raise RuntimeError("panda.xml 中未找到 hand body, 无法注入相机")
    (panda_dir / "panda_safety.xml").write_text("\n".join(out) + "\n",
                                                encoding="utf-8")


def hand_world_pose() -> tuple[np.ndarray, np.ndarray]:
    """编译标称场景, ready keyframe 下 hand 的 (世界位置, 世界旋转)。"""
    from vla_safety.env import scene as scene_mod
    model = scene_mod.compile_scene(SceneSpec.nominal())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    mujoco.mj_forward(model, data)
    hid = model.body("hand").id
    return data.xpos[hid].copy(), data.xmat[hid].reshape(3, 3).copy()


def main() -> None:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    panda_src = (PROJECT_ROOT / cfg["paths"]["panda_src"]).resolve()
    panda_dst = PROJECT_ROOT / cfg["paths"]["panda_dst"]

    # ---------------------------------------------------------------- 1) 复制
    if not panda_dst.exists():
        if not panda_src.exists():
            raise FileNotFoundError(f"Panda 资产源不存在: {panda_src}")
        print(f"[1/4] 复制 Panda 资产 {panda_src} -> {panda_dst}")
        shutil.copytree(panda_src, panda_dst,
                        ignore=shutil.ignore_patterns("*.png", "_menagerie_tmp"))
    else:
        print(f"[1/4] Panda 资产已存在: {panda_dst}")

    # ------------------------------------------------------- 2) 两遍标定注入
    print("[2/4] pass-1: 占位相机, 求 ready 位姿下 hand 世界位姿")
    patch_panda(panda_dst, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    p_hand, r_hand = hand_world_pose()

    r_des = desired_cam_world()
    cam_world_pos = p_hand + CAM_OFFSET_WORLD
    local_pos = r_hand.T @ (cam_world_pos - p_hand)
    local_rot = r_hand.T @ r_des
    local_quat = np.empty(4)
    mujoco.mju_mat2Quat(local_quat, local_rot.ravel())

    print("        pass-2: 写入解算位姿并复检")
    patch_panda(panda_dst, local_pos, local_quat)

    from vla_safety.env import scene as scene_mod
    model = scene_mod.compile_scene(SceneSpec.nominal())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    mujoco.mj_forward(model, data)
    cid = model.camera("wrist_depth").id
    err_pos = float(np.abs(data.cam_xpos[cid] - cam_world_pos).max())
    err_rot = float(np.abs(data.cam_xmat[cid].reshape(3, 3) - r_des).max())
    assert err_pos < 1e-5 and err_rot < 1e-5, \
        f"相机标定复检失败: pos误差 {err_pos:.2e}, rot误差 {err_rot:.2e}"
    print(f"        复检通过: pos误差 {err_pos:.2e}, rot误差 {err_rot:.2e}")

    # ------------------------------------------------------- 3) 动态自检
    # 静态 ready 位姿下手腕遮挡向下视线, 触发必须在工作高度验证:
    #   a) 巡航高度逼近障碍 -> 必须在撞上之前 (>= 0.06 m) 报冲突, 且质心在障碍表面
    #   b) 无障碍跑完整 降高/巡航/下降 三相位 -> 全程零误报
    print("[3/4] 走廊检测动态自检 (CPU, 默认参数)")
    from vla_safety.env import ManipSafetyEnv
    from vla_safety.perception.depth_safety import DepthSafetyChecker

    checker = DepthSafetyChecker(dict(cfg["safety"]), cfg["env"]["depth_size"],
                                 90.0, device="cpu")
    cube = (0.52, 0.0)
    obst = (0.42, 0.01)
    travel_z = float(cfg["env"]["travel_ee_z"])

    env = ManipSafetyEnv(SceneSpec(cube_pos=cube, obstacle_pos=obst),
                         render_size=cfg["env"]["render_size"],
                         depth_size=cfg["env"]["depth_size"],
                         workspace=cfg["env"]["workspace"])
    env.reset()
    ee_start = env.ee_pos.copy()
    env.set_command(np.array([0, 0, -1.0, -1.0]))
    for _ in range(400):
        env.tick()
        if abs(env.ee_pos[2] - travel_z) < 0.005:
            break
    to_xy = np.array(obst) - env.ee_pos[:2]
    u = to_xy / np.linalg.norm(to_xy)
    trig_dist, rep = None, None
    for step in range(600):
        env.set_command(np.array([u[0] * 0.8, u[1] * 0.8, 0, -1.0]))
        env.tick()
        if step % 3 == 0:
            depth, cpos, cmat = env.render_depth()
            rep = checker.check(depth, cpos, cmat, env.ee_pos, env.tcp_pos,
                                env.cube_pos, env.command[:3] * 0.15)
            d_now = float(np.linalg.norm(np.array(obst) - env.ee_pos[:2]))
            if rep.conflict:
                trig_dist = d_now
                break
            if d_now < 0.05:
                break
    assert trig_dist is not None and trig_dist >= 0.06, \
        f"逼近障碍未及时触发 (触发距离 {trig_dist})"
    err_centroid = float(np.linalg.norm(rep.centroid[:2] - np.array(obst)))
    assert err_centroid < 0.08, f"冲突质心偏离障碍轴线 {err_centroid:.3f} m"
    print(f"        逼近触发: 距障碍 {trig_dist:.3f} m, n={rep.n_points}, "
          f"context={rep.context}, 质心-轴线偏差 {err_centroid * 100:.1f} cm")

    rgb = env.render_rgb()
    depth_img, _, _ = env.render_depth()
    depth_vis = np.clip(depth_img / 1.2 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(PROJECT_ROOT / "assets" / "sanity_rgb.png")
    Image.fromarray(depth_vis).save(PROJECT_ROOT / "assets" / "sanity_depth.png")
    env.close()

    env2 = ManipSafetyEnv(SceneSpec(cube_pos=cube, obstacle_pos=None),
                          render_size=cfg["env"]["render_size"],
                          depth_size=cfg["env"]["depth_size"],
                          workspace=cfg["env"]["workspace"])
    env2.reset()
    fp = 0
    cube_xy = np.array(cube)
    for phase, n_max in [("descend_travel", 400), ("cruise", 500), ("descend", 400)]:
        for _ in range(n_max):
            ee = env2.ee_pos
            if phase == "descend_travel":
                if abs(ee[2] - travel_z) < 0.005:
                    break
                env2.set_command(np.array([0, 0, -1.0, -1.0]))
            elif phase == "cruise":
                d = cube_xy - ee[:2]
                if np.linalg.norm(d) < 0.01:
                    break
                u2 = d / max(np.linalg.norm(d), 1e-9)
                env2.set_command(np.array([u2[0] * 0.8, u2[1] * 0.8, 0, -1.0]))
            else:
                if ee[2] < float(cfg["env"]["grasp_ee_z"]) + 0.01:
                    break
                env2.set_command(np.array([0, 0, -0.8, -1.0]))
            env2.tick()
            if env2.tick_count % 3 == 0:
                depth, cpos, cmat = env2.render_depth()
                r = checker.check(depth, cpos, cmat, env2.ee_pos, env2.tcp_pos,
                                  env2.cube_pos, env2.command[:3] * 0.15)
                fp += int(r.conflict)
    assert fp == 0, f"无障碍三相位误报 {fp} 次 —— 自体/桌面/目标过滤不足"
    print("        无障碍三相位 (降高/巡航/下降): 零误报")
    env2.close()

    # ---------------------------------------------------------------- 4) 摘要
    import torch
    summary = {
        "ee_start": [round(float(x), 4) for x in ee_start],
        "cam_world_pos": [round(float(x), 4) for x in cam_world_pos],
        "cam_local_pos": [round(float(x), 6) for x in local_pos],
        "cam_local_quat": [round(float(x), 8) for x in local_quat],
        "cam_tilt_deg": CAM_TILT_DEG,
        "calib_err": {"pos": err_pos, "rot": err_rot},
        "selftest": {"obstacle_points": rep.n_points,
                     "centroid_err_m": err_centroid},
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(summary, PROJECT_ROOT / "assets" / "calib.json")
    print(f"[4/4] 完成。EE 起点 {summary['ee_start']}, "
          f"相机 {summary['cam_world_pos']}; 摘要 -> assets/calib.json")


if __name__ == "__main__":
    main()
