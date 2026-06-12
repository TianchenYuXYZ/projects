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

# ---- 相机 1: wrist_depth (安全走廊感知) ------------------------------------
# 挂点在腕部正上方稍前, 65° 前下视: 视野覆盖俯角 20-110°, 近距触发时障碍
# 在俯角 33-84°。挂手后上方会让 hand 占住视野中心 (实测教训)。
CAM_TILT_DEG = 65.0
CAM_OFFSET_WORLD = np.array([0.02, 0.0, 0.15])
CAM_FOVY = 90.0

# ---- 相机 2: wrist_rgb (VLA 抓取对中视角) -----------------------------------
# 夹爪正下方的方块从任何 "近侧" 挂点看都被手指/手掌挡住 (实测: 上方挂点
# 0 红像素)。手指沿一条水平轴开合, 唯一无遮挡视线是沿 *指缝方向*
# (垂直于开合轴) 横置 + 近垂直俯视 —— 视线从两指之间的间隙穿过。
# 开合轴在编译后的模型里读 (随 ready 位姿 yaw 而定), 不写死。
# 挂点必须低于手掌平面 (raise<0), 否则手掌轮廓会吞掉向下视线 (实测教训:
# raise=+0.04 时方块在悬停/下降相位完全消失在手部剪影后)。横置 0.10 +
# 掌下 0.02 ≈ RealSense D405 在 Franka 手上的标准装法。
CAM2_LATERAL = 0.10                # 沿指缝方向的横向偏移 (m)
CAM2_RAISE = -0.02                 # 相对 hand 原点的高度 (负 = 掌下)
CAM2_LOOK_DOWN_Z = 0.18            # 注视点: hand 原点正下方 0.18m (TCP 区域)
CAM2_FOVY = 90.0


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


def patch_panda(panda_dir: Path, pos: np.ndarray, quat: np.ndarray,
                pos2: np.ndarray | None = None,
                quat2: np.ndarray | None = None) -> None:
    """注入 attachment_site + wrist_depth (+ 可选 wrist_rgb) 到 hand body。"""
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
            if pos2 is not None:
                out.append(
                    f'{indent}<camera name="wrist_rgb" '
                    f'pos="{pos2[0]:.6f} {pos2[1]:.6f} {pos2[2]:.6f}" '
                    f'quat="{quat2[0]:.8f} {quat2[1]:.8f} {quat2[2]:.8f} '
                    f'{quat2[3]:.8f}" fovy="{CAM2_FOVY}"/>'
                )
            injected = True
    if not injected:
        raise RuntimeError("panda.xml 中未找到 hand body, 无法注入相机")
    (panda_dir / "panda_safety.xml").write_text("\n".join(out) + "\n",
                                                encoding="utf-8")


def hand_world_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ready keyframe 下 hand 的 (世界位置, 世界旋转, 手指开合轴-世界系)。"""
    from vla_safety.env import scene as scene_mod
    model = scene_mod.compile_scene(SceneSpec.nominal())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    mujoco.mj_forward(model, data)
    hid = model.body("hand").id
    lf = data.xpos[model.body("left_finger").id]
    rf = data.xpos[model.body("right_finger").id]
    f_axis = lf - rf
    f_axis[2] = 0.0
    n = np.linalg.norm(f_axis)
    f_axis = f_axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    return data.xpos[hid].copy(), data.xmat[hid].reshape(3, 3).copy(), f_axis


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
    print("[2/4] pass-1: 占位相机, 求 ready 位姿下 hand 世界位姿与手指开合轴")
    patch_panda(panda_dst, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    p_hand, r_hand, f_axis = hand_world_pose()

    # 相机 1: wrist_depth (前下视)
    r_des = desired_cam_world()
    cam_world_pos = p_hand + CAM_OFFSET_WORLD
    local_pos = r_hand.T @ (cam_world_pos - p_hand)
    local_rot = r_hand.T @ r_des
    local_quat = np.empty(4)
    mujoco.mju_mat2Quat(local_quat, local_rot.ravel())

    # 相机 2: wrist_rgb (指缝方向横置 + 近垂直俯视 TCP)
    up = np.array([0.0, 0.0, 1.0])
    gap_dir = np.cross(up, f_axis)
    gap_dir /= np.linalg.norm(gap_dir)
    cam2_world = p_hand + gap_dir * CAM2_LATERAL + np.array([0, 0, CAM2_RAISE])
    look_target = p_hand + np.array([0, 0, -CAM2_LOOK_DOWN_Z])
    look = look_target - cam2_world
    look /= np.linalg.norm(look)
    x2 = np.cross(look, up)
    x2 /= np.linalg.norm(x2)
    z2 = -look
    y2 = np.cross(z2, x2)
    r2_des = np.column_stack([x2, y2, z2])
    local2_pos = r_hand.T @ (cam2_world - p_hand)
    local2_rot = r_hand.T @ r2_des
    local2_quat = np.empty(4)
    mujoco.mju_mat2Quat(local2_quat, local2_rot.ravel())

    print("        pass-2: 写入解算位姿并复检 (双相机)")
    patch_panda(panda_dst, local_pos, local_quat, local2_pos, local2_quat)

    from vla_safety.env import scene as scene_mod
    model = scene_mod.compile_scene(SceneSpec.nominal())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    mujoco.mj_forward(model, data)
    err = []
    for name, want_p, want_r in [("wrist_depth", cam_world_pos, r_des),
                                 ("wrist_rgb", cam2_world, r2_des)]:
        cid = model.camera(name).id
        err.append(float(np.abs(data.cam_xpos[cid] - want_p).max()))
        err.append(float(np.abs(data.cam_xmat[cid].reshape(3, 3) - want_r).max()))
    err_pos, err_rot = max(err[0], err[2]), max(err[1], err[3])
    assert err_pos < 1e-5 and err_rot < 1e-5, \
        f"相机标定复检失败: pos误差 {err_pos:.2e}, rot误差 {err_rot:.2e}"
    print(f"        复检通过 (双相机): pos误差 {err_pos:.2e}, rot误差 {err_rot:.2e}")

    # --------------------------------------------- 2.5) 腕视角可视性自检
    # wrist_rgb 的全部价值在于下降/悬停时能看到夹爪正下方的方块 (红色),
    # 视线必须从指缝穿过。逐高度检查红色像素数。
    print("[2.5/4] wrist_rgb 指缝可视性自检")
    from vla_safety.env import ManipSafetyEnv

    def red_pixels(img: np.ndarray) -> int:
        # 比例判据: 方块在手臂阴影中是暗红 (~RGB 85,24,20), 绝对阈值会漏;
        # 桌面是暖皮色 (r-g 差大但比例小), 差值阈值会误。r>2g 且 r>2b 两者全分开。
        r = img[:, :, 0].astype(int)
        g = img[:, :, 1].astype(int)
        b = img[:, :, 2].astype(int)
        return int(((r > 2 * g) & (r > 2 * b) & (r > 40)).sum())

    envv = ManipSafetyEnv(SceneSpec(cube_pos=(0.50, 0.05)),
                          render_size=cfg["env"]["render_size"],
                          depth_size=cfg["env"]["depth_size"],
                          workspace=cfg["env"]["workspace"])
    envv.reset()
    vis = {}
    for tz in (0.30, 0.21, 0.15, 0.125):
        for _ in range(900):
            ee = envv.ee_pos
            dxy = np.array([0.50, 0.05]) - ee[:2]
            if abs(ee[2] - tz) < 0.004 and np.linalg.norm(dxy) < 0.006:
                break
            v = np.zeros(4)
            v[3] = -1.0
            if np.linalg.norm(dxy) >= 0.006:
                u = dxy / np.linalg.norm(dxy)
                v[0], v[1] = u[0] * 0.6, u[1] * 0.6
            v[2] = float(np.clip((tz - ee[2]) * 8, -0.8, 0.8))
            envv.set_command(v)
            envv.tick()
        vis[tz] = red_pixels(envv.render_wrist_rgb())
    print(f"        红色像素 (z=0.30/0.21/0.15/0.125): "
          f"{[vis[k] for k in (0.30, 0.21, 0.15, 0.125)]}")
    assert vis[0.21] >= 20 and vis[0.15] >= 20, \
        f"悬停/下降相位腕视角看不到方块: {vis} —— 指缝视线被遮挡"
    Image.fromarray(envv.render_wrist_rgb()).save(
        PROJECT_ROOT / "assets" / "sanity_wrist.png")
    envv.close()

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
