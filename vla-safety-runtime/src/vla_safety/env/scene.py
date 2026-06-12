"""SceneSpec -> MJCF XML -> MjModel。

世界坐标约定: 桌面顶面 z = 0, Panda 基座在原点 (安装座向下延伸到地面)。
与训练分布的差异轴:
  - obstacle: 真实参与碰撞的立柱 (训练时从不出现, OOD-obstacle 轴)
  - 灯光/颜色/干扰物: 视觉域偏移 (OOD-visual 轴, 干扰物不参与碰撞)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
import numpy as np

from vla_safety.common import PROJECT_ROOT, SceneSpec

PANDA_DIR = PROJECT_ROOT / "assets" / "franka_emika_panda"
PANDA_XML = "panda_safety.xml"          # scripts/00 生成的带腕部深度相机的副本

TABLE_TOP_Z = 0.0
TABLE_HALF = (0.32, 0.42, 0.03)
TABLE_CENTER = (0.50, 0.0)
CUBE_HALF = 0.0125
FLOOR_Z = -0.76

# Franka 标准 ready 位姿: 末端竖直向下, 适合 top-down 抓取
HOME_QPOS_ARM = "0 -0.785398 0 -2.356194 0 1.570796 0.785398"
HOME_QPOS_FINGERS = "0.04 0.04"
KEY_CTRL = "0 -0.785398 0 -2.356194 0 1.570796 0.785398 255"

# 第三人称相机 (VLA 的 RGB 观测)
CAM0_POS = np.array([1.05, -0.45, 0.55])
CAM0_LOOKAT = np.array([0.45, 0.0, 0.08])
CAM0_FOVY = 50.0


def _camera_xyaxes(pos: np.ndarray, lookat: np.ndarray) -> str:
    """由相机位置和注视点计算 MJCF camera 的 xyaxes (右轴 + 上轴)。"""
    forward = lookat - pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return " ".join(f"{v:.6f}" for v in np.concatenate([right, up]))


def build_scene_xml(spec: SceneSpec) -> str:
    cube_rgba = " ".join(f"{v:.3f}" for v in spec.cube_rgba)
    table_rgba = " ".join(f"{v:.3f}" for v in spec.table_rgba)
    floor_rgba = " ".join(f"{v:.3f}" for v in spec.floor_rgba)
    cx, cy = spec.cube_pos

    # 灯光方向指向桌心
    lp = np.array(spec.light_pos, dtype=float)
    target = np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z])
    dirv = target - lp
    dirv = dirv / np.linalg.norm(dirv)
    d = float(spec.light_diffuse)
    light_xml = (
        f'<light name="light0" pos="{lp[0]} {lp[1]} {lp[2]}" '
        f'dir="{dirv[0]:.4f} {dirv[1]:.4f} {dirv[2]:.4f}" '
        f'diffuse="{d} {d} {d}" ambient="0.18 0.18 0.18" castshadow="true"/>'
    )

    # 障碍立柱: 真实碰撞体, 命名固定便于接触检测
    obstacle_xml = ""
    if spec.obstacle_pos is not None:
        ox, oy = spec.obstacle_pos
        hx, hy, hz = spec.obstacle_half
        orgba = " ".join(f"{v:.3f}" for v in spec.obstacle_rgba)
        obstacle_xml = (
            f'<geom name="obstacle" type="box" size="{hx} {hy} {hz}" '
            f'pos="{ox} {oy} {TABLE_TOP_Z + hz}" rgba="{orgba}" '
            f'friction="1.0 0.005 0.0001"/>'
        )

    distractors_xml = []
    for i, dt in enumerate(spec.distractors):
        x, y = dt["pos"]
        s = float(dt["size"])
        rgba = " ".join(f"{v:.3f}" for v in dt["rgba"])
        if dt["type"] == "box":
            g = (f'<geom name="distr{i}" type="box" size="{s} {s} {s}" '
                 f'pos="{x} {y} {TABLE_TOP_Z + s}"')
        elif dt["type"] == "sphere":
            g = (f'<geom name="distr{i}" type="sphere" size="{s}" '
                 f'pos="{x} {y} {TABLE_TOP_Z + s}"')
        else:  # cylinder
            g = (f'<geom name="distr{i}" type="cylinder" size="{s} {s}" '
                 f'pos="{x} {y} {TABLE_TOP_Z + s}"')
        # 干扰物纯视觉: 不参与碰撞 (OOD-visual 轴不改变可达性)
        distractors_xml.append(g + f' rgba="{rgba}" contype="0" conaffinity="0"/>')

    key_qpos = (f"{HOME_QPOS_ARM} {HOME_QPOS_FINGERS} "
                f"{cx} {cy} {TABLE_TOP_Z + CUBE_HALF} 1 0 0 0")

    return f"""
<mujoco model="vla_safety_scene">
  <include file="{PANDA_XML}"/>
  <statistic center="0.4 0 0.2" extent="1.2"/>
  <visual>
    <headlight diffuse="0.3 0.3 0.3" ambient="0.15 0.15 0.15" specular="0 0 0"/>
    <global offwidth="512" offheight="512"/>
  </visual>
  <worldbody>
    {light_xml}
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 {FLOOR_Z}"
          rgba="{floor_rgba}"/>
    <geom name="table" type="box"
          size="{TABLE_HALF[0]} {TABLE_HALF[1]} {TABLE_HALF[2]}"
          pos="{TABLE_CENTER[0]} {TABLE_CENTER[1]} {TABLE_TOP_Z - TABLE_HALF[2]}"
          rgba="{table_rgba}" friction="1.2 0.005 0.0001"/>
    <geom name="pedestal" type="box" size="0.12 0.12 {(-FLOOR_Z) / 2}"
          pos="0 0 {FLOOR_Z / 2}" rgba="0.25 0.25 0.27 1"/>
    <body name="cube" pos="{cx} {cy} {TABLE_TOP_Z + CUBE_HALF}">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
            rgba="{cube_rgba}" mass="0.05" friction="1.5 0.01 0.0001"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
    {obstacle_xml}
    {''.join(distractors_xml)}
    <camera name="cam0" mode="fixed" pos="{CAM0_POS[0]} {CAM0_POS[1]} {CAM0_POS[2]}"
            xyaxes="{_camera_xyaxes(CAM0_POS, CAM0_LOOKAT)}" fovy="{CAM0_FOVY}"/>
  </worldbody>
  <keyframe>
    <key name="start" qpos="{key_qpos}" ctrl="{KEY_CTRL}"/>
  </keyframe>
</mujoco>
"""


def compile_scene(spec: SceneSpec) -> mujoco.MjModel:
    """XML 写入 Panda 资产目录旁的临时文件再编译 (保证相对 include/mesh 路径可解析)。"""
    if not (PANDA_DIR / PANDA_XML).exists():
        raise FileNotFoundError(
            f"未找到 {PANDA_DIR / PANDA_XML}; 先运行 scripts/00_setup_assets.py"
        )
    xml = build_scene_xml(spec)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".xml", dir=PANDA_DIR, delete=False, encoding="utf-8"
    ) as f:
        f.write(xml)
        tmp = Path(f.name)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
