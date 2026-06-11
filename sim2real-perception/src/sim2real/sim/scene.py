"""SceneConfig -> MJCF XML -> MjModel.

世界坐标约定: 桌面顶面 z = 0, Panda 基座在原点 (安装座向下延伸到地面),
方块初始在桌面 (0.5, 0) 附近。所有视觉随机化 (纹理/光照/相机/干扰物)
都通过重新生成 XML + 重新编译模型实现, 物理参数保持不变。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
import numpy as np

from sim2real.common import PROJECT_ROOT, SceneConfig

PANDA_DIR = PROJECT_ROOT / "assets" / "franka_emika_panda"
PANDA_XML = "panda.xml"

TABLE_TOP_Z = 0.0
TABLE_HALF = (0.32, 0.42, 0.03)          # 桌板半尺寸
TABLE_CENTER = (0.50, 0.0)
# 25mm 方块: Panda 指尖 pad 的有效夹持区只有 ~17mm (hand-0.094 ~ hand-0.111),
# 方块太高会被指根碰撞网格 (hand-0.090) 先顶开
CUBE_HALF = 0.0125
FLOOR_Z = -0.76

# Franka 标准 ready 位姿: 末端竖直向下, 适合 top-down 抓取
HOME_QPOS_ARM = "0 -0.785398 0 -2.356194 0 1.570796 0.785398"
HOME_QPOS_FINGERS = "0.04 0.04"


def _camera_xyaxes(pos: np.ndarray, lookat: np.ndarray) -> str:
    """由相机位置和注视点计算 MJCF camera 的 xyaxes (右轴 + 上轴)。"""
    forward = lookat - pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    vals = np.concatenate([right, up])
    return " ".join(f"{v:.6f}" for v in vals)


def _texture_asset(name: str, tex_path: str | None) -> tuple[str, str]:
    """返回 (asset 片段, material 引用名)。无纹理时仅用 rgba 纯色。"""
    if tex_path is None:
        return "", ""
    p = Path(tex_path).resolve().as_posix()
    asset = (
        f'<texture name="tex_{name}" type="2d" file="{p}"/>\n'
        f'<material name="mat_{name}" texture="tex_{name}" texrepeat="3 3" '
        f'texuniform="false" reflectance="0.05"/>\n'
    )
    return asset, f'material="mat_{name}"'


def build_scene_xml(scene: SceneConfig) -> str:
    """根据 SceneConfig 生成完整场景 MJCF (include 菜单库 Panda)。"""
    table_asset, table_mat = _texture_asset("table", scene.table_texture)
    floor_asset, floor_mat = _texture_asset("floor", scene.floor_texture)

    lights_xml = []
    for i, lt in enumerate(scene.lights):
        px, py, pz = lt["pos"]
        d = float(lt["diffuse"])
        a = float(lt["ambient"])
        # 光照方向指向桌心
        target = np.array([TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z])
        dirv = target - np.array([px, py, pz])
        dirv = dirv / np.linalg.norm(dirv)
        lights_xml.append(
            f'<light name="light{i}" pos="{px} {py} {pz}" '
            f'dir="{dirv[0]:.4f} {dirv[1]:.4f} {dirv[2]:.4f}" '
            f'diffuse="{d} {d} {d}" ambient="{a} {a} {a}" castshadow="true"/>'
        )

    distractors_xml = []
    for i, dt in enumerate(scene.distractors):
        x, y = dt["pos"]
        s = float(dt["size"])
        rgba = " ".join(f"{v:.3f}" for v in dt["rgba"])
        # 干扰物纯视觉: contype=conaffinity=0, 不参与碰撞, 保证回放轨迹不变
        if dt["type"] == "box":
            g = (f'<geom name="distr{i}" type="box" size="{s} {s} {s}" '
                 f'pos="{x} {y} {TABLE_TOP_Z + s}"')
        elif dt["type"] == "sphere":
            g = (f'<geom name="distr{i}" type="sphere" size="{s}" '
                 f'pos="{x} {y} {TABLE_TOP_Z + s}"')
        else:  # cylinder
            g = (f'<geom name="distr{i}" type="cylinder" size="{s} {s}" '
                 f'pos="{x} {y} {TABLE_TOP_Z + s}"')
        distractors_xml.append(
            g + f' rgba="{rgba}" contype="0" conaffinity="0"/>'
        )

    cube_rgba = " ".join(f"{v:.3f}" for v in scene.cube_rgba)
    table_rgba = " ".join(f"{v:.3f}" for v in scene.table_rgba)
    floor_rgba = " ".join(f"{v:.3f}" for v in scene.floor_rgba)
    cam_pos = np.array(scene.camera_pos, dtype=float)
    cam_lookat = np.array(scene.camera_lookat, dtype=float)
    cx, cy = scene.cube_pos

    # keyframe qpos = 7 arm + 2 fingers + 7 cube freejoint
    key_qpos = (f"{HOME_QPOS_ARM} {HOME_QPOS_FINGERS} "
                f"{cx} {cy} {TABLE_TOP_Z + CUBE_HALF} 1 0 0 0")
    # ctrl = 7 个关节位置伺服 + 夹爪 (255 = 全开)
    key_ctrl = "0 -0.785398 0 -2.356194 0 1.570796 0.785398 255"

    return f"""
<mujoco model="s2r_scene">
  <include file="{PANDA_XML}"/>
  <statistic center="0.4 0 0.2" extent="1.2"/>
  <visual>
    <headlight diffuse="0.3 0.3 0.3" ambient="0.15 0.15 0.15" specular="0 0 0"/>
    <global offwidth="512" offheight="512"/>
  </visual>
  <asset>
    {table_asset}{floor_asset}
  </asset>
  <worldbody>
    {''.join(lights_xml)}
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 {FLOOR_Z}"
          rgba="{floor_rgba}" {floor_mat}/>
    <geom name="table" type="box"
          size="{TABLE_HALF[0]} {TABLE_HALF[1]} {TABLE_HALF[2]}"
          pos="{TABLE_CENTER[0]} {TABLE_CENTER[1]} {TABLE_TOP_Z - TABLE_HALF[2]}"
          rgba="{table_rgba}" {table_mat} friction="1.2 0.005 0.0001"/>
    <geom name="pedestal" type="box" size="0.12 0.12 {(-FLOOR_Z) / 2}"
          pos="0 0 {FLOOR_Z / 2}" rgba="0.25 0.25 0.27 1"/>
    <body name="cube" pos="{cx} {cy} {TABLE_TOP_Z + CUBE_HALF}">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
            rgba="{cube_rgba}" mass="0.05" friction="1.5 0.01 0.0001"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
    {''.join(distractors_xml)}
    <camera name="cam0" mode="fixed" pos="{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}"
            xyaxes="{_camera_xyaxes(cam_pos, cam_lookat)}" fovy="{scene.camera_fovy}"/>
  </worldbody>
  <keyframe>
    <key name="start" qpos="{key_qpos}" ctrl="{key_ctrl}"/>
  </keyframe>
</mujoco>
"""


def compile_scene(scene: SceneConfig) -> mujoco.MjModel:
    """生成 XML 写入 Panda 资产目录旁的临时文件再编译 (保证相对 include/mesh 路径可解析)。"""
    if not (PANDA_DIR / PANDA_XML).exists():
        raise FileNotFoundError(
            f"未找到 {PANDA_DIR / PANDA_XML}; 先运行 scripts/00_setup_assets.py 下载 Panda 模型"
        )
    xml = build_scene_xml(scene)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".xml", dir=PANDA_DIR, delete=False, encoding="utf-8"
    ) as f:
        f.write(xml)
        tmp = Path(f.name)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
