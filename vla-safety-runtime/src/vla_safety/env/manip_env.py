"""ManipSafetyEnv: 100Hz 控制 tick 范式的 Panda 桌面抓取环境。

与常见 step(action) 环境的区别 —— 本项目复现的是运行时系统, 控制环、
安全检查、VLA 决策运行在不同频率上, 所以环境暴露的是:

  - set_command(a):   设置当前保持的速度指令 (来自 ring buffer 消费语义)
  - tick():           推进一个 10ms 控制步 (目标积分 + 微分 IK + 物理子步),
                      返回该 tick 内是否发生障碍接触 (逐物理子步检测, 不漏帧)
  - render_rgb():     第三人称 RGB (VLA 观测)
  - render_depth():   腕部深度图 + 相机位姿 (安全监控的唯一感知输入)

动作契约 (common.ACTION_DIM=4, 归一化 [-1,1]):
  a[0:3] = 末端速度 / V_MAX (世界系),  a[3] = 夹爪 (>0 闭合)
姿态全程锁定 ready 位姿 (top-down), 不暴露旋转自由度。
"""
from __future__ import annotations

import mujoco
import numpy as np

from vla_safety.common import CONTROL_DT, SceneSpec, V_MAX
from vla_safety.env import scene as scene_mod

IK_DAMPING = 1e-4
TARGET_LEASH = 0.03       # 目标点距当前末端的最大超前量 (anti-windup)
ROT_WEIGHT = 2.0
GRIPPER_OPEN_CTRL = 255.0
SUCCESS_LIFT = 0.10
TCP_OFFSET = 0.103        # attachment_site 沿手部 z 轴到指尖中心的距离 (m)

PANDA_BODY_NAMES = (
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
)


class ManipSafetyEnv:
    def __init__(self, spec: SceneSpec | None = None,
                 render_size: int = 96, depth_size: int = 96,
                 workspace: dict | None = None):
        self.render_size = render_size
        self.depth_size = depth_size
        ws = workspace or {"x": [0.15, 0.70], "y": [-0.35, 0.35], "z": [0.03, 0.55]}
        self._ws_lo = np.array([ws["x"][0], ws["y"][0], ws["z"][0]])
        self._ws_hi = np.array([ws["x"][1], ws["y"][1], ws["z"][1]])
        self._rgb_renderer: mujoco.Renderer | None = None
        self._depth_renderer: mujoco.Renderer | None = None
        self._load(spec or SceneSpec.nominal())

    # ------------------------------------------------------------------ setup
    def _load(self, spec: SceneSpec) -> None:
        self.spec = spec
        self.model = scene_mod.compile_scene(spec)
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(CONTROL_DT / self.model.opt.timestep))
        assert self.n_sub >= 1, f"物理步长 {self.model.opt.timestep} 大于控制周期"

        self._site_id = self.model.site("attachment_site").id
        self._cube_body_id = self.model.body("cube").id
        self._cube_geom_id = self.model.geom("cube_geom").id
        self._cam_rgb_id = self.model.camera("cam0").id
        self._cam_depth_id = self.model.camera("wrist_depth").id
        self._cam_wrist_id = self.model.camera("wrist_rgb").id
        self._key_id = self.model.key("start").id
        self._finger_adr = [
            self.model.joint(n).qposadr[0]
            for n in ("finger_joint1", "finger_joint2")
        ]
        self._arm_ctrl_range = self.model.actuator_ctrlrange[:7].copy()

        # 障碍接触检测: panda 全部碰撞 geom + cube geom  vs  obstacle geom
        self._obstacle_geom_id = None
        if spec.obstacle_pos is not None:
            self._obstacle_geom_id = self.model.geom("obstacle").id
        panda_body_ids = set()
        for name in PANDA_BODY_NAMES:
            try:
                panda_body_ids.add(self.model.body(name).id)
            except KeyError:
                pass
        if not panda_body_ids:
            raise RuntimeError("未在模型中找到 Panda 机身 body, 检查 panda_safety.xml")
        self._robot_geom_ids = {
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] in panda_body_ids
        }

        if self._rgb_renderer is not None:
            self._rgb_renderer.close()
        if self._depth_renderer is not None:
            self._depth_renderer.close()
        self._rgb_renderer = mujoco.Renderer(self.model, self.render_size, self.render_size)
        self._depth_renderer = mujoco.Renderer(self.model, self.depth_size, self.depth_size)
        self._depth_renderer.enable_depth_rendering()

    # ------------------------------------------------------------- public API
    def reset(self, spec: SceneSpec | None = None) -> None:
        if spec is not None:
            self._load(spec)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_id)
        mujoco.mj_forward(self.model, self.data)
        self._target_pos = self.ee_pos.copy()
        self._target_quat = self.ee_quat.copy()
        self._cmd = np.zeros(4, dtype=np.float64)
        self._grip_ctrl = GRIPPER_OPEN_CTRL
        self.tick_count = 0
        self.violation_ticks = 0

    def set_command(self, a: np.ndarray) -> None:
        """设置当前保持的速度指令 (控制环每 tick 消费最新值, 与 ring buffer 语义一致)。"""
        self._cmd = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)

    @property
    def command(self) -> np.ndarray:
        return self._cmd.copy()

    def tick(self) -> bool:
        """推进一个控制步。返回该 tick 内是否发生 robot/cube 与障碍的接触。"""
        v = self._cmd[:3] * V_MAX
        ee = self.ee_pos
        self._target_pos = np.clip(
            self._target_pos + v * CONTROL_DT, ee - TARGET_LEASH, ee + TARGET_LEASH
        )
        self._target_pos = np.clip(self._target_pos, self._ws_lo, self._ws_hi)
        self._grip_ctrl = (1.0 - self._cmd[3]) / 2.0 * GRIPPER_OPEN_CTRL

        # IK 每 tick 解一次 (目标每 tick 仅移动 ~1.5mm, 位置伺服在子步间足够跟随)
        ctrl_arm = self._diff_ik(self._target_pos, self._target_quat)
        self.data.ctrl[:7] = np.clip(
            ctrl_arm, self._arm_ctrl_range[:, 0], self._arm_ctrl_range[:, 1]
        )
        self.data.ctrl[7] = self._grip_ctrl
        violated = False
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            # 逐物理子步检测: 控制 tick 末检测会漏掉瞬态接触
            if self._obstacle_contact():
                violated = True
        self.tick_count += 1
        if violated:
            self.violation_ticks += 1
        return violated

    def is_success(self) -> bool:
        return bool(self.cube_pos[2] > scene_mod.TABLE_TOP_Z + SUCCESS_LIFT)

    def render_rgb(self) -> np.ndarray:
        self._rgb_renderer.update_scene(self.data, camera=self._cam_rgb_id)
        return self._rgb_renderer.render()

    def render_wrist_rgb(self) -> np.ndarray:
        """腕部 RGB 相机 (指缝方向横置, 近垂直俯视 TCP)。第二视角直接
        编码方块相对夹爪的误差向量, 是侧视相机最后一厘米精度的补充。
        与 wrist_depth (前下视, 安全走廊) 是两个独立挂点。"""
        self._rgb_renderer.update_scene(self.data, camera=self._cam_wrist_id)
        return self._rgb_renderer.render()

    def render_depth(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (depth HxW float32 米, cam_pos (3,), cam_mat (3,3) 列为相机轴)。"""
        self._depth_renderer.update_scene(self.data, camera=self._cam_depth_id)
        depth = self._depth_renderer.render().astype(np.float32)
        cam_pos = self.data.cam_xpos[self._cam_depth_id].copy()
        cam_mat = self.data.cam_xmat[self._cam_depth_id].reshape(3, 3).copy()
        return depth, cam_pos, cam_mat

    @property
    def depth_fovy(self) -> float:
        return float(self.model.cam_fovy[self._cam_depth_id])

    def close(self) -> None:
        for r in (self._rgb_renderer, self._depth_renderer):
            if r is not None:
                r.close()
        self._rgb_renderer = self._depth_renderer = None

    # ------------------------------------------------------------- properties
    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site_id].copy()

    @property
    def ee_quat(self) -> np.ndarray:
        m = self.data.site_xmat[self._site_id].reshape(3, 3)
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, m.ravel())
        return q

    @property
    def tcp_pos(self) -> np.ndarray:
        """指尖中心: attachment_site 沿手部 z 轴 (approach 方向) 平移 TCP_OFFSET。"""
        m = self.data.site_xmat[self._site_id].reshape(3, 3)
        return self.ee_pos + m[:, 2] * TCP_OFFSET

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_id].copy()

    @property
    def gripper_width(self) -> float:
        return float(sum(self.data.qpos[a] for a in self._finger_adr))

    def proprio(self) -> np.ndarray:
        """(4,) = ee_pos + 夹爪开度。"""
        return np.concatenate([self.ee_pos, [self.gripper_width]]).astype(np.float32)

    # -------------------------------------------------------------- internals
    def _obstacle_contact(self) -> bool:
        if self._obstacle_geom_id is None:
            return False
        oid = self._obstacle_geom_id
        watched = self._robot_geom_ids | {self._cube_geom_id}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == oid and g2 in watched) or (g2 == oid and g1 in watched):
                return True
        return False

    def _diff_ik(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """阻尼最小二乘: 6D 末端误差 -> 7 关节位置增量。"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        err = np.zeros(6)
        err[:3] = target_pos - self.ee_pos
        err[3:] = ROT_WEIGHT * _quat_error(target_quat, self.ee_quat)
        J[3:] *= ROT_WEIGHT

        JT = J.T
        dq = JT @ np.linalg.solve(J @ JT + IK_DAMPING * np.eye(6), err)
        dq = np.clip(dq, -0.05, 0.05)
        return self.data.qpos[:7] + dq


# ------------------------------------------------------------ quaternion utils
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """返回把 current 转到 target 的角速度向量 (世界系, rad)。"""
    inv = np.array([current[0], -current[1], -current[2], -current[3]])
    dq = _quat_mul(target, inv)
    if dq[0] < 0:
        dq = -dq
    vel = np.empty(3)
    mujoco.mju_quat2Vel(vel, dq, 1.0)
    return vel
