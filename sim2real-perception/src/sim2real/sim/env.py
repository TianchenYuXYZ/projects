"""ManipEnv: MuJoCo + Franka Panda 桌面抓取环境。

动作空间 (7,), 全部归一化到 [-1, 1]:
  a[0:3] = 末端位置增量 dpos / DPOS_MAX
  a[3:6] = 末端旋转增量 (axis-angle) / DROT_MAX
  a[6]   = 夹爪: > 0 闭合, <= 0 张开 (连续映射到 ctrl)
观测: image (HxWx3 uint8 RGB, 固定相机), proprio (8,) = 7 关节角 + 夹爪开度。
末端控制用阻尼最小二乘微分 IK 映射到 7 个关节位置伺服。
"""
from __future__ import annotations

import mujoco
import numpy as np

from sim2real.common import Obs, SceneConfig
from sim2real.sim import scene as scene_mod

DPOS_MAX = 0.02          # m / 控制步
DROT_MAX = 0.10          # rad / 控制步
CONTROL_DT = 0.05        # 20 Hz 控制频率
SUCCESS_LIFT = 0.10      # 方块抬离桌面高度阈值 (m)
IK_DAMPING = 1e-4
TARGET_LEASH = 0.03      # 目标位置距当前末端的最大超前量 (anti-windup)
ROT_WEIGHT = 2.0         # IK 中旋转误差的权重 (位置误差被 leash 限幅后量级 ~0.03)
GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0


class ManipEnv:
    def __init__(self, scene: SceneConfig | None = None, render_size: int = 224):
        self.render_size = render_size
        self._renderer: mujoco.Renderer | None = None
        self._load(scene or SceneConfig.nominal())

    # ------------------------------------------------------------------ setup
    def _load(self, scene: SceneConfig) -> None:
        self.scene = scene
        self.model = scene_mod.compile_scene(scene)
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(CONTROL_DT / self.model.opt.timestep))

        # 末端参考点: 优先 attachment_site, 否则退到 hand body
        try:
            self._site_id = self.model.site("attachment_site").id
            self._use_site = True
        except KeyError:
            self._site_id = self.model.body("hand").id
            self._use_site = False

        self._cube_jnt_adr = self.model.joint("cube_joint").qposadr[0]
        self._cube_body_id = self.model.body("cube").id
        self._cam_id = self.model.camera("cam0").id
        self._key_id = self.model.key("start").id
        # 手指关节地址 (menagerie panda: finger_joint1/2)
        self._finger_adr = [
            self.model.joint(n).qposadr[0]
            for n in ("finger_joint1", "finger_joint2")
        ]
        self._arm_ctrl_range = self.model.actuator_ctrlrange[:7].copy()

        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self.model, self.render_size, self.render_size)

    # ------------------------------------------------------------- public API
    def reset(self, scene: SceneConfig | None = None) -> Obs:
        if scene is not None:
            self._load(scene)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_id)
        # 目标末端位姿初始化为当前位姿
        mujoco.mj_forward(self.model, self.data)
        self._target_pos = self.ee_pos.copy()
        self._target_quat = self.ee_quat.copy()
        self._grip_ctrl = GRIPPER_OPEN_CTRL
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[Obs, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # leash: 防止目标点跑到伺服可达范围之外, 误差爆掉后 IK 会牺牲姿态去追位置
        ee = self.ee_pos
        self._target_pos = np.clip(
            self._target_pos + a[:3] * DPOS_MAX,
            ee - TARGET_LEASH, ee + TARGET_LEASH,
        )
        if np.linalg.norm(a[3:6]) > 1e-8:
            dq = _axis_angle_to_quat(a[3:6] * DROT_MAX)
            self._target_quat = _quat_mul(dq, self._target_quat)
        self._grip_ctrl = (1.0 - a[6]) / 2.0 * GRIPPER_OPEN_CTRL

        for _ in range(self.n_sub):
            ctrl_arm = self._diff_ik(self._target_pos, self._target_quat)
            self.data.ctrl[:7] = np.clip(
                ctrl_arm, self._arm_ctrl_range[:, 0], self._arm_ctrl_range[:, 1]
            )
            self.data.ctrl[7] = self._grip_ctrl
            mujoco.mj_step(self.model, self.data)

        succ = self.is_success()
        return self._obs(), succ, {"cube_pos": self.cube_pos.copy()}

    def is_success(self) -> bool:
        return bool(self.cube_pos[2] > scene_mod.TABLE_TOP_Z + SUCCESS_LIFT)

    def render(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._renderer.render()

    def replay_render(self, qpos: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """运动学回放渲染: 视觉 DR 不改物理, 直接按记录的 qpos 摆姿态出图。"""
        frames = np.empty(
            (len(indices), self.render_size, self.render_size, 3), dtype=np.uint8
        )
        for i, t in enumerate(indices):
            self.data.qpos[: qpos.shape[1]] = qpos[t]
            mujoco.mj_forward(self.model, self.data)
            frames[i] = self.render()
        return frames

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------- properties
    @property
    def ee_pos(self) -> np.ndarray:
        if self._use_site:
            return self.data.site_xpos[self._site_id].copy()
        return self.data.xpos[self._site_id].copy()

    @property
    def ee_quat(self) -> np.ndarray:
        if self._use_site:
            m = self.data.site_xmat[self._site_id].reshape(3, 3)
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, m.ravel())
            return q
        return self.data.xquat[self._site_id].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_id].copy()

    @property
    def gripper_width(self) -> float:
        return float(sum(self.data.qpos[a] for a in self._finger_adr))

    def proprio(self) -> np.ndarray:
        return np.concatenate(
            [self.data.qpos[:7], [self.gripper_width]]
        ).astype(np.float32)

    # -------------------------------------------------------------- internals
    def _obs(self) -> Obs:
        return Obs(image=self.render(), proprio=self.proprio())

    def _diff_ik(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """阻尼最小二乘: 6D 末端误差 -> 7 关节位置增量。"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        if self._use_site:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self._site_id)
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


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    q = np.empty(4)
    axis = rotvec / angle if angle > 1e-12 else np.array([1.0, 0.0, 0.0])
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    return q


def _quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """返回把 current 转到 target 的角速度向量 (世界系, rad)。"""
    inv = np.array([current[0], -current[1], -current[2], -current[3]])
    dq = _quat_mul(target, inv)
    if dq[0] < 0:
        dq = -dq
    vel = np.empty(3)
    mujoco.mju_quat2Vel(vel, dq, 1.0)
    return vel
