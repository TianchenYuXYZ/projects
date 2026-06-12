"""Episode 运行器: sim-time 确定性调度器。

统计实验 (E1/E4) 的执行核心。三个异频组件的时序结构在 *仿真时间* 里
显式建模, 保证逐 seed 可复现:

  控制环   100 Hz   每 tick 消费最新指令 (ring buffer 的最新值语义)
  安全检查 ~33 Hz   每 period_ticks 一次深度走廊检测
  VLA      ~6.7 Hz  t 时刻快照 obs -> t + think_ticks 时刻指令才生效
                    (思考期间旧指令保持 —— 正是 "thinking vs acting
                    frequency gap" 导致碰撞不等人的机制)

准入控制语义 (文档 3.2): recovery 执行期间 VLA 在飞结果作废 (它基于
触发前的世界状态); arm 执行完毕 -> 冻结等待 VLA 以新鲜观测重新接管。

墙钟延迟 (E2) 不在这里测 —— 那是 runtime/streams + C++ 控制环的事。
"""
from __future__ import annotations

import dataclasses

import numpy as np

from vla_safety.common import V_MAX
from vla_safety.env.manip_env import ManipSafetyEnv
from vla_safety.safety.monitor import OracleMonitor, RecoveryOutcome, RecoveryPlan
from vla_safety.baselines.qp_fallback import QPProjectionFallback
from vla_safety.vla.policy import VLAPolicy


@dataclasses.dataclass
class EpisodeResult:
    success: bool
    violation_ticks: int
    violation_free_success: bool
    first_violation_tick: int | None
    n_triggers: int
    triggers: list[dict]
    qp_interventions: int
    ticks_run: int
    decisions: int
    final_dist_goal: float
    traj: list[list[float]] | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.traj is None:
            d.pop("traj")
        return d


@dataclasses.dataclass
class _ActiveRecovery:
    plan: RecoveryPlan
    grip: float                  # 触发时刻的夹爪指令 (透传, 安全层不动夹爪)
    phase: str = "arm"           # arm -> window
    step: int = 0
    tick_in_step: int = 0
    window_end: int = -1
    contact: bool = False


def _dist_goal(env: ManipSafetyEnv) -> float:
    return float(np.linalg.norm(env.tcp_pos - env.cube_pos))


def run_episode(env: ManipSafetyEnv, policy: VLAPolicy, cfg: dict,
                monitor: OracleMonitor | None = None,
                qp: QPProjectionFallback | None = None,
                qp_checker=None,
                record_traj: bool = False) -> EpisodeResult:
    """env 必须已 reset 到目标场景。monitor 与 qp 互斥。"""
    assert not (monitor is not None and qp is not None), "monitor 与 qp 互斥"
    think_ticks = int(cfg["vla"]["decision_period_ticks"])
    safety_period = int(cfg["safety"]["period_ticks"])
    min_speed = float(cfg["safety"]["min_speed_trigger"])
    ticks_per_step = int(cfg["recovery"]["ticks_per_step"])
    post_window = int(cfg["recovery"]["post_window_ticks"])
    horizon = int(cfg["env"]["episode_ticks"])

    pending: dict | None = None          # 在飞的 VLA 推理 {ready_at, cmd}
    rec: _ActiveRecovery | None = None
    n_triggers = 0
    decisions = 0
    first_violation: int | None = None
    success = False
    traj: list[list[float]] = []
    qp_before = qp.interventions if qp is not None else 0
    monitor_log_start = len(monitor.trigger_log) if monitor is not None else 0

    for t in range(horizon):
        # ---------------- 1) VLA 推理流水线 (思考延迟显式建模) ----------------
        if pending is not None and t >= pending["ready_at"]:
            if rec is None or rec.phase == "window":
                env.set_command(pending["cmd"])      # 准入: arm 执行期之外放行
            pending = None                           # arm 执行期: 在飞结果作废
        if pending is None and (rec is None or rec.phase == "window"):
            img = env.render_rgb()
            wimg = env.render_wrist_rgb()
            cmd = policy.act(img, wimg, env.proprio())
            decisions += 1
            pending = {"ready_at": t + think_ticks, "cmd": cmd}

        # ---------------- 2) 安全检查 tick (monitor 或 qp) --------------------
        if t % safety_period == 0 and rec is None:
            v_world = env.command[:3] * V_MAX
            speed = float(np.linalg.norm(v_world))
            if monitor is not None and speed >= min_speed:
                depth, cpos, cmat = env.render_depth()
                rep = monitor.check(depth, cpos, cmat, env.ee_pos, env.tcp_pos,
                                    env.cube_pos, v_world)
                if monitor.should_trigger(rep, speed, n_triggers):
                    plan = monitor.plan_recovery(rep, t, _dist_goal(env))
                    rec = _ActiveRecovery(plan=plan, grip=float(env.command[3]))
                    n_triggers += 1
                    pending = None                   # 丢弃基于触发前世界的在飞结果
            elif qp is not None and speed >= min_speed:
                depth, cpos, cmat = env.render_depth()
                rep = qp_checker.check(depth, cpos, cmat, env.ee_pos, env.tcp_pos,
                                       env.cube_pos, v_world)
                if rep.conflict:
                    env.set_command(qp.project(env.command, rep, env.ee_pos))

        # ---------------- 3) recovery arm 执行 (token -> 速度指令) ------------
        if rec is not None and rec.phase == "arm":
            tokens = rec.plan.arm.tokens[rec.step]
            v = monitor.tokenizer.decode(tokens)
            env.set_command(np.array([v[0], v[1], v[2], rec.grip]))
            rec.tick_in_step += 1
            if rec.tick_in_step >= ticks_per_step:
                rec.tick_in_step = 0
                rec.step += 1
                if rec.step >= len(rec.plan.arm.tokens):
                    rec.phase = "window"
                    rec.window_end = t + post_window
                    # 冻结等待 VLA 以新鲜观测接管 (夹爪保持)
                    env.set_command(np.array([0.0, 0.0, 0.0, rec.grip]))

        # ---------------- 4) 推进物理 (逐子步接触检测) -------------------------
        violated = env.tick()
        if violated and first_violation is None:
            first_violation = t
        if rec is not None and violated:
            rec.contact = True
        if record_traj and t % 10 == 0:
            traj.append([float(x) for x in env.ee_pos])

        # ---------------- 5) 观察窗结算 -> bandit reward ----------------------
        if rec is not None and rec.phase == "window" and t >= rec.window_end:
            v_now = env.command[:3] * V_MAX
            if np.linalg.norm(v_now) < min_speed:
                to_goal = env.cube_pos - env.tcp_pos       # 意图方向: 朝目标
                norm = np.linalg.norm(to_goal)
                v_now = (to_goal / norm) * V_MAX * 0.5 if norm > 1e-9 else v_now
            depth, cpos, cmat = env.render_depth()
            rep_end = monitor.check(depth, cpos, cmat, env.ee_pos, env.tcp_pos,
                                    env.cube_pos, v_now)
            outcome = RecoveryOutcome(
                contact_during=rec.contact,
                conflict_at_end=rep_end.conflict,
                dist_goal_end=_dist_goal(env),
                success=env.is_success(),
            )
            monitor.resolve(rec.plan, outcome)
            rec = None

        # ---------------- 6) 任务完成判定 -------------------------------------
        if env.is_success():
            success = True
            break

    # 仍在飞的 recovery 以 episode 末状态结算 (不丢样本)
    if rec is not None and monitor is not None:
        outcome = RecoveryOutcome(
            contact_during=rec.contact,
            conflict_at_end=False,
            dist_goal_end=_dist_goal(env),
            success=success,
        )
        monitor.resolve(rec.plan, outcome)

    triggers = (monitor.trigger_log[monitor_log_start:]
                if monitor is not None else [])
    return EpisodeResult(
        success=success,
        violation_ticks=env.violation_ticks,
        violation_free_success=bool(success and env.violation_ticks == 0),
        first_violation_tick=first_violation,
        n_triggers=n_triggers,
        triggers=triggers,
        qp_interventions=(qp.interventions - qp_before) if qp is not None else 0,
        ticks_run=env.tick_count,
        decisions=decisions,
        final_dist_goal=_dist_goal(env),
        traj=traj if record_traj else None,
    )
