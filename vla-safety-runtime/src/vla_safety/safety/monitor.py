"""MAB Oracle Monitor: 准入检查 + recovery arm 仲裁 + 在线 reward 累计。

"Oracle" 的含义 (文档 3.2 节): 不修改 VLA 的内部 logits, 只在 VLA 输出
action 之后做准入检查 —— 深度检测到几何冲突时, 用 bandit 选一个
recovery arm (VLA 原生 token 序列) 替换掉 VLA 的原 action。

reward 契约 (二元, 文档 3.2 节 "避障成功 ∧ 任务可恢复"):
  avoid_ok   = recovery 执行期 + 观察窗 W 内无障碍接触, 且窗末走廊无冲突
  recover_ok = 窗末 |tcp-goal| <= 触发时刻 + slack, 或任务已成功
  r = avoid_ok ∧ recover_ok
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from vla_safety.perception.depth_safety import (CONTEXTS, ConflictReport,
                                                DepthSafetyChecker)
from vla_safety.safety.arms import RecoveryArm, build_arm_library
from vla_safety.safety.bandit import ContextualBandit
from vla_safety.vla.tokenizer import ActionTokenizer


@dataclasses.dataclass
class RecoveryPlan:
    arm: RecoveryArm
    context: str
    trigger_tick: int
    dist_goal_trigger: float
    report: ConflictReport


@dataclasses.dataclass
class RecoveryOutcome:
    contact_during: bool        # arm 执行期 + 观察窗内是否发生障碍接触
    conflict_at_end: bool       # 窗末沿恢复后意图方向是否仍有走廊冲突
    dist_goal_end: float
    success: bool


class OracleMonitor:
    def __init__(self, safety_cfg: dict, recovery_cfg: dict, bandit_cfg: dict,
                 tokenizer: ActionTokenizer, depth_size: int, fovy_deg: float,
                 device: str = "cuda",
                 stream: "torch.cuda.Stream | None" = None,
                 timing: bool = False):
        self.cfg = safety_cfg
        self.checker = DepthSafetyChecker(
            safety_cfg, depth_size, fovy_deg, device=device,
            stream=stream, timing=timing,
        )
        self.tokenizer = tokenizer
        self.arms = build_arm_library(
            tokenizer,
            arm_speed=float(recovery_cfg["arm_speed"]),
            steps=int(recovery_cfg["steps_per_arm"]),
        )
        rng = np.random.default_rng(int(bandit_cfg.get("seed", 0)))
        fixed_idx = next(
            (a.index for a in self.arms if a.name == bandit_cfg.get("fixed_arm")), 0
        )
        self.bandit = ContextualBandit(
            algo=bandit_cfg["algo"], n_arms=len(self.arms), contexts=CONTEXTS,
            rng=rng, ucb_c=float(bandit_cfg.get("ucb_c", 1.2)), fixed_arm=fixed_idx,
        )
        self.progress_slack = float(recovery_cfg["progress_slack"])
        self.trigger_log: list[dict] = []

    # ------------------------------------------------------------------ 感知
    def check(self, depth, cam_pos, cam_mat, ee, tcp, goal,
              v_world) -> ConflictReport:
        return self.checker.check(depth, cam_pos, cam_mat, ee, tcp, goal, v_world)

    def should_trigger(self, report: ConflictReport, speed: float,
                       triggers_so_far: int) -> bool:
        return (report.conflict
                and speed >= float(self.cfg["min_speed_trigger"])
                and triggers_so_far < int(self.cfg["max_triggers_per_ep"]))

    # ------------------------------------------------------------------ 仲裁
    def plan_recovery(self, report: ConflictReport, tick: int,
                      dist_goal: float) -> RecoveryPlan:
        arm_idx = self.bandit.select(report.context)
        return RecoveryPlan(
            arm=self.arms[arm_idx], context=report.context,
            trigger_tick=tick, dist_goal_trigger=dist_goal, report=report,
        )

    # ------------------------------------------------------------------ 结算
    def resolve(self, plan: RecoveryPlan, outcome: RecoveryOutcome) -> float:
        avoid_ok = (not outcome.contact_during) and (not outcome.conflict_at_end)
        recover_ok = (outcome.success
                      or outcome.dist_goal_end
                      <= plan.dist_goal_trigger + self.progress_slack)
        reward = 1.0 if (avoid_ok and recover_ok) else 0.0
        self.bandit.update(plan.context, plan.arm.index, reward)
        self.trigger_log.append({
            "tick": plan.trigger_tick,
            "context": plan.context,
            "arm": plan.arm.name,
            "reward": reward,
            "avoid_ok": bool(avoid_ok),
            "recover_ok": bool(recover_ok),
            "dist_trigger": plan.dist_goal_trigger,
            "dist_end": outcome.dist_goal_end,
            "n_points": plan.report.n_points,
            "bearing_deg": plan.report.bearing_deg,
        })
        return reward
