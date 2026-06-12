"""上下文多臂老虎机: training-free 的 recovery arm 仲裁。

理论依据 (文档引用): Safe-LUCB / Linear Stochastic Bandits Under Safety
Constraints —— CMAB-style monitor 在 fast controller 与 conservative
fallback 之间仲裁。这里用最朴素也最稳健的 tabular 形式: 每个几何上下文
(障碍方位 x 距离) 一张独立的 arm 统计表。

training-free 的含义: 没有离线 reward 标注、没有梯度、没有 GPU——
select 是一次 argmax (UCB1) 或 Beta 采样 (Thompson), 纯 CPU, 微秒级。
"""
from __future__ import annotations

import numpy as np


class ContextualBandit:
    def __init__(self, algo: str, n_arms: int, contexts: list[str],
                 rng: np.random.Generator,
                 ucb_c: float = 1.2, fixed_arm: int = 0):
        assert algo in ("ucb1", "thompson", "random", "fixed"), algo
        self.algo = algo
        self.n_arms = n_arms
        self.contexts = list(contexts)
        self.rng = rng
        self.ucb_c = ucb_c
        self.fixed_arm = fixed_arm
        self.counts = {c: np.zeros(n_arms, dtype=np.int64) for c in self.contexts}
        self.succ = {c: np.zeros(n_arms, dtype=np.float64) for c in self.contexts}
        self.history: list[dict] = []   # 全量决策日志 (E3 画 regret/分布演化)

    # --------------------------------------------------------------- select
    def select(self, ctx: str) -> int:
        if ctx not in self.counts:
            raise KeyError(f"未知 context: {ctx}")
        if self.algo == "fixed":
            return self.fixed_arm
        if self.algo == "random":
            return int(self.rng.integers(self.n_arms))
        if self.algo == "ucb1":
            return self._ucb1(ctx)
        return self._thompson(ctx)

    def _ucb1(self, ctx: str) -> int:
        n = self.counts[ctx]
        # 未探索的 arm 优先 (按索引序, 保证确定性)
        unplayed = np.flatnonzero(n == 0)
        if unplayed.size > 0:
            return int(unplayed[0])
        total = float(n.sum())
        mean = self.succ[ctx] / n
        bonus = self.ucb_c * np.sqrt(2.0 * np.log(total) / n)
        return int(np.argmax(mean + bonus))

    def _thompson(self, ctx: str) -> int:
        # Beta(1+成功, 1+失败) 共轭后验采样
        alpha = 1.0 + self.succ[ctx]
        beta = 1.0 + (self.counts[ctx] - self.succ[ctx])
        samples = self.rng.beta(alpha, beta)
        return int(np.argmax(samples))

    # --------------------------------------------------------------- update
    def update(self, ctx: str, arm: int, reward: float) -> None:
        assert reward in (0.0, 1.0), "本项目 reward 定义为二元 (避障成功 ∧ 任务可恢复)"
        self.counts[ctx][arm] += 1
        self.succ[ctx][arm] += reward
        self.history.append({"ctx": ctx, "arm": int(arm), "reward": float(reward)})

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        return {
            "algo": self.algo,
            "counts": {c: v.tolist() for c, v in self.counts.items()},
            "succ": {c: v.tolist() for c, v in self.succ.items()},
            "history": self.history,
        }
