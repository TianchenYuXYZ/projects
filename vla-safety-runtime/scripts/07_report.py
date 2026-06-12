"""聚合全部实验产物 -> results/report.md + 图表。

输入 (各实验脚本的输出, 缺失项跳过并在报告中标注):
  eval_baseline.json / eval_ucb1.json / eval_thompson.json /
  eval_random.json / eval_fixed.json / eval_qp.json
  latency.json / ablation_serial.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vla_safety  # noqa: F401
from vla_safety.common import PROJECT_ROOT, load_yaml
from vla_safety.perception.depth_safety import CONTEXTS
from vla_safety.safety.arms import ARM_NAMES

RESULTS = PROJECT_ROOT / "results"
VARIANTS = ["baseline", "ucb1", "thompson", "random", "fixed", "qp"]
SUITES = ["id_clean", "ood_obstacle", "ood_obstacle_visual"]
LABELS = {"baseline": "VLA alone", "ucb1": "Monitor (UCB1)",
          "thompson": "Monitor (TS)", "random": "Monitor (random arm)",
          "fixed": "Monitor (fixed retreat)", "qp": "CBF-projection fallback"}


def load_eval() -> dict:
    out = {}
    for v in VARIANTS:
        p = RESULTS / f"eval_{v}.json"
        if p.exists():
            out[v] = json.loads(p.read_text(encoding="utf-8"))
    return out


def fig_e1(evals: dict) -> str | None:
    if not evals:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, title in [
            (axes[0], "violation_free_success_rate",
             "Violation-free completion"),
            (axes[1], "success_rate", "Task success (违规与否不论)")]:
        xs, width = np.arange(len(SUITES)), 0.13
        present = [v for v in VARIANTS if v in evals]
        for k, v in enumerate(present):
            vals = [evals[v].get(s, {}).get("aggregate", {}).get(metric, np.nan)
                    for s in SUITES]
            ax.bar(xs + (k - len(present) / 2) * width, vals, width,
                   label=LABELS[v])
        ax.set_xticks(xs, SUITES)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = RESULTS / "fig_e1_outcomes.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def fig_e2(lat: dict | None) -> str | None:
    if lat is None:
        return None
    conds = lat["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(conds.keys())
    # 左: e2e p50/p99 对比
    p50 = [conds[c]["e2e_ms"]["p50"] for c in names]
    p99 = [conds[c]["e2e_ms"]["p99"] for c in names]
    xs = np.arange(len(names))
    axes[0].bar(xs - 0.18, p50, 0.36, label="p50")
    axes[0].bar(xs + 0.18, p99, 0.36, label="p99")
    axes[0].axhline(5.0, color="r", ls="--", lw=1, label="5 ms 预算")
    axes[0].set_xticks(xs, names, rotation=12)
    axes[0].set_ylabel("e2e 延迟 (ms)")
    axes[0].set_title("恢复链路 (深度帧就绪 -> SHM 写完成)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    # 右: prio_load 分解
    c = conds.get("prio_load")
    if c:
        seg_names = ["GPU 链路", "MAB select", "token 解码", "SHM 写"]
        seg_p99 = [c["t_gpu_ms"]["p99"],
                   c["t_plan_us"]["p99"] / 1e3,
                   c["t_tok_us"]["p99"] / 1e3,
                   c["t_write_us"]["p99"] / 1e3]
        shm = lat.get("shm", {}).get("spin", {})
        if shm:
            seg_names.append("跨进程可见")
            seg_p99.append(shm["latency_us"]["p99"] / 1e3)
        axes[1].barh(seg_names[::-1], seg_p99[::-1])
        for i, v in enumerate(seg_p99[::-1]):
            axes[1].text(v, i, f" {v:.3f} ms", va="center", fontsize=9)
        axes[1].set_xlabel("p99 (ms), 解码满载 + 高优先级 stream")
        axes[1].set_title("链路分解 (prio_load)")
        axes[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "fig_e2_latency.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def fig_e3(evals: dict) -> str | None:
    """bandit 学习曲线 (跨集累计) + UCB1 各 context 的 arm 选择分布。"""
    algos = [v for v in ["ucb1", "thompson", "random", "fixed"] if v in evals]
    if not algos:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for v in algos:
        # 用 ood_obstacle suite 的 bandit history
        suite = evals[v].get("ood_obstacle")
        if not suite or "bandit" not in suite:
            continue
        hist = suite["bandit"]["history"]
        if not hist:
            continue
        r = np.array([h["reward"] for h in hist])
        cum = np.cumsum(r) / (np.arange(len(r)) + 1)
        axes[0].plot(cum, label=f"{LABELS[v]} (n={len(r)})")
    axes[0].set_xlabel("recovery 触发序号 (跨集在线累计)")
    axes[0].set_ylabel("累计平均 reward")
    axes[0].set_title("training-free 在线学习曲线")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    if "ucb1" in evals and "ood_obstacle" in evals["ucb1"]:
        counts = evals["ucb1"]["ood_obstacle"]["bandit"]["counts"]
        mat = np.array([counts[c] for c in CONTEXTS], dtype=float)
        row_sum = mat.sum(axis=1, keepdims=True)
        frac = np.divide(mat, row_sum, out=np.zeros_like(mat),
                         where=row_sum > 0)
        im = axes[1].imshow(frac, cmap="viridis", aspect="auto")
        axes[1].set_xticks(range(len(ARM_NAMES)), ARM_NAMES,
                           rotation=20, fontsize=8)
        axes[1].set_yticks(range(len(CONTEXTS)), CONTEXTS, fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    axes[1].text(j, i, int(mat[i, j]), ha="center",
                                 va="center", color="w", fontsize=8)
        axes[1].set_title("UCB1: context x arm 选择次数 (颜色=占比)")
        fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    out = RESULTS / "fig_e3_bandit.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def fig_e6(ser: dict | None) -> str | None:
    if ser is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    modes = list(ser.keys())
    axes[0].bar(modes, [ser[m]["achieved_hz"] for m in modes],
                color=["#777", "#2a7", "#c44"])
    axes[0].axhline(100, color="k", ls="--", lw=1)
    axes[0].set_ylabel("达成控制频率 (Hz)")
    axes[0].set_title("100Hz 控制环: 解耦 vs 串行")
    axes[1].bar(modes, [ser[m]["interval_ms"]["p99"] for m in modes],
                color=["#777", "#2a7", "#c44"])
    axes[1].axhline(10, color="k", ls="--", lw=1, label="10ms 周期")
    axes[1].set_ylabel("tick 间隔 p99 (ms)")
    axes[1].set_yscale("log")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "fig_e6_serial.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def main() -> None:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    evals = load_eval()
    lat = None
    lat_p = RESULTS / "latency.json"
    if lat_p.exists():
        lat = json.loads(lat_p.read_text(encoding="utf-8"))
    ser = None
    ser_p = RESULTS / "ablation_serial.json"
    if ser_p.exists():
        ser = json.loads(ser_p.read_text(encoding="utf-8"))

    figs = {"e1": fig_e1(evals), "e2": fig_e2(lat),
            "e3": fig_e3(evals), "e6": fig_e6(ser)}

    lines = ["# VLA Runtime Safety — 实验报告", ""]
    lines += ["复现目标 (research_experience_breakdown2.docx): "
              "感知-策略解耦 + training-free MAB Oracle + <5ms 恢复通道。", ""]

    # E1 表
    if evals:
        lines += ["## E1 — 安全效果 (violation-free completion)", ""]
        header = "| variant | " + " | ".join(SUITES) + " |"
        lines += [header, "|" + "---|" * (len(SUITES) + 1)]
        for v in VARIANTS:
            if v not in evals:
                continue
            cells = []
            for s in SUITES:
                a = evals[v].get(s, {}).get("aggregate")
                cells.append(f"{a['violation_free_success_rate']:.1%} "
                             f"(succ {a['success_rate']:.1%})" if a else "—")
            lines += [f"| {LABELS[v]} | " + " | ".join(cells) + " |"]
        lines += ["", f"![E1]({figs['e1']})", ""] if figs["e1"] else [""]

    # E2 表
    if lat:
        lines += ["## E2 — 恢复链路延迟 (墙钟)", ""]
        lines += ["| 条件 | e2e p50 (ms) | e2e p99 (ms) | GPU 段 p99 (ms) |",
                  "|---|---|---|---|"]
        for c, r in lat["conditions"].items():
            lines += [f"| {c} | {r['e2e_ms']['p50']:.2f} | "
                      f"{r['e2e_ms']['p99']:.2f} | {r['t_gpu_ms']['p99']:.2f} |"]
        shm = lat.get("shm", {})
        if shm:
            lines += ["", f"跨进程可见 (C++ 自旋读): p50 "
                      f"{shm['spin']['latency_us']['p50']:.1f} µs, p99 "
                      f"{shm['spin']['latency_us']['p99']:.1f} µs; "
                      f"200Hz 节拍拾取 p99 "
                      f"{shm['hz200']['latency_us']['p99'] / 1000:.2f} ms "
                      "(含半周期相位, 上界 5ms 由频率决定)。"]
        hl = lat.get("headline", {})
        if hl:
            lines += ["", f"**满载恢复链路 p99 = "
                      f"{hl['recovery_p99_ms_under_load']:.2f} ms "
                      f"(声明 {hl['claim']}: "
                      f"{'PASS' if hl['met'] else 'FAIL'})**"]
        lines += ["", f"![E2]({figs['e2']})", ""] if figs["e2"] else [""]

    # E3
    if figs["e3"]:
        lines += ["## E3 — MAB 在线学习", "",
                  "reward = 避障成功 ∧ 任务可恢复 (二元, 见 monitor.resolve)。",
                  "", f"![E3]({figs['e3']})", ""]
        for v in ["ucb1", "thompson", "random", "fixed"]:
            if v in evals and "ood_obstacle" in evals[v]:
                a = evals[v]["ood_obstacle"]["aggregate"]
                rr = a.get("recovery_reward_rate")
                lines += [f"- {LABELS[v]}: recovery reward 率 "
                          f"{rr:.1%}, 平均触发 {a['mean_triggers']:.1f} 次/集"
                          if rr is not None else f"- {LABELS[v]}: 无触发"]
        lines += [""]

    # E6
    if ser:
        lines += ["## E6 — 解耦 vs 串行控制环", "",
                  "| 模式 | 达成 Hz | deadline miss | 间隔 p99 (ms) |",
                  "|---|---|---|---|"]
        for m, r in ser.items():
            lines += [f"| {m} | {r['achieved_hz']:.1f} | "
                      f"{r['deadline_miss_rate']:.2%} | "
                      f"{r['interval_ms']['p99']:.1f} |"]
        lines += ["", f"![E6]({figs['e6']})", ""] if figs["e6"] else [""]

    # 文档声明映射
    lines += ["## 文档声明 ↔ 实测对照", ""]
    rows = []
    if evals.get("baseline") and evals.get("ucb1"):
        b = evals["baseline"].get("ood_obstacle", {}).get("aggregate")
        u = evals["ucb1"].get("ood_obstacle", {}).get("aggregate")
        if b and u:
            rows.append(("violation-free completion 14% → 72%",
                         f"{b['violation_free_success_rate']:.0%} → "
                         f"{u['violation_free_success_rate']:.0%} (ood_obstacle)"))
    if lat:
        hl = lat.get("headline", {})
        rows.append(("<5ms 安全恢复 (不重训/不抢主推理 GPU)",
                     f"满载 p99 {hl.get('recovery_p99_ms_under_load', float('nan')):.2f} ms"))
    rows.append(("training-free (无 reward 标注/无梯度/无重训)",
                 "UCB1/TS 在线累计, select 为 CPU argmax (µs 级)"))
    rows.append(("recovery 是 VLA 原生 token, 绕过外部 planner",
                 "arms 经 ActionTokenizer 编码, 与 VLA 共享动作词表"))
    lines += ["| 文档声明 | 实测/实现 |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    lines += [""]

    out = RESULTS / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")
    for k, v in figs.items():
        print(f"   fig {k}: {v or '(缺数据, 跳过)'}")


if __name__ == "__main__":
    main()
