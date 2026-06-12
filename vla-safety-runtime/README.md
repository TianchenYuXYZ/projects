# VLA Runtime Safety — 感知-策略解耦 + MAB Oracle Monitor

复现 `research_experience_breakdown2.docx` 描述的系统:在**不重训 VLA、不引入外部
planner、不抢占主推理 GPU** 的前提下,为 RT-2 风格的 VLA 策略提供毫秒级安全恢复。

```
┌─────────────────── GPU-A 语义 (stream_main) ───────────────────┐
│  mini-VLA (RT-2 范式): RGB ──► CNN+Transformer ──► 4 action    │
│  token (256-bin) 自回归解码, ~6.7Hz "thinking"                  │
└────────────────────────────────────────────────────────────────┘
        │ 提议动作                                ▲ 恢复后无缝接管
        ▼                                        │
┌──────────────── 准入检查 (Oracle Monitor, CPU µs级) ────────────┐
│  几何冲突? ──否──► 放行 VLA token                                │
│      │是                                                        │
│      ▼                                                          │
│  Contextual MAB (UCB1/TS, training-free)                        │
│  arms = VLA 原生 token 序列: retreat_up / shift_y± / retreat_x  │
│         / freeze   (夹爪通道透传, 永不主动松爪)                  │
└────────────────────────────────────────────────────────────────┘
        ▲ 冲突报告                                │ recovery tokens
        │                                        ▼
┌────────── GPU-B 语义 (stream_safety, 高优先级) ─────────────────┐   ┌─────────────┐
│  腕部深度图 ─► pinned 异步上传 ─► 反投影+扫掠走廊检测 ─► 归约回传 │   │ seqlock SHM │──► C++ 200Hz
│  ~33Hz, cudaStreamCreateWithPriority 保证不被解码 kernel 淹没    │   │ (64B 单槽)  │    控制环
└────────────────────────────────────────────────────────────────┘   └─────────────┘
```

## 文档声明 → 实现映射

| 文档声明 | 本仓库实现 |
|---|---|
| RT-2 把动作离散成整数 bin 当 token 生成 | `vla/tokenizer.py` (256-bin) + `vla/model.py` (自回归 decoder) |
| 感知-策略解耦 (dual-T4) | `runtime/streams.py`: 单卡双 stream 优先级隔离 (VPEngine 同思路单卡版); `device_a/device_b` 参数保留真双卡形态 |
| 深度感知安全检查 | `perception/depth_safety.py`: 针孔反投影 + 自体/桌面/任务目标三重过滤 + 扫掠胶囊走廊 |
| training-free MAB Oracle | `safety/bandit.py` (UCB1/Thompson, tabular contextual) + `safety/monitor.py` (准入检查 + 二元 reward 在线累计) |
| recovery 绕过外部 planner, 直接是 action token | `safety/arms.py`: arms 经同一 tokenizer 编码, 与 VLA 共享动作词表 |
| 高优先级 CUDA stream + pinned 异步 + 无锁环 | `runtime/streams.py` + `runtime/ring.py` + `cpp/` (seqlock, Windows 命名共享内存, QPC 同源时钟跨进程测延迟) |
| thinking (1-10Hz) vs acting (50-200Hz) gap | `runtime/episode.py`: 推理延迟在仿真时间里显式建模 (150ms), 思考期间旧指令保持 |
| CBF-QP fallback 的 distribution shift 问题 | `baselines/qp_fallback.py`: 解析投影对照组 |

## 复现步骤

```bash
python scripts/00_setup_assets.py     # Panda 资产 + 腕部深度相机标定注入 + 几何自检
python scripts/01_gen_demos.py        # 340 条专家示范 (无障碍训练分布)
python scripts/02_train_vla.py        # mini-VLA 行为克隆 (~2.6M 参数)
python scripts/03_eval_baseline.py    # E1: VLA 单独 (3 suites x 60 集)
python scripts/04_run_safety.py       # E1/E3/E4: monitor 变体 + QP 对照
python scripts/05_bench_latency.py    # E2: 恢复链路墙钟延迟 (需先构建 C++)
python scripts/06_ablation_serial.py  # E6: 解耦 vs 串行控制环
python scripts/07_report.py           # 聚合 -> results/report.md
python tests/test_units.py            # 单元测试
```

C++ 构建 (VS2022 BuildTools):

```bash
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022"
cmake --build cpp/build --config Release
cpp/build/Release/test_seqlock.exe    # seqlock 协议压力自测
```

## 实验结果

见 [results/report.md](results/report.md)。

## 诚实声明 (与文档的偏差)

1. **单卡而非 dual-T4**: 本机只有一张 RTX 3060 (6GB)。解耦用 CUDA stream
   优先级实现 (文档引用的 VPEngine 即该思路的单卡版); 代码保留双卡参数。
2. **mini-VLA 而非 RT-2**: 55B 模型无法本地复现。保留范式 (token 离散化 +
   自回归解码 + 思考/行动频率差), 推理延迟在仿真时间显式建模为 150ms;
   GPU 争用实验用 `decode_load()` 负载仿真拉满 stream_main。
3. **深度来自仿真渲染**: 相当于理想深度传感器。延迟测量边界从 "深度帧已在
   主机内存" 开始 (等价于传感器 DMA 完成), 包含上传/核函数/回传/仲裁/写入。
4. **QP 对照是解析投影**: 单约束 CBF-QP 的闭式特例, 对照点是 distribution
   shift 而非求解器延迟。
5. **单腕部相机视野有限**: 横移类 recovery 的目标区域不在视野内 (盲移),
   由 reward 机制兜底; 多相机覆盖列为后续工作。
6. **环境差异**: 文档场景为 RT-2 级 VLA + 真机; 本复现为 MuJoCo 桌面抓取
   (Franka Panda, 立柱障碍 OOD), 数字趋势可比, 绝对值不可直接对标。

## 目录

```
configs/default.yaml        全部超参 (频率契约 / 走廊几何 / bandit / 评测)
src/vla_safety/
  env/        MuJoCo 场景 + 100Hz tick 环境 + 场景采样 + 脚本专家
  vla/        tokenizer / mini-VLA / 数据集 / 推理封装
  perception/ 深度反投影 + 走廊冲突检测 (GPU)
  safety/     recovery arms / contextual bandit / Oracle Monitor
  baselines/  CBF 风格解析投影 fallback
  runtime/    sim-time episode 调度器 / CUDA stream 装备 / seqlock 写端 / 评测执行器
cpp/          seqlock 头 + 200Hz 控制环消费端 + 协议压力测试
scripts/      00-07 全流程
tests/        单元测试 (无 mujoco 依赖)
```
