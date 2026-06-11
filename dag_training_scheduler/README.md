# redis-dag-scheduler — minimal runnable reconstruction

~550 行的最小可运行版本,复刻 PCB Auto 实习项目的核心机制:
DAG 拓扑调度 + capability-tagged streams(消 HOL blocking)+ 容错三件套。

## 运行

```bash
redis-server --daemonize yes
pip install redis
python3 demo.py
```

demo 一次运行演示四件事:
- **A** 重复提交 → 幂等 key 命中,返回已有 dag_id
- **B** `train` 跑到一半杀掉 A100 worker → 心跳 3s 过期 → reaper XCLAIM 回收重发 → 替补 worker 重跑(attempts 不增加:crash ≠ 应用失败)
- **C** `eval_model` 第一次模拟 OOM → exponential backoff(~2s)→ 重试成功
- **D** L4 任务流与 A100|L40S 流物理隔离,互不阻塞

## 架构

```
 researcher                workers (无中央调度大脑)              daemons
┌──────────┐   submit   ┌─────────────────────────────┐   ┌──────────────┐
│ @task    │──────────▶ │            Redis            │◀──│   reaper     │
│ Pipeline │  DAG→Redis │  dag:{id}:node:{n}   Hash   │   │ ① 死worker回收│
└──────────┘            │  dag:{id}:indeg:{n}  Counter│   │   XPENDING+   │
                        │  dag:{id}:children   Set    │   │   心跳判定+    │
      worker 完成任务后  │  stream:gpu:{tag}    Stream │   │   XCLAIM      │
      自己推进 DAG:      │  zset:delayed_retries ZSET  │   │ ② 到期重试    │
   ┌────────────────┐   │  worker:{id}:hb      TTL key│   │   ZRANGEBYSCORE│
   │ SADD edge-guard│   │  stream:dead_letter  Stream │   │ ③ orphan sweep│
   │ → DECR indeg   │   └─────────────────────────────┘   │   自愈兜底     │
   │ → ==0 则 XADD  │                                     └──────────────┘
   └────────────────┘
```

## 文件

| 文件 | 机制 | 面试 talking point |
|---|---|---|
| `scheduler/dag.py` | `@task` decorator、Kahn cycle check、整图写入 Redis、根节点 seed | "Pipeline 表达为 DAG,submit 时序列化成 Hash + in-degree counter + 邻接 Set" |
| `scheduler/core.py` | `enqueue()` 按 gpu tag 路由到对应 Stream | "Virtual output queueing:按 capability 切流,L4 worker 永远看不见 A100 任务" |
| `scheduler/worker.py` | `_matching_streams()` + XREADGROUP | "Worker 启动声明 capability,主动 pull 匹配的 stream" |
| `scheduler/worker.py` | `_chain_reaction()`:SADD edge-guard → 原子 DECR → ==0 入队 | "分布式 Kahn:每个 worker 完成后顺手做减法,没有中央大脑" |
| `scheduler/worker.py` | 执行锁 `SET NX EX` + 心跳判活后偷锁 | "Idempotent job key 防 reaper 误判导致的双跑" |
| `scheduler/worker.py` | `_on_failure()`:ZADD backoff+jitter,超限进 dead-letter | "失败走 Sorted Set 延迟队列,jitter 防惊群" |
| `scheduler/reaper.py` | XPENDING → 查心跳 → XCLAIM → 重发 | "死亡由 TTL 心跳判定,任务从 PEL 里抢回来" |
| `scheduler/reaper.py` | `orphan_sweep()`:PENDING 且所有 parent DONE → 补发 | "DECR 和 XADD 非原子,sweep 从第一性原理重算 readiness 兜底" |
| `scheduler/core.py` | `events:{dag}` lifecycle stream | "Observability:每个状态流转写事件流,dashboard 直接消费" |

## 关键正确性论证

**At-least-once + 幂等 = 实际 exactly-once 效果。** 投递可能重复
(reaper 误判、crash 后重发),靠三层守卫吸收:
1. 状态检查:DONE/FAILED 的重复投递直接 ACK 丢弃(但 DONE 会重放 chain reaction);
2. 执行锁:`SET NX`,持有者心跳存活则丢弃重复投递;
3. edge-guard:`SADD done_parents` 每条边只触发一次 DECR,重放安全。

**ACK 放在最后。** 完成顺序:标记 DONE → chain reaction → 完成检查 → XACK。
任何一步崩溃,消息留在 PEL,reaper 重投,DONE 分支幂等重放 chain reaction
——不会有 child 被永久遗漏。

**完成判定用全量扫描而非计数器。** O(N) 但幂等,两个最后兄弟节点
race 也无害。计数器版本在 crash 窗口下会少减/多减。

## 与生产版本的差距(被问 "如何生产化" 时的答案)

- 偷锁应该用 Lua compare-and-swap,这里是裸 SET(有微小 race 窗口)
- 长任务的执行锁需要租约续期(本版固定 60s lease)
- 没有 priority / aging(starvation 防护),没有 gang scheduling
- 没有 Prometheus exporter(events stream 已留好接口)
- task payload 只有函数名,生产版应带序列化参数 + 结果存储(artifact store)
- worker 容量为 1(一次一个任务),生产版按 GPU 显存做装箱
