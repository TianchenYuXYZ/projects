# redis-dag-scheduler — 简明中英说明

这是一个最小可运行的 DAG 调度器示例，把任务依赖关系和调度状态保存在 Redis 中，演示按硬件能力分流、容错重试与幂等提交的核心思路。

This is a minimal, runnable example of a DAG scheduler that stores task dependencies and runtime state in Redis. It demonstrates capability-based stream routing, fault-tolerant retries, and idempotent submissions.

---

## 简短介绍（中文）

- 目的：演示如何用 Redis 做一个轻量级的分布式 DAG 调度器。
- 核心点：按 GPU 能力把任务推到不同的 Stream，worker 只拉自己能处理的流；没有中心调度器，worker 自行推进 DAG；reaper 负责回收死掉的任务并重试。

## Short intro (English)

- Purpose: show how to build a lightweight distributed DAG scheduler using Redis.
- Core ideas: push tasks into different Streams by GPU capability; workers only pull streams they can handle; there is no central scheduler — workers advance the DAG themselves; a reaper process recovers crashed work and retries tasks.

- Goal: a minimal DAG scheduler using Redis for state and streams.
- Key ideas: capability-tagged streams (one stream per GPU tag), workers pull matching streams, no central scheduler — workers advance the DAG, a reaper recovers crashed tasks.

---

## 快速上手 / Quick start

确保 Redis 正在运行，然后安装依赖并运行 demo：

```bash
redis-server --daemonize yes
pip install redis
python demo.py
```

Demo 运行会展示：重复提交的幂等（duplicate submit）、worker 崩溃后由 reaper 回收并重跑、失败任务的指数退避重试、以及不同能力流互不阻塞。

## Quick start (English)

The demo shows idempotent duplicate submission, reaper recovery of crashed workers, exponential-backoff retries for failing tasks, and capability-based stream isolation so that different GPU streams do not block each other.

---

## demo 流程说明

1. 启动组件：主进程会清空 Redis，然后启动 `reaper`（守护进程）和若干 `worker` 进程（例如两个 L4、一个 A100）。
2. 提交 DAG：`Pipeline.submit()` 会检查无环、计算 checksum（用于幂等）、把每个节点写入 Redis（节点信息、入度计数、后继集合），并把没有依赖的根节点推入对应的 tag stream。
3. worker 拉取任务并执行：worker 从自己关注的 stream 拉任务，拿到任务后申请执行锁并写心跳，执行成功后把自己标记为 DONE，并对后继节点做“边级幂等的 DECR”；当后继的入度归零时，入队运行。
4. 故障恢复（reaper）：如果 worker 崩溃或心跳超时，`reaper` 会通过 Redis 的 PEL/XPENDING + `XCLAIM` 把未完成的消息抢回并重新分配给活着的 worker。
5. 失败重试：任务执行失败会进入延迟重试的 ZSET（指数退避 + 抖动），到达重试时间后再入队；超过最大重试次数则落到 dead-letter 流。

## Demo flow (English)

1. Start components: the main process clears Redis, then spawns a `reaper` daemon and several `worker` processes (e.g. two L4 workers and one A100 worker).
2. Submit DAG: `Pipeline.submit()` validates the DAG (no cycles), computes a checksum for idempotency, writes node metadata and indegree counters to Redis, and seeds root nodes into the appropriate tag streams.
3. Worker execution: a worker pulls messages from streams it subscribes to, obtains an execution lease and writes heartbeats, runs the job, marks the node DONE, and performs an edge-level idempotent DECR on its children; when a child's indegree reaches zero it is enqueued.
4. Fault recovery (reaper): if a worker crashes or heartbeats expire, the `reaper` scans PEL/XPENDING and uses `XCLAIM` to claim pending messages and reassign them to live workers.
5. Retry on failure: failed tasks are parked in a delayed-retry ZSET with exponential backoff and jitter; when the retry time arrives the task is re-enqueued, and tasks that exceed max retries go to the dead-letter stream.

---

## 关键概念

- 幂等提交：提交时计算 DAG 的 checksum，同样的 DAG 不会重复创建新的 run；
- 能力标签流：每个 GPU 类型对应一个 Redis Stream，worker 只看匹配的流，避免不同能力任务互相阻塞；
- 分布式 Kahn：没有中央计算节点，任意完成任务的 worker 都会尝试把它的后继节点的入度减一，并在入度为 0 时把后继入队；
- Reaper：负责检测死掉的 worker（心跳过期）、抢回消息并重试。

## Key concepts (English)

- Idempotent submission: the DAG checksum prevents creating duplicate runs for the same pipeline.
- Capability-tagged streams: one Redis Stream per GPU capability; workers only observe matching streams, avoiding head-of-line blocking across different hardware.
- Distributed Kahn: there is no central orchestrator — any worker that finishes a task applies an idempotent DECR to its children and enqueues them when indegree hits zero.
- Reaper: detects dead workers (heartbeat expiry), reclaims pending messages, and triggers retries.

---

## 演示要点（demo 中展示的场景）

- 场景 A：重复提交 -> 幂等 key 命中，返回已有 `dag_id`；
- 场景 B：在 `train` 运行时杀掉 A100 worker -> reaper 回收任务并交给新 worker 重跑；
- 场景 C：`eval_model` 第一次故意失败 -> 指数退避后重试成功；
- 场景 D：L4 与 A100 流物理隔离，L4 任务不会被 A100 上的长任务阻塞。

## Demo highlights (English)

- Scenario A: duplicate submit -> idempotency key hit, returns existing `dag_id`.
- Scenario B: kill the A100 worker while `train` is running -> reaper reclaims the in-flight task and a replacement worker reruns it.
- Scenario C: `eval_model` fails on first attempt -> exponential backoff retry succeeds on a later attempt.
- Scenario D: L4 and A100 streams are isolated, so L4 tasks continue even while a long A100 job occupies that stream.

---

## 重要保证（简明）

- 系统以 at-least-once 投递为基础，但通过幂等设计（状态检查、执行锁、边级幂等）实现实际上的“近似 exactly-once”效果；
- 任务完成后才 XACK，任何步骤崩溃都会被 reaper 拿回并重试，避免子任务永久丢失；
- 使用全量扫描判定完成（O(N)）以换取简单的幂等安全性。

## Guarantees (English)

- The system is built on at-least-once delivery but achieves near exactly-once effects via idempotency: state checks, execution leases, and edge-level idempotent guards.
- ACKs are written only after marking DONE and performing chain reaction; if a crash occurs the message remains in the PEL and the reaper will retry, preventing permanent loss of children tasks.
- Completion is determined via an idempotent full-scan (O(N)) to avoid counter races during crashes.

---

## 代码文件速览

- `scheduler/dag.py` — `@task` 装饰器、Pipeline 定义、DAG 校验与提交；
- `scheduler/core.py` — Redis key 设计、enqueue 与事件记录；
- `scheduler/worker.py` — worker 主循环、任务执行、链式推进逻辑；
- `scheduler/reaper.py` — 死 worker 检测、XCLAIM 抢回、延迟重试扫描；
- `demo.py` — 演示脚本，启动 reaper + workers 并提交一个示例 pipeline。

## Files at a glance (English)

- `scheduler/dag.py` — `@task` decorator, Pipeline definition, DAG validation and submission;
- `scheduler/core.py` — Redis key schema, enqueue logic and event logging;
- `scheduler/worker.py` — worker main loop, execution, and chain-reaction logic;
- `scheduler/reaper.py` — dead worker detection, `XCLAIM` reclamation, delayed-retry scanning;
- `demo.py` — demo script that starts reaper + workers and submits an example pipeline.

---

## 生产化注意事项（简短）

- 执行锁应使用 Lua 的 compare-and-swap 保证原子性并实现租约续期；
- 长任务需要续约租约或分片；
- 需要添加监控导出（Prometheus）、任务参数序列化与结果持久化、并基于实际 GPU 资源做装箱/调度策略。

## Production notes (English)

- Use Lua compare-and-swap for execution locks to ensure atomicity and support renewing leases for long-running tasks.
- Long-running tasks should renew leases or be sharded; consider job chunking for safety.
- Add monitoring (Prometheus), serialize task parameters and persist results (artifact store), and implement packing/placement logic based on real GPU memory and topology.

---


如果你希望，我可以：

- 把 `print_summary()` 的示例输出写到 README；
- 生成 `requirements.txt` 并在本地尝试运行 `demo.py`（需要本地 Redis）；
- 或者把某部分画成时间线图帮助理解。

If you want, I can:

- Add an example `print_summary()` output to the README;
- Generate a `requirements.txt` and try running `demo.py` locally (requires a local Redis instance);
- Or draw a timeline/sequence diagram to illustrate key flows.

Thanks!

