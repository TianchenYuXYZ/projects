"""
demo.py — end-to-end demonstration. One run shows:

  A. duplicate submission   -> idempotency key returns the existing dag_id
  B. worker crash mid-task  -> heartbeat expires, reaper XCLAIMs, a
                               replacement A100 worker reruns `train`
  C. flaky task (eval)      -> fails once, exponential-backoff retry, succeeds
  D. heterogeneous routing  -> L4 tasks keep flowing while `train` occupies
                               the big-GPU stream (no head-of-line blocking)

Pipeline (the classic PCB-Auto shape):

    ingest ──> preprocess ──┐
       └─────> augment   ───┴──> train ──┬──> eval_model ──> report
                            (A100|L40S)  └──> export_onnx ──┘
"""

import multiprocessing as mp
import time

from scheduler import Pipeline, task, worker_main, reaper_main
from scheduler.core import (
    DONE, get_redis, k_events, k_meta, k_node, k_nodes,
)

R = get_redis()

# ---------------------------------------------------------------- tasks --
@task(gpu="L4")
def ingest():
    time.sleep(0.5)

@task(gpu="L4", depends_on=[ingest])
def preprocess():
    time.sleep(1.0)

@task(gpu="L4", depends_on=[ingest])
def augment():
    time.sleep(1.0)

@task(gpu="A100|L40S", depends_on=[preprocess, augment])
def train():
    time.sleep(6.0)        # long enough for us to murder its worker

@task(gpu="L4", depends_on=[train], max_retries=3)
def eval_model():
    # Simulated flakiness: first attempt OOMs, retry succeeds.
    if get_redis().incr("demo:eval_tries") == 1:
        raise RuntimeError("simulated CUDA OOM")
    time.sleep(0.5)

@task(gpu="L4", depends_on=[train])
def export_onnx():
    time.sleep(0.5)

@task(gpu="L4", depends_on=[eval_model, export_onnx])
def report():
    time.sleep(0.3)


PIPELINE = Pipeline(ingest, preprocess, augment, train, eval_model, export_onnx, report)


# ---------------------------------------------------------------- infra --
def spawn(name, target, *args):
    p = mp.Process(target=target, args=args, name=name, daemon=True)
    p.start()
    return p


def wait_for_status(dag_id, node, status, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if R.hget(k_node(dag_id, node), "status") == status:
            return True
        time.sleep(0.05)
    return False


def print_summary(dag_id):
    print("\n========== final node states ==========")
    for n in sorted(R.smembers(k_nodes(dag_id))):
        h = R.hgetall(k_node(dag_id, n))
        print(f"  {n:<12} {h['status']:<8} attempts={h.get('attempts')} "
              f"owner={h.get('owner', '-')}")
    print(f"  DAG status: {R.hget(k_meta(dag_id), 'status')}")

    print("\n========== event timeline (events stream) ==========")
    t0 = None
    for _, e in R.xrange(k_events(dag_id)):
        ts = float(e["ts"])
        t0 = t0 or ts
        extra = {k: v for k, v in e.items() if k not in ("node", "type", "ts")}
        print(f"  +{ts - t0:6.2f}s  {e['node']:<12} {e['type']:<16} {extra or ''}")


# ----------------------------------------------------------------- main --
def main():
    R.flushdb()
    print("== starting reaper + workers ==")
    spawn("reaper", reaper_main)
    spawn("w", worker_main, "w-l4-1", "L4")
    spawn("w", worker_main, "w-l4-2", "L4")
    a100 = spawn("w", worker_main, "w-a100-1", "A100")
    time.sleep(0.5)

    print("\n== scenario A: submit + duplicate submit ==")
    dag_id = PIPELINE.submit(R, params={"dataset": "pcb-defects-v1"})
    PIPELINE.submit(R, params={"dataset": "pcb-defects-v1"})   # -> dedup hit

    print("\n== scenario B: kill the A100 worker mid-`train` ==")
    assert wait_for_status(dag_id, "train", "running"), "train never started"
    time.sleep(1.5)                       # let it run a bit, then crash it
    print(">>> killing w-a100-1 (simulated node failure)")
    a100.terminate()
    a100.join()
    time.sleep(1.0)
    print(">>> spawning replacement w-a100-2")
    spawn("w", worker_main, "w-a100-2", "A100")

    # scenario C (flaky eval) and D (L4 stream stays unblocked) happen
    # automatically as the DAG progresses.
    t0 = time.time()
    while time.time() - t0 < 60:
        if R.hget(k_meta(dag_id), "status") in ("done", "failed"):
            break
        time.sleep(0.2)

    print_summary(dag_id)


if __name__ == "__main__":
    main()
