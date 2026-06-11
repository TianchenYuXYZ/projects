"""
worker.py — a stateless worker node. There is NO central scheduler brain:
each worker, after finishing a task, advances the DAG itself ("chain
reaction") using atomic Redis ops. The only daemons besides workers are the
reaper and the retry dispatcher (see reaper.py).

Consume loop (step 4 of the flow):
  - declare hardware capability at startup (e.g. "L4")
  - discover all known tag streams and subscribe ONLY to matching ones
    (an L4 worker matches "L4" and "L4|*" alternations -> never sees A100
    work -> head-of-line blocking is structurally impossible)
  - XREADGROUP with a shared consumer group gives exactly-one-consumer
    delivery per message + a Pending Entries List (PEL) the reaper can
    XCLAIM from if this worker dies.

Fault-tolerance pieces implemented here:
  - heartbeat thread:    SET worker:{id}:hb EX ttl, every interval
  - execution lock:      SET dag:{d}:lock:{n} worker_id NX EX lease
                         -> guards against double-run when the reaper
                            falsely declares a slow-but-alive worker dead
  - exponential backoff: failed task -> ZADD delayed_retries with
                         score = now + base * 2^attempt + jitter
  - edge-level idempotency for the chain reaction:
                         SADD done_parents:{child} me  (returns 1 only once)
                         so a redelivered DONE task never double-DECRs.
"""

import json
import random
import threading
import time
import traceback

from .core import (
    DONE, FAILED, GROUP, K_DEAD, K_DELAYED, K_TAGS, READY, RETRY_WAIT, RUNNING,
    enqueue, event, get_redis,
    k_children, k_done_parents, k_hb, k_indeg, k_lock, k_meta, k_node,
    k_nodes, k_stream, K_ACTIVE,
)
from .dag import REGISTRY

HB_INTERVAL = 1.0   # seconds between heartbeats
HB_TTL = 3          # heartbeat key expiry -> "dead" after missing ~3 beats
LOCK_LEASE = 60     # execution lease; production would renew this periodically
RETRY_BASE = 1.0    # backoff base (seconds)
RETRY_JITTER = 0.5  # uniform jitter to avoid thundering-herd retries


class Worker:
    def __init__(self, worker_id, hardware):
        self.id = worker_id
        self.hardware = hardware            # e.g. "L4"
        self.r = None
        self._stop = threading.Event()

    # ---- liveness ------------------------------------------------------
    def _heartbeat_loop(self):
        hb = get_redis()                    # own connection: thread-safe & fork-safe
        while not self._stop.is_set():
            hb.set(k_hb(self.id), "1", ex=HB_TTL)
            time.sleep(HB_INTERVAL)

    def _matching_streams(self):
        """Subscribe to every tag stream whose alternation contains my hardware.
        e.g. hardware=L40S matches tags 'L40S' and 'A100|L40S'."""
        tags = [t for t in self.r.smembers(K_TAGS) if self.hardware in t.split("|")]
        return {k_stream(t): ">" for t in tags}

    # ---- main loop -------------------------------------------------------
    def run(self):
        self.r = get_redis()
        self.r.set(k_hb(self.id), "1", ex=HB_TTL)
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        print(f"[{self.id}] online (hardware={self.hardware})")

        while not self._stop.is_set():
            streams = self._matching_streams()
            if not streams:
                time.sleep(0.5)
                continue
            resp = self.r.xreadgroup(GROUP, self.id, streams, count=1, block=1000)
            for stream, msgs in resp or []:
                for msg_id, fields in msgs:
                    self._handle(stream, msg_id, fields["dag"], fields["node"])

    # ---- one delivery ----------------------------------------------------
    def _handle(self, stream, msg_id, dag_id, node):
        nodekey = k_node(dag_id, node)
        status = self.r.hget(nodekey, "status")

        # Redelivery of an already-finished task (worker died between
        # completing the work and XACK). The WORK must not rerun, but the
        # chain reaction MUST be replayed — it is idempotent via SADD guards.
        if status == DONE:
            self._chain_reaction(dag_id, node)
            self.r.xack(stream, GROUP, msg_id)
            return
        if status == FAILED:
            self.r.xack(stream, GROUP, msg_id)
            return

        # Execution lock: at-least-once delivery means duplicates are
        # possible (reaper false positive on a slow worker). First SET NX
        # wins; a loser only proceeds if the holder's heartbeat is gone.
        if not self.r.set(k_lock(dag_id, node), self.id, nx=True, ex=LOCK_LEASE):
            owner = self.r.get(k_lock(dag_id, node))
            if owner and self.r.exists(k_hb(owner)):
                # genuinely running elsewhere -> drop this duplicate
                self.r.xack(stream, GROUP, msg_id)
                return
            # stale lock from a dead worker -> steal it
            # (production: Lua compare-and-swap; demo keeps it simple)
            self.r.set(k_lock(dag_id, node), self.id, ex=LOCK_LEASE)

        self.r.hset(nodekey, mapping={"status": RUNNING, "owner": self.id,
                                      "started_at": time.time()})
        event(self.r, dag_id, node, "running", worker=self.id)
        print(f"[{self.id}] RUN  {dag_id}/{node}")

        try:
            REGISTRY[node]()                       # <- the actual user code
        except Exception as e:
            print(f"[{self.id}] FAIL {dag_id}/{node}: {e}")
            self._on_failure(stream, msg_id, dag_id, node, e)
        else:
            print(f"[{self.id}] DONE {dag_id}/{node}")
            self._on_success(stream, msg_id, dag_id, node)
        finally:
            self.r.delete(k_lock(dag_id, node))

    # ---- success path: mark done -> chain reaction -> ack ----------------
    def _on_success(self, stream, msg_id, dag_id, node):
        self.r.hset(k_node(dag_id, node),
                    mapping={"status": DONE, "finished_at": time.time()})
        event(self.r, dag_id, node, "done", worker=self.id)
        self._chain_reaction(dag_id, node)
        self._maybe_finish_dag(dag_id)
        self.r.xack(stream, GROUP, msg_id)   # ACK LAST: crash before this
        #                                      -> reaper redelivers
        #                                      -> DONE branch replays chain

    def _chain_reaction(self, dag_id, node):
        """Kahn's algorithm, distributed: for each child, atomically record
        'parent {node} is done' (SADD, fires exactly once per edge), and only
        then DECR the in-degree. DECR hitting 0 == all parents done -> route
        the child into its capability stream."""
        for child in self.r.smembers(k_children(dag_id, node)):
            if self.r.sadd(k_done_parents(dag_id, child), node) == 1:
                if self.r.decr(k_indeg(dag_id, child)) == 0:
                    gpu = self.r.hget(k_node(dag_id, child), "gpu")
                    enqueue(self.r, dag_id, child, gpu)

    def _maybe_finish_dag(self, dag_id):
        """Completion check by scanning node statuses (idempotent — safe if
        two last siblings race). O(N) per completion is fine at this scale."""
        nodes = self.r.smembers(k_nodes(dag_id))
        statuses = [self.r.hget(k_node(dag_id, n), "status") for n in nodes]
        if all(s == DONE for s in statuses):
            self.r.hset(k_meta(dag_id), mapping={"status": "done",
                                                 "finished_at": time.time()})
            self.r.srem(K_ACTIVE, dag_id)
            event(self.r, dag_id, None, "dag_done")

    # ---- failure path: backoff retry or dead-letter -----------------------
    def _on_failure(self, stream, msg_id, dag_id, node, exc):
        nodekey = k_node(dag_id, node)
        attempts = self.r.hincrby(nodekey, "attempts", 1)
        max_retries = int(self.r.hget(nodekey, "max_retries"))
        # We OWN this failure (it's an application error, not a worker death),
        # so we ACK the stream message — retry goes through the ZSET instead.
        self.r.xack(stream, GROUP, msg_id)

        if attempts > max_retries:
            self.r.hset(nodekey, "status", FAILED)
            self.r.hset(k_meta(dag_id), "status", "failed")
            self.r.srem(K_ACTIVE, dag_id)
            self.r.xadd(K_DEAD, {"dag": dag_id, "node": node, "error": str(exc)})
            event(self.r, dag_id, node, "dead_letter", error=str(exc))
            return

        delay = RETRY_BASE * (2 ** attempts) + random.uniform(0, RETRY_JITTER)
        retry_at = time.time() + delay
        self.r.hset(nodekey, "status", RETRY_WAIT)
        self.r.zadd(K_DELAYED, {f"{dag_id}|{node}": retry_at})
        event(self.r, dag_id, node, "retry_scheduled",
              attempt=attempts, delay=f"{delay:.1f}s")
        print(f"[{self.id}] retry {node} in {delay:.1f}s (attempt {attempts})")

    def stop(self):
        self._stop.set()


def worker_main(worker_id, hardware):
    """Entry point for multiprocessing.Process."""
    Worker(worker_id, hardware).run()
