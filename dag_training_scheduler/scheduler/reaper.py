"""
reaper.py — the only daemon besides workers. Three duties, one loop:

  1. reap_dead_workers
     For every tag stream, inspect the consumer group's Pending Entries
     List (XPENDING). A pending entry whose consumer's heartbeat key has
     EXPIRED belongs to a dead worker -> XCLAIM it, re-enqueue a fresh
     message into the tag stream, ACK the claimed one.
     ("Death" is decided by the TTL heartbeat, not by message idle time.)

  2. dispatch_due_retries
     Poll the delayed-retry ZSET: ZRANGEBYSCORE -inf now. ZREM is the
     atomic claim — whichever dispatcher removes the member owns the
     re-enqueue, so running two reapers is safe.

  3. orphan_sweep  (the self-healing net)
     The fast path (DECR -> XADD in the worker) is two non-atomic steps;
     a crash in between would strand a child forever. Instead of a Lua
     script, this sweep recomputes readiness FROM FIRST PRINCIPLES:
       node status == PENDING  and  all parents' status == DONE
     -> enqueue it. Any counter corruption or lost XADD is repaired within
     one sweep period. (Duplicate enqueues this might cause are absorbed
     by the worker's status check + execution lock.)
"""

import time

from .core import (
    DONE, GROUP, K_ACTIVE, K_DELAYED, K_TAGS, PENDING,
    deps_of, enqueue, event, get_redis,
    k_hb, k_node, k_nodes, k_stream,
)

POLL = 1.0  # seconds between reaper passes


def reap_dead_workers(r):
    for tag in r.smembers(K_TAGS):
        stream = k_stream(tag)
        try:
            pending = r.xpending_range(stream, GROUP, min="-", max="+", count=100)
        except Exception:
            continue
        for entry in pending:
            consumer, msg_id = entry["consumer"], entry["message_id"]
            if consumer == "reaper" or r.exists(k_hb(consumer)):
                continue  # holder is alive (or it's us)
            claimed = r.xclaim(stream, GROUP, "reaper",
                               min_idle_time=0, message_ids=[msg_id])
            for mid, fields in claimed:
                dag_id, node = fields["dag"], fields["node"]
                status = r.hget(k_node(dag_id, node), "status")
                if status != DONE:
                    print(f"[reaper] worker {consumer} dead -> requeue {dag_id}/{node}")
                    event(r, dag_id, node, "reclaimed", dead_worker=consumer)
                    gpu = r.hget(k_node(dag_id, node), "gpu")
                    enqueue(r, dag_id, node, gpu)
                r.xack(stream, GROUP, mid)


def dispatch_due_retries(r):
    for member in r.zrangebyscore(K_DELAYED, "-inf", time.time()):
        if r.zrem(K_DELAYED, member):          # atomic claim
            dag_id, node = member.split("|", 1)
            gpu = r.hget(k_node(dag_id, node), "gpu")
            print(f"[reaper] retry due -> requeue {dag_id}/{node}")
            enqueue(r, dag_id, node, gpu)


def orphan_sweep(r):
    for dag_id in r.smembers(K_ACTIVE):
        for node in r.smembers(k_nodes(dag_id)):
            if r.hget(k_node(dag_id, node), "status") != PENDING:
                continue
            parents = deps_of(r, dag_id, node)
            if parents and all(
                r.hget(k_node(dag_id, p), "status") == DONE for p in parents
            ):
                print(f"[reaper] orphan repaired -> enqueue {dag_id}/{node}")
                event(r, dag_id, node, "orphan_repaired")
                gpu = r.hget(k_node(dag_id, node), "gpu")
                enqueue(r, dag_id, node, gpu)


def reaper_main():
    r = get_redis()
    print("[reaper] online")
    while True:
        reap_dead_workers(r)
        dispatch_due_retries(r)
        orphan_sweep(r)
        time.sleep(POLL)
