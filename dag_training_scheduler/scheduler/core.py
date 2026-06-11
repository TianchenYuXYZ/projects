"""
core.py — Redis key schema, states, and shared primitives.

Design notes (interview mapping):
  * Every piece of scheduler state lives in Redis under a small, flat key
    schema. All transitions use O(1) atomic primitives (HSET / DECR / SADD /
    XADD / ZADD) — no central scheduler process owns the state.
  * Streams are partitioned BY GPU CAPABILITY TAG ("virtual output queueing"):
    one stream per tag, e.g. stream:gpu:L4, stream:gpu:A100|L40S.
    This is what kills head-of-line blocking.
"""

import json
import os
import time

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# ---- task / dag states -------------------------------------------------
PENDING = "pending"        # in the DAG, dependencies not yet satisfied
READY = "ready"            # in-degree hit 0, sitting in a tag stream
RUNNING = "running"        # claimed by a worker
RETRY_WAIT = "retry_wait"  # failed, parked in the delayed-retry ZSET
DONE = "done"
FAILED = "failed"          # exhausted retries -> dead-letter

GROUP = "workers"          # single consumer group name shared by all streams

# ---- key schema ---------------------------------------------------------
def k_meta(dag):           return f"dag:{dag}:meta"            # Hash: status, submitted_at
def k_node(dag, n):        return f"dag:{dag}:node:{n}"        # Hash: status, gpu, deps, attempts...
def k_indeg(dag, n):       return f"dag:{dag}:indeg:{n}"       # String counter (Kahn's algorithm)
def k_children(dag, n):    return f"dag:{dag}:children:{n}"    # Set: adjacency list
def k_done_parents(dag, n):return f"dag:{dag}:done_parents:{n}"# Set: edge-level idempotency guard
def k_nodes(dag):          return f"dag:{dag}:nodes"           # Set: all node names in this dag
def k_stream(tag):         return f"stream:gpu:{tag}"          # Stream: per-capability work queue
def k_hb(worker):          return f"worker:{worker}:hb"        # String w/ TTL: liveness heartbeat
def k_lock(dag, n):        return f"dag:{dag}:lock:{n}"        # String w/ TTL: execution lease
def k_events(dag):         return f"events:{dag}"              # Stream: lifecycle event log

K_TAGS = "set:known_tags"          # Set of every tag stream ever created (reaper iterates this)
K_ACTIVE = "set:active_dags"       # Set of dags not yet done/failed (sweep iterates this)
K_DELAYED = "zset:delayed_retries" # ZSET member="{dag}|{node}" score=retry_at timestamp
K_DEAD = "stream:dead_letter"      # Stream of permanently failed tasks (alerting hook)
K_IDEM = "hash:idempotency"        # Hash: submission checksum -> dag_id


def get_redis():
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def ensure_group(r, stream):
    """Create the consumer group from id 0 so no message is ever skipped."""
    try:
        r.xgroup_create(stream, GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def enqueue(r, dag_id, node, tag):
    """PENDING/RETRY_WAIT -> READY: route the task into its capability stream.

    This is the 'router' step. Note the task is pushed to the stream named
    after ITS resource tag — an L4 worker will never even see this message
    if the tag doesn't match its hardware.
    """
    stream = k_stream(tag)
    r.sadd(K_TAGS, tag)
    ensure_group(r, stream)
    r.hset(k_node(dag_id, node), mapping={"status": READY, "ready_at": time.time()})
    r.xadd(stream, {"dag": dag_id, "node": node})
    event(r, dag_id, node, "ready", tag=tag)


def event(r, dag_id, node, etype, **extra):
    """Observability: append-only lifecycle log per DAG (consumed by dashboards)."""
    fields = {"node": node or "-", "type": etype, "ts": f"{time.time():.3f}"}
    fields.update({k: str(v) for k, v in extra.items()})
    r.xadd(k_events(dag_id), fields)


def node_status(r, dag_id, node):
    return r.hget(k_node(dag_id, node), "status")


def deps_of(r, dag_id, node):
    raw = r.hget(k_node(dag_id, node), "deps")
    return json.loads(raw) if raw else []
