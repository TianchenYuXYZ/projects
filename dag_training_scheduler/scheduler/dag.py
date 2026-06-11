"""
dag.py — the user-facing API: @task decorator + Pipeline.submit().

Lifecycle covered here (steps 1-3 of the flow):
  1. DEFINE   — researcher decorates plain functions with resource tags + deps.
  2. PARSE    — submit() runs a cycle check (Kahn) and extracts the topology.
  3. PERSIST  — the whole DAG is written into Redis in one pipeline:
                node hashes, in-degree counters, adjacency sets.
                Roots (in-degree 0) are seeded into their tag streams.

Idempotent submission:
  The submission checksum = sha256(canonical spec + params). Re-submitting a
  semantically identical pipeline returns the EXISTING dag_id instead of
  creating a duplicate run (prevents two jobs clobbering one checkpoint path).
"""

import hashlib
import json
import time
import uuid

from .core import (
    K_ACTIVE, K_IDEM, PENDING, enqueue, event,
    k_children, k_indeg, k_meta, k_node, k_nodes,
)

# Global registry: task name -> python callable.
# Workers import the same module, so they can resolve names back to functions.
REGISTRY = {}


class TaskSpec:
    def __init__(self, name, gpu, deps, max_retries):
        self.name, self.gpu, self.deps, self.max_retries = name, gpu, deps, max_retries


def task(gpu="L4", depends_on=None, max_retries=3):
    """Declare a pipeline node. `gpu` may be a single tag ("A100") or an
    alternation ("A100|L40S") — the alternation is itself a stream tag that
    multiple worker types subscribe to."""
    def deco(fn):
        deps = [d._spec.name for d in (depends_on or [])]
        fn._spec = TaskSpec(fn.__name__, gpu, deps, max_retries)
        REGISTRY[fn.__name__] = fn
        return fn
    return deco


class Pipeline:
    def __init__(self, *tasks):
        self.specs = [t._spec for t in tasks]

    # -- step 2: static validation (cycle check via Kahn's algorithm) ------
    def _validate(self):
        names = {s.name for s in self.specs}
        indeg = {s.name: 0 for s in self.specs}
        children = {s.name: [] for s in self.specs}
        for s in self.specs:
            for d in s.deps:
                if d not in names:
                    raise ValueError(f"{s.name} depends on unknown task {d}")
                indeg[s.name] += 1
                children[d].append(s.name)
        queue = [n for n, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if seen != len(self.specs):
            raise ValueError("cycle detected in pipeline DAG")

    def _checksum(self, params):
        canonical = json.dumps(
            {
                "tasks": sorted(
                    [[s.name, s.gpu, sorted(s.deps), s.max_retries] for s in self.specs]
                ),
                "params": params,  # stands in for dataset path + config + git SHA
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -- step 3: persist to Redis & seed roots ------------------------------
    def submit(self, r, params=None):
        self._validate()
        checksum = self._checksum(params or {})

        # Idempotent submission: HSETNX wins exactly once per checksum.
        dag_id = f"dag-{uuid.uuid4().hex[:8]}"
        if not r.hsetnx(K_IDEM, checksum, dag_id):
            existing = r.hget(K_IDEM, checksum)
            print(f"[submit] duplicate submission -> returning existing {existing}")
            return existing

        pipe = r.pipeline()
        pipe.hset(k_meta(dag_id), mapping={"status": "running", "submitted_at": time.time()})
        pipe.sadd(K_ACTIVE, dag_id)
        for s in self.specs:
            pipe.sadd(k_nodes(dag_id), s.name)
            pipe.hset(k_node(dag_id, s.name), mapping={
                "status": PENDING,
                "gpu": s.gpu,
                "deps": json.dumps(s.deps),
                "attempts": 0,
                "max_retries": s.max_retries,
            })
            pipe.set(k_indeg(dag_id, s.name), len(s.deps))
            for d in s.deps:
                pipe.sadd(k_children(dag_id, d), s.name)
        pipe.execute()
        event(r, dag_id, None, "submitted", checksum=checksum[:12])

        # Seed: every root (in-degree 0) goes straight into its tag stream.
        for s in self.specs:
            if not s.deps:
                enqueue(r, dag_id, s.name, s.gpu)

        print(f"[submit] {dag_id} submitted ({len(self.specs)} tasks)")
        return dag_id
