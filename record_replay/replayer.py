"""Replays a recorded Gaussian-elimination session one step at a time.

Public API
----------
replayer = Replayer(data_dir)
replayer.load_session(session_id)   → None
replayer.get_step(n)                → scipy.sparse.csr_matrix
replayer.total_steps()              → int
replayer.event_at(n)                → dict | None

Step semantics
--------------
Step 0  = initial matrix (no operations applied, from initial.msgpack)
Step N  = state after N operations have been applied
event_at(N) returns the event that caused the N-1 → N transition.

Checkpoint strategy
-------------------
get_step() finds the largest checkpoint C ≤ N, loads it as a baseline, then
replays events C..N-1 forward.  If the internal cache is already at a step M
with C ≤ M ≤ N, it uses M as the baseline instead, avoiding re-loading from
disk.  Sequential playback therefore costs exactly one event per step.

The working matrix during replay is kept as scipy.sparse.lil_matrix, which
supports efficient in-place row mutations.  The caller receives a CSR matrix;
dense conversion happens only in the viewer layer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import scipy.sparse as sp

from storage import read_msgpack


class Replayer:
    def __init__(self, data_dir: str | Path):
        self._data_dir = Path(data_dir)
        self._session_dir: Optional[Path] = None
        self._initial: Optional[sp.csr_matrix] = None
        self._events: list[dict] = []
        self._ckpt_steps: list[int] = []   # sorted checkpoint step numbers on disk
        self._cache_step: Optional[int] = None
        self._cache_mat: Optional[sp.csr_matrix] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_session(self, session_id: str) -> None:
        """Load all session data from disk; resets the internal cache."""
        self._session_dir = self._data_dir / f"session_{session_id}"
        self._initial = read_msgpack(self._session_dir / "initial.msgpack")
        self._events = read_msgpack(self._session_dir / "commands.msgpack")

        self._ckpt_steps = sorted(
            int(m.group(1))
            for p in self._session_dir.iterdir()
            if (m := re.fullmatch(r"checkpoint_(\d+)\.msgpack", p.name))
        )

        # Prime the cache at step 0 so the first get_step(0) is a hit
        self._cache_step = 0
        self._cache_mat = self._initial.copy()

    def total_steps(self) -> int:
        """Number of recorded events (slider runs 0 … total_steps)."""
        return len(self._events)

    def get_step(self, n: int) -> sp.csr_matrix:
        """Return the sparse CSR matrix after n operations have been applied.

        Never converts to dense; that is deferred to the rendering layer.
        """
        n = max(0, min(n, self.total_steps()))

        if self._cache_step == n:
            return self._cache_mat

        # Largest checkpoint step ≤ n  (step 0 == initial.msgpack)
        best_ckpt = 0
        for s in self._ckpt_steps:
            if s <= n:
                best_ckpt = s
            else:
                break

        # Prefer the cache over re-loading from disk when it is a valid start
        if (self._cache_step is not None
                and best_ckpt <= self._cache_step <= n):
            mat = self._cache_mat.tolil()
            start = self._cache_step
        else:
            mat = self._load_checkpoint(best_ckpt).tolil()
            start = best_ckpt

        for i in range(start, n):
            _apply_event(mat, self._events[i])

        result = mat.tocsr()
        self._cache_step = n
        self._cache_mat = result
        return result

    def event_at(self, n: int) -> Optional[dict]:
        """Event that caused the (n-1) → n transition, or None for step 0."""
        if n <= 0 or n > len(self._events):
            return None
        return self._events[n - 1]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_checkpoint(self, step: int) -> sp.csr_matrix:
        if step == 0:
            return self._initial.copy()
        return read_msgpack(self._session_dir / f"checkpoint_{step}.msgpack")


# ---------------------------------------------------------------------------
# Row-operation engine — works on lil_matrix in-place for full sparsity
# ---------------------------------------------------------------------------

def _apply_event(mat: sp.lil_matrix, ed: dict) -> None:
    etype = ed["event_type"]
    p = ed["params"]

    if etype == "swap_rows":
        i, j = p["row_i"], p["row_j"]
        mat.rows[i], mat.rows[j] = mat.rows[j], mat.rows[i]
        mat.data[i], mat.data[j] = mat.data[j], mat.data[i]

    elif etype == "scale_row":
        row, scalar = p["row"], p["scalar"]
        mat.data[row] = [v * scalar for v in mat.data[row]]

    elif etype == "add_scaled_row":
        target, source, scalar = p["target"], p["source"], p["scalar"]
        # Merge scalar * source into target using a column-keyed dict so that
        # column index ordering and fill-in are handled correctly.
        tgt = dict(zip(mat.rows[target], mat.data[target]))
        for col, val in zip(mat.rows[source], mat.data[source]):
            tgt[col] = tgt.get(col, 0.0) + scalar * val
        # Drop numerical zeros (same threshold as the GE generator)
        tgt = {c: v for c, v in tgt.items() if abs(v) > 1e-14}
        if tgt:
            pairs = sorted(tgt.items())
            mat.rows[target] = [c for c, _ in pairs]
            mat.data[target] = [v for _, v in pairs]
        else:
            mat.rows[target] = []
            mat.data[target] = []

    elif etype == "xor_rows":
        target, source = p["target"], p["source"]
        # Symmetric difference of column sets: cols in exactly one of the two
        # rows become 1; cols in both cancel to 0 (F₂ XOR semantics).
        tgt_cols = set(mat.rows[target])
        src_cols = set(mat.rows[source])
        result = sorted(tgt_cols.symmetric_difference(src_cols))
        mat.rows[target] = result
        mat.data[target] = [1.0] * len(result)
