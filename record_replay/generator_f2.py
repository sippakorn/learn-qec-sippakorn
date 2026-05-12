"""F₂ Gaussian elimination as a recording event generator.

Mirrors forward_eliminate() from core/gaussian_elimination.py exactly,
emitting events for every row operation on the augmented matrix H_aug = [H|s].

Events emitted:
  swap_rows  — pivot row swapped into current position
  xor_rows   — source row XORed into target row (F₂ elimination)
"""

import time

import numpy as np
import scipy.sparse as sp

from events import Event
from recorder import Recorder, CHECKPOINT_INTERVAL


class F2GaussianEliminationGenerator:
    def __init__(self, recorder: Recorder, H: np.ndarray, s: np.ndarray):
        self._recorder = recorder
        n_vars = H.shape[1]
        self._n_vars = n_vars

        # Build augmented matrix [H | s] as float64 for replayer compatibility
        H_aug = np.hstack((H, s[:, np.newaxis])).astype(np.float64)
        self._mat = H_aug
        self._nrows = H_aug.shape[0]
        self._step = 0

    # ------------------------------------------------------------------
    # Row operations — emit event then mutate
    # ------------------------------------------------------------------

    def _swap_rows(self, i: int, j: int) -> None:
        self._recorder.record(Event(
            event_id=-1,
            event_type="swap_rows",
            params={"row_i": i, "row_j": j},
            timestamp=time.time(),
            step=self._step,
        ))
        self._mat[[i, j]] = self._mat[[j, i]]
        self._step += 1
        self._maybe_checkpoint()

    def _xor_rows(self, target: int, source: int) -> None:
        self._recorder.record(Event(
            event_id=-1,
            event_type="xor_rows",
            params={"target": target, "source": source},
            timestamp=time.time(),
            step=self._step,
        ))
        self._mat[target] = (self._mat[target] + self._mat[source]) % 2
        self._step += 1
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        if self._step % CHECKPOINT_INTERVAL == 0:
            self._recorder.checkpoint(sp.csr_matrix(self._mat), self._step)

    # ------------------------------------------------------------------
    # Algorithm — mirrors forward_eliminate() from gaussian_elimination.py
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[int], list[int]]:
        """Run Gauss-Jordan elimination over F₂ on H_aug, recording every op.

        Returns (pivot_cols, free_cols) matching gaussian_elimination_f2().
        """
        pivot_cols = []
        current_row = 0

        for col in range(self._n_vars):
            # Find pivot row: first row >= current_row with a 1 in this column
            pivot_row = None
            for row in range(current_row, self._nrows):
                if self._mat[row, col] == 1.0:
                    pivot_row = row
                    break

            if pivot_row is None:
                continue  # free variable column — skip

            # Swap pivot into current position
            if pivot_row != current_row:
                self._swap_rows(current_row, pivot_row)

            # Eliminate this column from ALL other rows (full Gauss-Jordan)
            for row in range(self._nrows):
                if row != current_row and self._mat[row, col] == 1.0:
                    self._xor_rows(target=row, source=current_row)

            pivot_cols.append(col)
            current_row += 1

        free_cols = [c for c in range(self._n_vars) if c not in pivot_cols]
        return pivot_cols, free_cols
