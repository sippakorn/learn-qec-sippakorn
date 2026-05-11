"""Gaussian elimination as a recording event generator.

Each row operation is emitted to the Recorder *before* the matrix is mutated,
so the command log faithfully describes what happened in order.

Partial pivoting (largest absolute value in the active column) is used for
numerical stability on real-valued matrices.
"""

import time

import numpy as np
import scipy.sparse as sp

from events import Event
from recorder import Recorder, CHECKPOINT_INTERVAL


class GaussianEliminationGenerator:
    def __init__(self, recorder: Recorder, matrix: sp.spmatrix):
        self._recorder = recorder
        # Work on a dense float64 copy; the initial sparse matrix is already
        # stored as the session baseline by the recorder.
        self._mat = matrix.toarray().astype(np.float64)
        self._nrows, self._ncols = self._mat.shape
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

    def _scale_row(self, row: int, scalar: float) -> None:
        self._recorder.record(Event(
            event_id=-1,
            event_type="scale_row",
            params={"row": row, "scalar": float(scalar)},
            timestamp=time.time(),
            step=self._step,
        ))
        self._mat[row] *= scalar
        self._step += 1
        self._maybe_checkpoint()

    def _add_scaled_row(self, target: int, source: int, scalar: float) -> None:
        self._recorder.record(Event(
            event_id=-1,
            event_type="add_scaled_row",
            params={"target": target, "source": source, "scalar": float(scalar)},
            timestamp=time.time(),
            step=self._step,
        ))
        self._mat[target] += scalar * self._mat[source]
        self._step += 1
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        if self._step % CHECKPOINT_INTERVAL == 0:
            sparse_snapshot = sp.csr_matrix(self._mat)
            self._recorder.checkpoint(sparse_snapshot, self._step)

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        """Run reduced-row-echelon Gaussian elimination with partial pivoting.

        Returns the (dense) result matrix after all operations.
        """
        pivot_row = 0
        for col in range(self._ncols):
            if pivot_row >= self._nrows:
                break

            # Partial pivot: find row with largest |value| in this column
            sub = self._mat[pivot_row:, col]
            local_max = int(np.argmax(np.abs(sub)))
            max_row = local_max + pivot_row

            if self._mat[max_row, col] == 0.0:
                continue  # whole column is zero below pivot_row — skip

            if max_row != pivot_row:
                self._swap_rows(pivot_row, max_row)

            # Normalise pivot row so the pivot entry becomes 1
            pivot_val = self._mat[pivot_row, col]
            if pivot_val != 1.0:
                self._scale_row(pivot_row, 1.0 / pivot_val)

            # Eliminate all other rows in this column (full RREF)
            for row in range(self._nrows):
                if row == pivot_row:
                    continue
                factor = self._mat[row, col]
                if factor != 0.0:
                    self._add_scaled_row(row, pivot_row, -factor)

            pivot_row += 1

        return self._mat
