"""F₂ Gaussian elimination on the augmented matrix [H | s].

Row operations are plain methods decorated with @record_op so they emit
events when called inside a RecordingSession. Algorithm body itself is
recorder-agnostic — no Event construction, no recorder field.

Events emitted:
  swap_rows  — pivot row swapped into current position
  xor_rows   — source row XORed into target row (F₂ elimination)
"""

import numpy as np
import scipy.sparse as sp

from aspects import record_op


class F2GaussianEliminationGenerator:
    def __init__(self, H: np.ndarray, s: np.ndarray):
        n_vars = H.shape[1]
        self._n_vars = n_vars

        # Build augmented matrix [H | s] as float64 for replayer compatibility
        self._mat = np.hstack((H, s[:, np.newaxis])).astype(np.float64)
        self._nrows = self._mat.shape[0]

    # ------------------------------------------------------------------
    # Row operations — auto-recorded when a RecordingSession is active
    # ------------------------------------------------------------------

    @record_op("swap_rows")
    def _swap_rows(self, row_i: int, row_j: int) -> None:
        self._mat[[row_i, row_j]] = self._mat[[row_j, row_i]]

    @record_op("xor_rows")
    def _xor_rows(self, target: int, source: int) -> None:
        self._mat[target] = (self._mat[target] + self._mat[source]) % 2

    # ------------------------------------------------------------------
    # Algorithm — mirrors forward_eliminate() from gaussian_elimination.py
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[int], list[int]]:
        """Run Gauss-Jordan elimination over F₂ on H_aug.

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
