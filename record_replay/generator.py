"""Gaussian elimination on real-valued matrices.

Row operations are plain methods decorated with @record_op so they emit
events when the call site is inside a RecordingSession. Algorithm body
itself is recorder-agnostic — no Event construction, no recorder field.

Partial pivoting (largest absolute value in the active column) is used for
numerical stability on real-valued matrices.
"""

import numpy as np
import scipy.sparse as sp

from aspects import record_op


class GaussianEliminationGenerator:
    def __init__(self, matrix: sp.spmatrix):
        # Work on a dense float64 copy. The initial sparse matrix is the
        # session baseline; the caller stores it via recorder.start_session.
        self._mat = matrix.toarray().astype(np.float64)
        self._nrows, self._ncols = self._mat.shape

    # ------------------------------------------------------------------
    # Row operations — auto-recorded when a RecordingSession is active
    # ------------------------------------------------------------------

    @record_op("swap_rows")
    def _swap_rows(self, row_i: int, row_j: int) -> None:
        self._mat[[row_i, row_j]] = self._mat[[row_j, row_i]]

    @record_op("scale_row")
    def _scale_row(self, row: int, scalar: float) -> None:
        self._mat[row] *= scalar

    @record_op("add_scaled_row")
    def _add_scaled_row(self, target: int, source: int, scalar: float) -> None:
        self._mat[target] += scalar * self._mat[source]

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
