import numpy as np

# Try to import M4RI backend — falls back to numpy uint8 GE if unavailable
try:
    from ge_m4ri import ge_f2_m4ri as _ge_backend
    _BACKEND = "m4ri"
except ImportError:
    _ge_backend = None
    _BACKEND = "numpy"

from gaussian_elimination import make_augmented_matrix, print_matrix, xor_rows
from sparse_gaussian_elimination import make_sparse_matrix, print_sparse_matrix, xor_rows_sparse, read_solution_sparse
from sparse_gaussian_elimination_v2 import make_col_to_rows, xor_rows_sparse_v2
from utility import peeling_decoder


def ge_f2_numpy_uint8(H, s):
    """
    GE over F2 using numpy uint8 — vectorised row elimination.

    Per pivot step, eliminates all target rows simultaneously:
        mask      = Aug[:, col].astype(bool)   — find rows with 1 in col
        mask[cur] = False                       — exclude pivot row
        Aug[mask] ^= Aug[cur]                  — batch XOR, C-level

    Cost breakdown:
        Pivot scan : O(m) per pivot — C-level argmax
        Mask build : O(m) per pivot — C-level boolean cast
        Batch XOR  : O(n × n_elim) per pivot — vectorised, dominates

    Best suited for dense submatrices (residual stopping sets after peeling).
    For large sparse residuals, cluster decomposition is more effective
    since it decomposes the stopping set into small independent subproblems
    where both memory access and GE cost are minimised.

    Inputs:
        H: numpy 2D array, dtype=int or uint8, shape (m, n)
        s: numpy 1D array, dtype=int, shape (m,)

    Returns:
        solution:      numpy 1D array, dtype=int, shape (n,) or None
        is_consistent: bool
        free_cols:     list of int
    """
    m, n = H.shape

    Aug = np.zeros((m, n + 1), dtype=np.uint8)
    Aug[:, :n] = H.astype(np.uint8)
    Aug[:, n]  = s.astype(np.uint8)

    pivot_cols  = []
    current_row = 0

    for col in range(n):

        # Pivot search — first row >= current_row with 1 in col
        col_vals    = Aug[current_row:, col]
        pivot_local = np.argmax(col_vals)
        if col_vals[pivot_local] == 0:
            continue
        pivot_row = pivot_local + current_row

        # Swap pivot row into current position
        if pivot_row != current_row:
            Aug[[current_row, pivot_row]] = Aug[[pivot_row, current_row]]

        # Eliminate col from ALL other rows — one vectorised call
        mask              = Aug[:, col].astype(bool)
        mask[current_row] = False
        Aug[mask]        ^= Aug[current_row]

        pivot_cols.append(col)
        current_row += 1
        if current_row == m:
            break

    # Read solution
    free_cols = [c for c in range(n) if c not in pivot_cols]
    s_rref    = Aug[:, n]

    for row in range(m):
        if not np.any(Aug[row, :n]) and s_rref[row]:
            return None, False, free_cols

    solution = np.zeros(n, dtype=int)
    for i, col in enumerate(pivot_cols):
        solution[col] = int(s_rref[i])

    return solution, True, free_cols

def erasure_decode_sparse_v3(H, s, erasure_index_set):
    """
    Sparse ML erasure decoder — all three changes applied.

    Dispatches to ge_f2_numpy_uint8 when H is a dense submatrix
    (all columns are in erasure_index_set), otherwise falls back to
    the original sparse set-based GE for compatibility with the full
    parity-check matrix use case.

    The numpy uint8 path is 10-50x faster on dense residual submatrices
    produced by extract_residual_submatrix() in the scaling experiment.
    """
    n_vars = H.shape[1]

    # ── Fast path: M4RI or numpy uint8 GE on dense submatrix ───────────
    # Detected when erasure_index_set covers all columns of H.
    # This is always true for submatrices from extract_residual_submatrix.
    if len(erasure_index_set) == n_vars and erasure_index_set == set(range(n_vars)):
        backend = _ge_backend if _ge_backend is not None else ge_f2_numpy_uint8
        sol, ok, free_local = backend(H, s)
        if sol is None:
            return None, False, list(erasure_index_set)
        solution  = np.zeros(n_vars, dtype=int)
        col_list  = sorted(erasure_index_set)
        for j in range(n_vars):
            solution[j] = sol[j]
        free_cols = [col_list[j] for j in free_local]
        return solution, ok, free_cols

    # ── Slow path: original sparse set-based GE ───────────────────────────
    # Used when H is the full parity-check matrix and only some columns
    # are in the erasure set (original decoder interface).

    # Build sparse rows restricted to erased columns only
    # Extract all nonzeros once — O(nnz) not O(m x row_access_cost)
    if hasattr(H, "tocsr"):
        from scipy.sparse import csr_matrix
        rows_nz, cols_nz = csr_matrix(H).nonzero()
    else:
        rows_nz, cols_nz = np.where(H == 1)

    row_sets = {}
    for i, j in zip(rows_nz, cols_nz):
        if j not in erasure_index_set:
            continue
        if i not in row_sets:
            row_sets[i] = set()
        row_sets[i].add(j)

    # Keep rows as full-length list — forward_eliminate and read_solution
    # index into rows by row number so length must stay H.shape[0]
    rows = [row_sets.get(i, set()) for i in range(H.shape[0])]
    rhs  = list(s)

    # col_to_rows for erased columns only
    col_to_rows = {j: set() for j in erasure_index_set}
    for i, row_set in enumerate(rows):
        for j in row_set:
            col_to_rows[j].add(i)

    sorted_erasure = sorted(erasure_index_set)
    pivot_cols     = []
    current_row    = 0

    for col in sorted_erasure:

        candidates = [r for r in col_to_rows[col] if r >= current_row]
        if not candidates:
            continue

        pivot_row = min(candidates)

        if pivot_row != current_row:
            old_current = set(rows[current_row])
            old_pivot   = set(rows[pivot_row])

            rows[current_row], rows[pivot_row] = rows[pivot_row], rows[current_row]
            rhs[current_row],  rhs[pivot_row]  = rhs[pivot_row],  rhs[current_row]

            for j in old_pivot:
                col_to_rows[j].discard(pivot_row)
                col_to_rows[j].add(current_row)
            for j in old_current:
                col_to_rows[j].discard(current_row)
                col_to_rows[j].add(pivot_row)

        rows_to_eliminate = set(col_to_rows[col]) - {current_row}
        for row in rows_to_eliminate:
            xor_rows_sparse_v2(rows, rhs, col_to_rows,
                               target_row=row, pivot_row=current_row)

        pivot_cols.append(col)
        current_row += 1

    free_cols = [c for c in sorted_erasure if c not in pivot_cols]
    solution, is_consistent = read_solution_sparse(rows, rhs, n_vars, pivot_cols)
    return solution, is_consistent, free_cols


def erasure_decode_peeling(H, s, erasure_index_set):
    """
    Erasure decoder: peeling first, Sparse GE v3 fallback on residual.

    Peeling resolves all variables reachable without fill-in.
    Remaining stopping set is passed to Sparse GE v3.

    Inputs:
        H:                 numpy 2D array, dtype=int, shape (m, n)
        s:                 numpy 1D array, dtype=int, shape (m,)
        erasure_index_set: set of int

    Returns:
        solution:      numpy 1D array, dtype=int, shape (n,)
        is_consistent: bool
        free_cols:     list of int
        used_ge:       bool — True if GE fallback was needed
    """
    # Phase 1 — peeling
    solution, residual_erasure, residual_syndrome = peeling_decoder(
        H, s, erasure_index_set
    )

    # Peeling fully resolved everything
    if not residual_erasure:
        return solution, True, [], False

    # Phase 2 — GE fallback on residual stopping set
    s_residual = np.array(
        [residual_syndrome.get(i, 0) for i in range(H.shape[0])],
        dtype=int
    )

    ge_sol, is_consistent, free_cols = erasure_decode_sparse_v3(
        H, s_residual, residual_erasure
    )

    if is_consistent and ge_sol is not None:
        for j in residual_erasure:
            solution[j] = ge_sol[j]

    return solution, is_consistent, free_cols, True


def _get_row_nz(H, i):
    """
    Return nonzero column indices of row i.
    Works on both dense numpy arrays and scipy sparse matrices.
    """
    row = H[i]
    if hasattr(row, "toarray"):
        return row.toarray().ravel().nonzero()[0]
    return np.where(row == 1)[0]