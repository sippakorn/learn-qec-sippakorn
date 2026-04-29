import numpy as np
from gaussian_elimination import make_augmented_matrix, print_matrix, xor_rows
from sparse_gaussian_elimination import make_sparse_matrix, print_sparse_matrix, xor_rows_sparse, read_solution_sparse

def make_col_to_rows(rows, n_vars):
    """
    Build column adjacency dictionary from sparse row sets.

    col_to_rows[j] = set of row indices i where column j
                     is nonzero in row i.

    Inputs:
        rows:   list of sets
        n_vars: int

    Returns:
        col_to_rows: dict mapping int -> set of int
    """
    col_to_rows = {j: set() for j in range(n_vars)}
    for i, row_set in enumerate(rows):
        for j in row_set:
            col_to_rows[j].add(i)
    return col_to_rows


def xor_rows_sparse_v2(rows, rhs, col_to_rows, target_row, pivot_row):
    """
    XOR pivot_row into target_row, keeping col_to_rows consistent.

    Updates both the row sets and the column adjacency dict in-place.
    Cost: O(w) where w = weight of pivot row.

    Inputs:
        rows:       list of sets (modified in-place)
        rhs:        list of ints (modified in-place)
        col_to_rows: dict (modified in-place)
        target_row: int
        pivot_row:  int
    """
    # Columns that will change in target_row
    # = symmetric difference of the two rows
    changing_cols = rows[pivot_row] ^ rows[target_row]

    for j in changing_cols:
        if target_row in col_to_rows[j]:
            # j was in target — it disappears after XOR
            col_to_rows[j].discard(target_row)
        else:
            # j was not in target — it appears after XOR
            col_to_rows[j].add(target_row)

    # Update the row set and rhs
    rows[target_row] = rows[target_row] ^ rows[pivot_row]
    rhs[target_row]  = (rhs[target_row] + rhs[pivot_row]) % 2


def forward_eliminate_sparse_v2(rows, rhs, col_to_rows, n_vars):
    """
    Forward elimination (Gauss-Jordan) over F2.
    Uses col_to_rows for O(1) pivot search and O(w) elimination.

    Inputs:
        rows:        list of sets (modified in-place)
        rhs:         list of ints (modified in-place)
        col_to_rows: dict (modified in-place)
        n_vars:      int

    Returns:
        pivot_cols: list of pivot column indices in order found
    """
    num_rows    = len(rows)
    pivot_cols  = []
    current_row = 0

    for col in range(n_vars):

        # Step 1 — find pivot row using col_to_rows
        # Only rows that actually have a 1 in col are candidates
        candidates = [r for r in col_to_rows[col] if r >= current_row]

        if not candidates:
            # no pivot in this column — free variable
            continue

        pivot_row = min(candidates)   # deterministic choice

        # Step 2 — swap pivot row into current position
        if pivot_row != current_row:
            # Swap row sets
            rows[current_row], rows[pivot_row] = rows[pivot_row], rows[current_row]
            rhs[current_row],  rhs[pivot_row]  = rhs[pivot_row],  rhs[current_row]

            # Update col_to_rows to reflect the swap
            for j in rows[current_row]:
                col_to_rows[j].discard(pivot_row)
                col_to_rows[j].add(current_row)
            for j in rows[pivot_row]:
                col_to_rows[j].discard(current_row)
                col_to_rows[j].add(pivot_row)

        # Step 3 — eliminate col from ALL other rows
        # col_to_rows[col] tells us exactly which rows need elimination
        rows_to_eliminate = col_to_rows[col] - {current_row}
        for row in rows_to_eliminate:
            xor_rows_sparse_v2(rows, rhs, col_to_rows,
                               target_row=row, pivot_row=current_row)

        pivot_cols.append(col)
        current_row += 1

    return pivot_cols


def erasure_decode_sparse_v2(H, s, erasure_index_set):
    """
    Sparse ML erasure decoder with column adjacency dictionary.
    Interface identical to previous versions.

    Inputs:
        H:                 numpy 2D array, dtype=int, shape (m, n)
        s:                 numpy 1D array, dtype=int, shape (m,)
        erasure_index_set: set of int

    Returns:
        solution:      numpy 1D array or None
        is_consistent: bool
        free_cols:     list of int
    """
    n_vars = H.shape[1]

    # Build sparse rows restricted to erased columns
    rows = [
        set(j for j in np.where(H[i] == 1)[0] if j in erasure_index_set)
        for i in range(H.shape[0])
    ]
    rhs = list(s)

    # Build column adjacency dictionary
    col_to_rows = make_col_to_rows(rows, n_vars)

    pivot_cols = forward_eliminate_sparse_v2(rows, rhs, col_to_rows, n_vars)
    free_cols  = [c for c in erasure_index_set if c not in pivot_cols]

    solution, is_consistent = read_solution_sparse(rows, rhs, n_vars, pivot_cols)

    return solution, is_consistent, free_cols