import numpy as np
from gaussian_elimination import make_augmented_matrix, print_matrix, xor_rows
from sparse_gaussian_elimination import make_sparse_matrix, print_sparse_matrix, xor_rows_sparse, read_solution_sparse
from sparse_gaussian_elimination_v2 import make_col_to_rows, xor_rows_sparse_v2

def erasure_decode_sparse_v3(H, s, erasure_index_set):
    """
    Sparse ML erasure decoder — all three changes applied.
    Fixed: col_to_rows swap update now uses pre-swap row contents.
    """
    n_vars = H.shape[1]

    # Build sparse rows restricted to erased columns only
    rows = [
        set(j for j in np.where(H[i] == 1)[0] if j in erasure_index_set)
        for i in range(H.shape[0])
    ]
    rhs = list(s)

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

        # ── Row swap — capture contents BEFORE swap ──────────────────────
        if pivot_row != current_row:
            # Save pre-swap contents
            old_current = set(rows[current_row])
            old_pivot   = set(rows[pivot_row])

            # Swap rows and rhs
            rows[current_row], rows[pivot_row] = rows[pivot_row], rows[current_row]
            rhs[current_row],  rhs[pivot_row]  = rhs[pivot_row],  rhs[current_row]

            # Update col_to_rows using PRE-SWAP contents
            # old_pivot is now at current_row → update entries
            for j in old_pivot:
                col_to_rows[j].discard(pivot_row)
                col_to_rows[j].add(current_row)
            # old_current is now at pivot_row → update entries
            for j in old_current:
                col_to_rows[j].discard(current_row)
                col_to_rows[j].add(pivot_row)

        # Eliminate col from all other rows
        rows_to_eliminate = set(col_to_rows[col]) - {current_row}
        for row in rows_to_eliminate:
            xor_rows_sparse_v2(rows, rhs, col_to_rows,
                               target_row=row, pivot_row=current_row)

        pivot_cols.append(col)
        current_row += 1

    free_cols = [c for c in sorted_erasure if c not in pivot_cols]
    solution, is_consistent = read_solution_sparse(rows, rhs, n_vars, pivot_cols)
    return solution, is_consistent, free_cols