import numpy as np
from gaussian_elimination import make_augmented_matrix, print_matrix, xor_rows
from sparse_gaussian_elimination import make_sparse_matrix, print_sparse_matrix, xor_rows_sparse, read_solution_sparse
from sparse_gaussian_elimination_v2 import make_col_to_rows, xor_rows_sparse_v2
from utility import peeling_decoder

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
    # Reconstruct syndrome vector for GE from residual_syndrome dict
    s_residual = np.array(
        [residual_syndrome.get(i, 0) for i in range(H.shape[0])],
        dtype=int
    )

    ge_sol, is_consistent, free_cols = erasure_decode_sparse_v3(
        H, s_residual, residual_erasure
    )

    if is_consistent and ge_sol is not None:
        # Merge GE solution into peeling solution
        for j in residual_erasure:
            solution[j] = ge_sol[j]

    return solution, is_consistent, free_cols, True