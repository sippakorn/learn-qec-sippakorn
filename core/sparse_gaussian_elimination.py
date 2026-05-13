import numpy as np

def make_sparse_matrix(H, s):
    """
    Convert dense numpy H and syndrome s into sparse row representation.
    Each row stored as a set of nonzero column indices.
    RHS syndrome stored separately as a list.

    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)
        s: numpy 1D array, dtype=int, shape (m,)

    Returns:
        rows:   list of sets, rows[i] = set of column indices where H[i] == 1
        rhs:    list of ints, rhs[i] = s[i]
        n_vars: int, number of variable columns
    """
    n_vars = H.shape[1]
    rows   = [set(np.where(H[i] == 1)[0]) for i in range(H.shape[0])]
    rhs    = list(s)
    return rows, rhs, n_vars


def print_sparse_matrix(rows, rhs, n_vars):
    """
    Pretty print sparse matrix in dense form for debugging.
    Reconstructs the full row from the set of nonzero indices.

    Inputs:
        rows:   list of sets
        rhs:    list of ints
        n_vars: int
    """
    for i, row_set in enumerate(rows):
        dense = [1 if j in row_set else 0 for j in range(n_vars)]
        left_str  = " ".join(str(x) for x in dense)
        print(f"[ {left_str} | {rhs[i]} ]")
    print()


def xor_rows_sparse(rows, rhs, target_row, pivot_row):
    """
    XOR pivot_row into target_row using symmetric difference over F2.
    Modifies rows and rhs in-place.

    Cost: O(w) where w = row weight, NOT O(n).

    Inputs:
        rows:       list of sets (modified in-place)
        rhs:        list of ints (modified in-place)
        target_row: int
        pivot_row:  int
    """
    # Symmetric difference = XOR over F2
    rows[target_row] = rows[target_row] ^ rows[pivot_row]
    rhs[target_row]  = (rhs[target_row] + rhs[pivot_row]) % 2


def forward_eliminate_sparse(rows, rhs, n_vars):
    """
    Forward elimination (Gauss-Jordan) over F2 using sparse row sets.
    Modifies rows and rhs in-place to reduced row echelon form.

    Cost per elimination step: O(w) not O(n).

    Inputs:
        rows:   list of sets (modified in-place)
        rhs:    list of ints (modified in-place)
        n_vars: int, number of variable columns

    Returns:
        pivot_cols: list of pivot column indices in order found
    """
    num_rows    = len(rows)
    pivot_cols  = []
    current_row = 0

    for col in range(n_vars):

        # Step 1 — find a pivot row for this column
        # scan from current_row downward for a row containing col
        pivot_row = None
        for row in range(current_row, num_rows):
            if col in rows[row]:          # set lookup: O(1) not O(n)
                pivot_row = row
                break

        # no pivot found — free variable, skip
        if pivot_row is None:
            continue

        # Step 2 — swap pivot row into current position
        if pivot_row != current_row:
            rows[current_row], rows[pivot_row] = rows[pivot_row], rows[current_row]
            rhs[current_row],  rhs[pivot_row]  = rhs[pivot_row],  rhs[current_row]

        # Step 3 — eliminate this column from ALL other rows
        for row in range(num_rows):
            if row != current_row and col in rows[row]:
                xor_rows_sparse(rows, rhs, target_row=row, pivot_row=current_row)

        pivot_cols.append(col)
        current_row += 1

    return pivot_cols


def read_solution_sparse(rows, rhs, n_vars, pivot_cols):
    """
    Read solution from sparse RREF.
    Free variables set to 0 (minimum weight convention).

    Inputs:
        rows:       list of sets in RREF
        rhs:        list of ints
        n_vars:     int
        pivot_cols: list of pivot column indices

    Returns:
        solution:      numpy 1D array, dtype=int, shape (n_vars,)
                       or None if inconsistent
        is_consistent: bool
    """
    # Check consistency — empty row with rhs=1 means no solution
    for i, row_set in enumerate(rows):
        if len(row_set) == 0 and rhs[i] == 1:
            return None, False

    # Build solution
    solution = np.zeros(n_vars, dtype=int)
    for i, col in enumerate(pivot_cols):
        solution[col] = rhs[i]

    return solution, True


def erasure_decode_sparse(H, s, erasure_index_set):
    """
    Sparse ML erasure decoder for a classical linear code.
    Interface identical to erasure_decode_f2 for easy comparison.

    Inputs:
        H:                 numpy 2D array, dtype=int, shape (m, n)
        s:                 numpy 1D array, dtype=int, shape (m,)
        erasure_index_set: set of int, indices of erased bits

    Returns:
        solution:      numpy 1D array, dtype=int, shape (n,)
                       or None if decoding fails
        is_consistent: bool
        free_cols:     list of int
    """
    n_vars = H.shape[1]

    # Build sparse matrix restricted to erased columns only
    # Non-erased columns are simply never added to the row sets
    rows = [
        set(j for j in np.where(H[i] == 1)[0] if j in erasure_index_set)
        for i in range(H.shape[0])
    ]
    rhs = list(s)

    pivot_cols = forward_eliminate_sparse(rows, rhs, n_vars)
    free_cols  = [c for c in erasure_index_set if c not in pivot_cols]

    solution, is_consistent = read_solution_sparse(rows, rhs, n_vars, pivot_cols)

    return solution, is_consistent, free_cols