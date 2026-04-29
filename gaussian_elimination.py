import numpy as np

def make_augmented_matrix(H, s):
    """
    Build augmented matrix [H | s] as a numpy int array.
    
    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)
        s: numpy 1D array, dtype=int, shape (m,)
    
    Output:
        H_aug: numpy 2D array, dtype=int, shape (m, n+1)
    
    Example:
        H = np.array([[1,1,0,0],
                      [0,1,1,0],
                      [1,0,1,1]], dtype=int)
        s = np.array([1,1,0], dtype=int)
        
        Result:
        [[1 1 0 0 | 1]
         [0 1 1 0 | 1]
         [1 0 1 1 | 0]]
    """
    return np.hstack((H, s[:, np.newaxis]))

def print_matrix(H_aug, n_vars):
    """
    Pretty print augmented matrix [H | s].
    
    Inputs:
        H_aug: numpy 2D array
        n_vars: number of variable columns (excludes RHS)
    """
    for row in H_aug:
        left  = row[:n_vars]
        right = row[n_vars:]
        left_str  = " ".join(str(x) for x in left)
        right_str = " ".join(str(x) for x in right)
        print(f"[ {left_str} | {right_str} ]")
    print()

def xor_rows(M, target_row, pivot_row):
    """
    XOR pivot_row into target_row of matrix M in-place over F2.
    
    Inputs:
        M:          numpy 2D array (modified in-place)
        target_row: int, index of row to be updated
        pivot_row:  int, index of row used for elimination
    """
    M[target_row] = (M[target_row] + M[pivot_row]) % 2

def forward_eliminate(H_aug, n_vars):
    """
    Forward elimination (Gauss-Jordan) over F2.
    Modifies H_aug in-place to reduced row echelon form.
    
    Inputs:
        H_aug:  numpy 2D array, dtype=int, shape (m, n+1)
        n_vars: number of variable columns (excludes RHS)
    
    Returns:
        pivot_cols: list of pivot column indices in order found
    """
    num_rows = H_aug.shape[0]
    pivot_cols = []
    current_row = 0  # next row to assign a pivot

    for col in range(n_vars):

        # Step 1 — find a pivot row for this column
        # Search from current_row downward for a row with 1 in this column
        pivot_row = None
        for row in range(current_row, num_rows):
            if H_aug[row, col] == 1:
                pivot_row = row
                break

        # No pivot found in this column — free variable, skip
        if pivot_row is None:
            continue

        # Step 2 — swap pivot row into current position
        if pivot_row != current_row:
            H_aug[[current_row, pivot_row]] = H_aug[[pivot_row, current_row]]

        # Step 3 — eliminate this column from ALL other rows (Gauss-Jordan)
        for row in range(num_rows):
            if row != current_row and H_aug[row, col] == 1:
                xor_rows(H_aug, target_row=row, pivot_row=current_row)

        # Record this pivot column and advance
        pivot_cols.append(col)
        current_row += 1

    return pivot_cols


def read_solution(H_aug, n_vars, pivot_cols):
    """
    Read solution from reduced row echelon form.
    Free variables are set to 0 (minimum weight choice).
    Matches Connolly's convention exactly.
    
    Inputs:
        H_aug:      numpy 2D array in RREF, shape (m, n+1)
        n_vars:     number of variable columns
        pivot_cols: list of pivot column indices from forward_eliminate
    
    Returns:
        solution:   numpy 1D array, dtype=int, shape (n_vars,)
                    the predicted error vector
        is_consistent: bool
                    False if system has no solution (zero row with RHS=1)
    """
    num_rows = H_aug.shape[0]
    s_rref   = H_aug[:, n_vars]  # RHS column after elimination

    # Check consistency — a zero row with RHS=1 means no solution exists
    for row in range(num_rows):
        row_is_zero = not np.any(H_aug[row, :n_vars])
        if row_is_zero and s_rref[row] == 1:
            return None, False

    # Build solution — pivot variables read from RHS, free variables = 0
    solution = np.zeros(n_vars, dtype=int)
    for i, col in enumerate(pivot_cols):
        solution[col] = s_rref[i]

    return solution, True

def gaussian_elimination_f2(H, s):
    """
    Full GE pipeline over F₂.
    Note: free_cols here includes ALL non-pivot columns including zeros.
    Use erasure_decode_f2 for erasure decoding to get correct free_cols.
    """
    n_vars     = H.shape[1]
    H_aug      = make_augmented_matrix(H, s)
    pivot_cols = forward_eliminate(H_aug, n_vars)
    free_cols  = [c for c in range(n_vars) if c not in pivot_cols]
    solution, is_consistent = read_solution(H_aug, n_vars, pivot_cols)
    return solution, is_consistent, pivot_cols, free_cols


def erasure_decode_f2(H, s, erasure_index_set):
    """
    Maximum-likelihood erasure decoder for a classical linear code.
    Fixed: free_cols now restricted to erasure_index_set only.
    """
    n_vars   = H.shape[1]

    # Zero out columns for non-erased bits
    H_active = H.copy()
    for bit_index in range(n_vars):
        if bit_index not in erasure_index_set:
            H_active[:, bit_index] = 0

    solution, is_consistent, pivot_cols, _ = gaussian_elimination_f2(
        H_active, s
    )

    # Restrict free cols to erasure set — non-erased columns are
    # structurally zero and never genuinely free
    free_cols = [c for c in sorted(erasure_index_set)
                 if c not in pivot_cols]

    return solution, is_consistent, free_cols