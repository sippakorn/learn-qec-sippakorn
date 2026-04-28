import numpy as np
from gaussian_elimination import make_augmented_matrix, print_matrix, xor_rows, forward_eliminate


def main():
    # ── Test 1: Step 1 example (dense, has free variable) ──
    print("=== Test 1: Step 1 dense example ===")
    H = np.array([[1,1,0,1],
                [0,1,1,1],
                [1,0,1,0],
                [1,1,1,0]], dtype=int)
    s = np.array([1,0,1,1], dtype=int)

    H_aug = make_augmented_matrix(H, s)
    print("Before:")
    print_matrix(H_aug, n_vars=4)

    pivot_cols = forward_eliminate(H_aug, n_vars=4)
    print("After elimination:")
    print_matrix(H_aug, n_vars=4)
    print("Pivot columns:", pivot_cols)
    print()

    # ── Test 2: Step 3 example (sparse, one cycle) ──
    print("=== Test 2: Step 3 sparse example ===")
    H = np.array([[1,1,0,0],
                [0,1,1,0],
                [1,0,1,1]], dtype=int)
    s = np.array([1,1,0], dtype=int)

    H_aug = make_augmented_matrix(H, s)
    print("Before:")
    print_matrix(H_aug, n_vars=4)

    pivot_cols = forward_eliminate(H_aug, n_vars=4)
    print("After elimination:")
    print_matrix(H_aug, n_vars=4)
    print("Pivot columns:", pivot_cols)


if __name__ == "__main__":
    main()