import numpy as np
from gaussian_elimination import make_augmented_matrix, print_matrix, xor_rows
from sparse_gaussian_elimination import make_sparse_matrix, print_sparse_matrix, xor_rows_sparse


def main():
    # Reproduce Step 3 example
    H = np.array([[1, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 1, 1]], dtype=int)
    s = np.array([1, 1, 0], dtype=int)

    rows, rhs, n_vars = make_sparse_matrix(H, s)

    print("Sparse representation:")
    print("rows:", rows)
    print("rhs :", rhs)
    print()

    print("Pretty printed:")
    print_sparse_matrix(rows, rhs, n_vars)

    # Test xor_rows_sparse — r3 <- r3 XOR r1
    print("After r3 <- r3 XOR r1:")
    xor_rows_sparse(rows, rhs, target_row=2, pivot_row=0)
    print_sparse_matrix(rows, rhs, n_vars)


if __name__ == "__main__":
    main()