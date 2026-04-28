import numpy as np
from gaussian_elimination import make_augmented_matrix, print_matrix, xor_rows


def main():
    # Reproduce Step 3 example
    H = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [1, 0, 1, 1]], dtype=int)
    s = np.array([1, 1, 0], dtype=int)

    H_aug = make_augmented_matrix(H, s)

    print("Initial augmented matrix:")
    print_matrix(H_aug, n_vars=4)

    # Test xor_rows — should eliminate column 1 from row 3
    xor_rows(H_aug, target_row=2, pivot_row=0)
    print("After r3 <- r3 XOR r1:")
    print_matrix(H_aug, n_vars=4)


if __name__ == "__main__":
    main()