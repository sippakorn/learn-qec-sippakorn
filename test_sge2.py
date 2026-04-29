import numpy as np
from gaussian_elimination import make_augmented_matrix, print_matrix, xor_rows
from sparse_gaussian_elimination import make_sparse_matrix, print_sparse_matrix, xor_rows_sparse, erasure_decode_sparse


def main():
    H_hamming = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)

    print("=== Sparse: Case 1 — erasure {0,1,2} ===")
    s1 = np.array([0, 1, 0], dtype=int)
    sol, ok, free = erasure_decode_sparse(H_hamming, s1, {0,1,2})
    print("Consistent :", ok)
    print("Free cols  :", free)
    print("Solution   :", sol)
    print()

    print("=== Sparse: Case 2 — erasure {0,1,2,3} ===")
    s2 = np.array([1, 1, 1], dtype=int)
    sol, ok, free = erasure_decode_sparse(H_hamming, s2, {0,1,2,3})
    print("Consistent :", ok)
    print("Free cols  :", free)
    print("Solution   :", sol)
    print()

    print("=== Sparse: Case 3 — stopping set {0,1,3,5} ===")
    s3 = np.array([0, 0, 0], dtype=int)
    sol, ok, free = erasure_decode_sparse(H_hamming, s3, {0,1,3,5})
    print("Consistent :", ok)
    print("Free cols  :", free)
    print("Solution   :", sol)
    print()

    print("=== Sparse: Case 4 — inconsistent ===")
    s4 = np.array([1, 1, 1], dtype=int)
    sol, ok, free = erasure_decode_sparse(H_hamming, s4, {0,1,2})
    print("Consistent :", ok)
    print("Solution   :", sol)


if __name__ == "__main__":
    main()