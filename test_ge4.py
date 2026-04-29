import numpy as np
from gaussian_elimination import make_augmented_matrix, read_solution, forward_eliminate, gaussian_elimination_f2,erasure_decode_f2


def main():
    # [7,4,3] Hamming code parity-check matrix
    H_hamming = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)

    # ── Case 1: Small erasure {0,1,2}, codeword x=(1,0,1,0,0,0,0) ──
    print("=== Case 1: Erasure {0,1,2} — expect free variable ===")
    s1 = np.array([0, 1, 0], dtype=int)
    sol, ok, free = erasure_decode_f2(H_hamming, s1, erasure_index_set={0,1,2})
    print("Consistent :", ok)
    print("Free cols  :", free)
    print("Solution   :", sol)
    print("(t=0 gives 010, t=1 gives 101 on erased positions)")
    print()

    # ── Case 2: Larger erasure {0,1,2,3} ──
    print("=== Case 2: Erasure {0,1,2,3} — expect unique solution ===")
    # codeword x=(1,1,0,1,0,0,0), syndrome:
    # c1: 1+0+0+0 = 1
    # c2: 1+0+0+0 = 1
    # c3: 1+0+0+0 = 1
    s2 = np.array([1, 1, 1], dtype=int)
    sol, ok, free = erasure_decode_f2(H_hamming, s2, erasure_index_set={0,1,2,3})
    print("Consistent :", ok)
    print("Free cols  :", free)
    print("Solution   :", sol)
    print()

    # ── Case 3: Stopping set erasure {0,1,3,5} ──
    print("=== Case 3: Erasure {0,1,3,5} — stopping set, free variables ===")
    # zero syndrome for simplicity (all-zero codeword)
    s3 = np.array([0, 0, 0], dtype=int)
    sol, ok, free = erasure_decode_f2(H_hamming, s3, erasure_index_set={0,1,3,5})
    print("Consistent :", ok)
    print("Free cols  :", free)
    print("Solution   :", sol)
    print()

    # ── Case 4: Inconsistent syndrome ──
    print("=== Case 4: Inconsistent syndrome — expect failure ===")
    # Erase {0,1,2} but give unreachable syndrome
    s4 = np.array([1, 1, 1], dtype=int)
    sol, ok, free = erasure_decode_f2(H_hamming, s4, erasure_index_set={0,1,2})
    print("Consistent :", ok)
    print("Solution   :", sol)


if __name__ == "__main__":
    main()