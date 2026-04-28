import numpy as np
from gaussian_elimination import make_augmented_matrix, read_solution, forward_eliminate, gaussian_elimination_f2


def main():
    # ── Test 1: Step 1 dense example ──
    print("=== Test 1: Step 1 dense example ===")
    H = np.array([[1,1,0,1],
                [0,1,1,1],
                [1,0,1,0],
                [1,1,1,0]], dtype=int)
    s = np.array([1,0,1,1], dtype=int)

    sol, ok, pivots, free = gaussian_elimination_f2(H, s)
    print("Consistent:", ok)
    print("Pivot cols:", pivots)
    print("Free  cols:", free)
    print("Solution (free vars=0):", sol)
    print()

    # ── Test 2: Step 3 sparse example ──
    print("=== Test 2: Step 3 sparse example ===")
    H = np.array([[1,1,0,0],
                [0,1,1,0],
                [1,0,1,1]], dtype=int)
    s = np.array([1,1,0], dtype=int)

    sol, ok, pivots, free = gaussian_elimination_f2(H, s)
    print("Consistent:", ok)
    print("Pivot cols:", pivots)
    print("Free  cols:", free)
    print("Solution (free vars=0):", sol)
    print()

    # ── Test 3: Inconsistent system ──
    print("=== Test 3: Inconsistent system ===")
    H = np.array([[1,1,0],
                [1,1,0],
                [0,0,1]], dtype=int)
    s = np.array([1,0,1], dtype=int)

    sol, ok, pivots, free = gaussian_elimination_f2(H, s)
    print("Consistent:", ok)
    print("Solution:", sol)


if __name__ == "__main__":
    main()