import time
import numpy as np
from gaussian_elimination import erasure_decode_f2
from sparse_gaussian_elimination import erasure_decode_sparse
from sparse_gaussian_elimination_v2 import erasure_decode_sparse_v2


def benchmark_all(H, s, erasure_index_set, n_trials=500, label=""):
    """
    Compare dense, sparse v1, and sparse v2 on the same inputs.
    """
    print(f"=== {label} ===")

    # Dense
    start = time.perf_counter()
    for _ in range(n_trials):
        erasure_decode_f2(H, s, erasure_index_set)
    dense_ms = (time.perf_counter() - start) / n_trials * 1000

    # Sparse v1
    start = time.perf_counter()
    for _ in range(n_trials):
        erasure_decode_sparse(H, s, erasure_index_set)
    v1_ms = (time.perf_counter() - start) / n_trials * 1000

    # Sparse v2
    start = time.perf_counter()
    for _ in range(n_trials):
        erasure_decode_sparse_v2(H, s, erasure_index_set)
    v2_ms = (time.perf_counter() - start) / n_trials * 1000

    print(f"  Dense     : {dense_ms:.4f} ms")
    print(f"  Sparse v1 : {v1_ms:.4f} ms  ({dense_ms/v1_ms:.2f}x vs dense)")
    print(f"  Sparse v2 : {v2_ms:.4f} ms  ({dense_ms/v2_ms:.2f}x vs dense)")
    print()

np.random.seed(42)

H_hamming = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=int)

# Hamming [7,4,3]
benchmark_all(H_hamming, np.array([0,1,0], dtype=int),
              {0,1,2,3,4,5,6}, n_trials=500,
              label="[7,4,3] Hamming")

# n=100
n, m = 100, 75
H_rand = np.zeros((m, n), dtype=int)
for i in range(m):
    cols = np.random.choice(n, size=4, replace=False)
    H_rand[i, cols] = 1
benchmark_all(H_rand, np.zeros(m, dtype=int),
              set(range(50)), n_trials=500,
              label="Random LDPC n=100")

# n=500
n, m = 500, 375
H_large = np.zeros((m, n), dtype=int)
for i in range(m):
    cols = np.random.choice(n, size=4, replace=False)
    H_large[i, cols] = 1
benchmark_all(H_large, np.zeros(m, dtype=int),
              set(range(200)), n_trials=100,
              label="Random LDPC n=500")