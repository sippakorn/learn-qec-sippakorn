import time
import numpy as np
from gaussian_elimination import erasure_decode_f2
from sparse_gaussian_elimination import erasure_decode_sparse

def benchmark(H, s, erasure_index_set, n_trials=500):
    """
    Compare dense vs sparse decoder on the same inputs.
    Runs n_trials times each and reports average time in milliseconds.
    """
    # Dense
    start = time.perf_counter()
    for _ in range(n_trials):
        erasure_decode_f2(H, s, erasure_index_set)
    dense_ms = (time.perf_counter() - start) / n_trials * 1000

    # Sparse
    start = time.perf_counter()
    for _ in range(n_trials):
        erasure_decode_sparse(H, s, erasure_index_set)
    sparse_ms = (time.perf_counter() - start) / n_trials * 1000

    print(f"  Dense  : {dense_ms:.4f} ms per call")
    print(f"  Sparse : {sparse_ms:.4f} ms per call")
    print(f"  Speedup: {dense_ms/sparse_ms:.2f}x")
    print()


# ── Benchmark 1: Hamming [7,4,3] ──
print("=== Benchmark 1: [7,4,3] Hamming code ===")
H_hamming = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=int)
s1 = np.array([0, 1, 0], dtype=int)
benchmark(H_hamming, s1, {0,1,2,3,4,5,6})

# ── Benchmark 2: Random sparse (3,4)-regular LDPC, n=100 ──
print("=== Benchmark 2: Random (3,4)-regular LDPC, n=100 ===")
np.random.seed(42)
n, m = 100, 75
H_rand = np.zeros((m, n), dtype=int)
for i in range(m):
    cols = np.random.choice(n, size=4, replace=False)
    H_rand[i, cols] = 1
s_rand = np.zeros(m, dtype=int)
erasure_rand = set(range(50))   # erase first 50 bits
benchmark(H_rand, s_rand, erasure_rand)

# ── Benchmark 3: Random sparse (3,4)-regular LDPC, n=500 ──
print("=== Benchmark 3: Random (3,4)-regular LDPC, n=500 ===")
n, m = 500, 375
H_large = np.zeros((m, n), dtype=int)
for i in range(m):
    cols = np.random.choice(n, size=4, replace=False)
    H_large[i, cols] = 1
s_large = np.zeros(m, dtype=int)
erasure_large = set(range(200))  # erase first 200 bits
benchmark(H_large, s_large, erasure_large, n_trials=100)