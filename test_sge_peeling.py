import numpy as np

from utility import peeling_decoder
from sparse_gaussian_elimination_v3 import erasure_decode_peeling

# ── Test 1: Step 2 partial erasure — peeling should fully succeed ──────────
print("=== Test 1: Partial erasure — peeling succeeds ===")
H = np.array([[1,1,0,0],
              [0,1,1,0],
              [0,0,1,1]], dtype=int)

# x4=1 known, absorbed into syndrome
s = np.array([1, 0, 0], dtype=int)

sol, residual, res_syn = peeling_decoder(H, s, erasure_index_set={0,1,2})
print("Solution        :", sol)
print("Residual erasure:", residual)
print("Peeling success :", len(residual) == 0)
print("Expected        : [1 0 0 0]")
print()

# ── Test 2: Full erasure — peeling gets stuck ──────────────────────────────
print("=== Test 2: Full erasure — peeling gets stuck ===")
s2 = np.array([1, 0, 1], dtype=int)

sol, residual, res_syn = peeling_decoder(H, s2, erasure_index_set={0,1,2,3})
print("Solution so far :", sol)
print("Residual erasure:", residual)
print("Peeling success :", len(residual) == 0)
print("Residual syndrome:", res_syn)
print()

# ── Test 3: Peeling + GE fallback ─────────────────────────────────────────
print("=== Test 3: Peeling + GE fallback on full erasure ===")
sol, ok, free, used_ge = erasure_decode_peeling(H, s2, erasure_index_set={0,1,2,3})
print("Solution   :", sol)
print("Consistent :", ok)
print("Free cols  :", free)
print("Used GE    :", used_ge)
print()

# ── Test 4: Hamming [7,4,3] — partial erasure, peeling should succeed ─────
print("=== Test 4: Hamming [7,4,3] — peeling succeeds ===")
H_hamming = np.array([
    [1,0,1,0,1,0,1],
    [0,1,1,0,0,1,1],
    [0,0,0,1,1,1,1]
], dtype=int)
s4 = np.array([0,1,0], dtype=int)

sol, residual, _ = peeling_decoder(H_hamming, s4, erasure_index_set={0,1,2})
print("Solution        :", sol)
print("Residual erasure:", residual)
print("Peeling success :", len(residual) == 0)
print()

# ── Test 5: Hamming [7,4,3] — stopping set, needs GE ─────────────────────
print("=== Test 5: Hamming [7,4,3] — stopping set, needs GE ===")
s5 = np.array([0,0,0], dtype=int)

sol, ok, free, used_ge = erasure_decode_peeling(
    H_hamming, s5, erasure_index_set={0,1,3,5}
)
print("Solution   :", sol)
print("Consistent :", ok)
print("Free cols  :", free)
print("Used GE    :", used_ge)