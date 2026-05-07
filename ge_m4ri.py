# ge_m4ri.py
#
# ctypes wrapper around libm4ri for GE over F2.
# Uses m4ri_helper.so (compiled from m4ri_helper.c) to expose
# M4RI inline functions that are not exported from the shared library.
#
# Provides:
#   ge_f2_m4ri(H, s) -> (solution, is_consistent, free_cols)
#   Drop-in replacement for ge_f2_numpy_uint8.
#
# Speedup vs numpy uint8 GE (sparse LDPC-like matrices, w=4):
#   N=1076  (366 x 328)   :  0.84x  (overhead dominates at small size)
#   N=2500  (910 x 797)   :  1.64x
#   N=4409  (1700x1580)   :  4.02x
#   N=17636 (6932x6145)   : 11.20x
#
# The speedup grows with matrix size because M4RI's bit-packing
# (64 cols per word) and cache-optimal blocking amortise over more work,
# while numpy's column-scan cache misses compound at large m.
#
# Requirements:
#   libm4ri-dev:    apt-get install -y libm4ri-dev
#   m4ri_helper.so: gcc -O3 -shared -fPIC -o m4ri_helper.so \
#                       m4ri_helper.c -lm4ri
#   Both .so files must be accessible via _LIB_PATHS / _HELPER_PATHS below.

import ctypes
import numpy as np
import os

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load libraries ─────────────────────────────────────────────────────────
def _load(candidates):
    for path in candidates:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise ImportError(f"Library not found. Tried:\n" +
                      "\n".join(f"  {p}" for p in candidates))

_m4ri = _load([
    "/usr/lib/x86_64-linux-gnu/libm4ri.so",
    "/usr/local/lib/libm4ri.so",
    "/usr/lib/libm4ri.so",
])

_helper = _load([
    os.path.join(_DIR, "m4ri_helper.so"),
    "/home/claude/m4ri_helper.so",
    "/usr/local/lib/m4ri_helper.so",
])

# ── Opaque mzd_t pointer ───────────────────────────────────────────────────
class _mzd_t(ctypes.Structure):
    pass
_mzd_ptr = ctypes.POINTER(_mzd_t)

# ── M4RI core (exported from libm4ri.so) ──────────────────────────────────
_m4ri.mzd_init.restype  = _mzd_ptr
_m4ri.mzd_init.argtypes = [ctypes.c_int, ctypes.c_int]

_m4ri.mzd_free.restype  = None
_m4ri.mzd_free.argtypes = [_mzd_ptr]

_m4ri.mzd_echelonize.restype  = ctypes.c_int
_m4ri.mzd_echelonize.argtypes = [_mzd_ptr, ctypes.c_int]

# ── Helper wrappers (inline M4RI functions exposed via m4ri_helper.so) ────
_helper.helper_write_row.restype  = None
_helper.helper_write_row.argtypes = [
    _mzd_ptr, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
]

_helper.helper_read_row.restype  = None
_helper.helper_read_row.argtypes = [
    _mzd_ptr, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
]


# ── Matrix I/O ─────────────────────────────────────────────────────────────
def _numpy_to_mzd(H_u8):
    """
    Convert numpy uint8 2D array to mzd_t* using one C call per row.
    Much faster than one mzd_write_bit call per nonzero element.

    Inputs:
        H_u8: numpy 2D array, dtype=uint8, shape (m, n), C-contiguous

    Returns:
        ptr: ctypes POINTER(_mzd_t) — caller must call mzd_free
    """
    m, n   = H_u8.shape
    ptr    = _m4ri.mzd_init(m, n)
    c_buf  = (ctypes.c_uint8 * n)()
    for i in range(m):
        row = H_u8[i]
        if row.any():
            ctypes.memmove(c_buf, row.ctypes.data, n)
            _helper.helper_write_row(ptr, i, c_buf, n)
    return ptr


def _mzd_to_numpy(ptr, m, n):
    """
    Read mzd_t* back into numpy uint8 array using one C call per row.

    Inputs:
        ptr: ctypes POINTER(_mzd_t)
        m:   int, number of rows
        n:   int, number of columns

    Returns:
        out: numpy 2D array, dtype=uint8, shape (m, n)
    """
    out   = np.zeros((m, n), dtype=np.uint8)
    c_buf = (ctypes.c_uint8 * n)()
    for i in range(m):
        _helper.helper_read_row(ptr, i, c_buf, n)
        out[i] = np.frombuffer(c_buf, dtype=np.uint8)
    return out


# ── Main GE function ───────────────────────────────────────────────────────
def ge_f2_m4ri(H, s):
    """
    GE over F2 using M4RI (Method of Four Russians Improved).

    Drop-in replacement for ge_f2_numpy_uint8 in
    sparse_gaussian_elimination_v3.py.

    Advantages over numpy uint8 GE:
        Bit-packing:   64 cols per 64-bit word → 8x smaller matrix footprint
                       (6932x6146 bytes → 6932x97 words = 5.3 MB vs 42.6 MB)
        Cache-optimal: Method of Four Russians processes k columns
                       simultaneously using precomputed lookup tables,
                       keeping working set in L2/L3 cache
        Complexity:    O(n^3 / log n) vs O(n^3) for standard GE

    Measured speedup on sparse LDPC-like matrices (w=4):
        N=1076  (366 x 328)  :  0.84x  (overhead > savings at small N)
        N=2500  (910 x 797)  :  1.64x
        N=4409  (1700x1580)  :  4.02x
        N=17636 (6932x6145)  : 11.20x

    Inputs:
        H: numpy 2D array, dtype=int or uint8, shape (m, n)
           dense submatrix — all columns assumed in erasure set
        s: numpy 1D array, dtype=int, shape (m,)

    Returns:
        solution:      numpy 1D array, dtype=int, shape (n,) or None
        is_consistent: bool
        free_cols:     list of int — columns with no pivot (free variables)
    """
    m, n = H.shape

    # Build augmented matrix [H | s] as uint8
    Aug = np.zeros((m, n + 1), dtype=np.uint8)
    Aug[:, :n] = H.astype(np.uint8)
    Aug[:, n]  = s.astype(np.uint8)

    # Load into M4RI and run RREF
    ptr = _numpy_to_mzd(Aug)
    _m4ri.mzd_echelonize(ptr, 1)   # full=1 → RREF not just upper triangular

    # Read RREF back — one C call per row
    Aug_rref = _mzd_to_numpy(ptr, m, n + 1)
    _m4ri.mzd_free(ptr)

    # Identify pivot columns — first nonzero in H block of each row
    pivot_cols = []
    for i in range(m):
        row_H = Aug_rref[i, :n]
        if row_H.any():
            pivot_cols.append(int(np.argmax(row_H)))

    free_cols = [c for c in range(n) if c not in set(pivot_cols)]

    # Consistency check — row with zero H block and RHS=1 → no solution
    for i in range(m):
        if not Aug_rref[i, :n].any() and Aug_rref[i, n] == 1:
            return None, False, free_cols

    # Build solution from RREF RHS column
    solution = np.zeros(n, dtype=int)
    for i, col in enumerate(pivot_cols):
        solution[col] = int(Aug_rref[i, n])

    return solution, True, free_cols


M4RI_AVAILABLE = True