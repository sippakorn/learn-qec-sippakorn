# reorder_ge_experiment.py
#
# Compares GE reordering strategies on four HGP code families from
# Connolly et al., sweeping erasure rate to find the transition point
# where (if ever) reordering provides net speedup over bare M4RI GE.
#
# Code families:
#   [[625,  25]] — PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt
#   [[1225, 65]] — PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt
#   [[1600, 64]] — PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt
#   [[2025, 81]] — PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt
#
# Strategies benchmarked:
#   none     — M4RI GE on original H (baseline)
#   amd_row  — AMD row reordering  → M4RI
#   col_w    — Column weight order → M4RI
#   dm       — Dulmage-Mendelsohn  → M4RI
#   nd_row   — Nested Dissection row reordering → M4RI
#
# Metrics per (code, erasure_rate, strategy):
#   t_order_ms  — reordering computation time (ms)
#   t_ge_ms     — M4RI GE time on (re)ordered H (ms)
#   t_total_ms  — t_order_ms + t_ge_ms
#   speedup     — t_none_total / t_strategy_total  (overall)
#   ge_speedup  — t_none_ge    / t_strategy_ge     (GE only, excl. ordering cost)
#
# Plot layout: 4 rows (one per code family) × 2 columns
#   Left  — overall speedup vs erasure rate (includes ordering cost)
#   Right — GE-only speedup vs erasure rate (pure M4RI benefit)
#
# Usage
# ─────
#   Run benchmark + plot (default):
#       python reorder_ge_experiment.py
#
#   Benchmark only:
#       python reorder_ge_experiment.py --benchmark
#
#   Plot only (requires stat file):
#       python reorder_ge_experiment.py --plot
#
#   Custom parameters:
#       python reorder_ge_experiment.py --trials 30 --erasure-rates 0.30 0.35 0.40 0.45
#
#   Enable debug output per erasure rate:
#       python reorder_ge_experiment.py --debug
#
#   Custom data / stat directories:
#       python reorder_ge_experiment.py --data-dir ./codes/ --stat-file ./out/stats.msgpack

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time
import os
import argparse
import ctypes
import msgpack
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching, reverse_cuthill_mckee

# ── Try to import M4RI ─────────────────────────────────────────────────────
try:
    from ge_m4ri import ge_f2_m4ri
    _M4RI_AVAILABLE = True
except ImportError:
    _M4RI_AVAILABLE = False
    raise ImportError(
        "ge_m4ri.py not found. Ensure ge_m4ri.py and m4ri_helper.so are "
        "in the same directory and libm4ri is installed."
    )

# ── Experiment parameters ──────────────────────────────────────────────────
CODE_FAMILIES = {
    "n625"  : {
        "file"  : "PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt",
        "label" : "[[625, 25]]",
        "N"     : 625,
    },
    "n1225" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt",
        "label" : "[[1225, 65]]",
        "N"     : 1225,
    },
    "n1600" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt",
        "label" : "[[1600, 64]]",
        "N"     : 1600,
    },
    "n2025" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt",
        "label" : "[[2025, 81]]",
        "N"     : 2025,
    },
}

DEFAULT_ERASURE_RATES = [round(r, 2) for r in np.arange(0.30, 0.51, 0.02)]
DEFAULT_TRIALS        = 200
DEFAULT_SEED          = 42
DEFAULT_STAT_FILE     = "stats_reorder_ge_experiment.msgpack"
DEFAULT_PLOT_FILE     = "reorder_ge_experiment.png"

STRATEGY_META = {
    "none"   : {"label": "none (M4RI baseline)", "color": "#2c3e50",
                "linestyle": "-",  "marker": "o"},
    "amd_row": {"label": "AMD row",              "color": "#e74c3c",
                "linestyle": "--", "marker": "s"},
    "col_w"  : {"label": "Col weight",           "color": "#f39c12",
                "linestyle": "-.", "marker": "^"},
    "dm"     : {"label": "DM decomp",            "color": "#27ae60",
                "linestyle": ":",  "marker": "D"},
    "nd_row" : {"label": "Nested Dissection",    "color": "#9b59b6",
                "linestyle": "--", "marker": "P"},
    "rcm_row": {"label": "RCM row",               "color": "#1abc9c",
                "linestyle": "-.", "marker": "v"},
}

# ── External libraries ─────────────────────────────────────────────────────

# AMD (SuiteSparse)
try:
    _amd = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libamd.so")
    _amd.amd_order.restype  = ctypes.c_int
    _amd.amd_order.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _amd.amd_defaults.restype  = None
    _amd.amd_defaults.argtypes = [ctypes.POINTER(ctypes.c_double)]
    _AMD_AVAILABLE = True
except OSError:
    _AMD_AVAILABLE = False

# METIS (Nested Dissection)
try:
    _metis = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libmetis.so")
    _idx_t = ctypes.c_int32
    _METIS_OK      = 1
    _METIS_NOPTIONS = 40
    _metis.METIS_SetDefaultOptions.restype  = ctypes.c_int
    _metis.METIS_SetDefaultOptions.argtypes = [ctypes.POINTER(_idx_t)]
    _metis.METIS_NodeND.restype  = ctypes.c_int
    _metis.METIS_NodeND.argtypes = [
        ctypes.POINTER(_idx_t), ctypes.POINTER(_idx_t),
        ctypes.POINTER(_idx_t), ctypes.POINTER(_idx_t),
        ctypes.POINTER(_idx_t), ctypes.POINTER(_idx_t),
        ctypes.POINTER(_idx_t),
    ]
    _METIS_AVAILABLE = True
except OSError:
    _METIS_AVAILABLE = False


# ── Reordering strategies ──────────────────────────────────────────────────

def _amd_order_matrix(A_sparse):
    """AMD fill-reducing ordering for symmetric sparse matrix A."""
    n      = A_sparse.shape[0]
    A_csc  = csc_matrix(A_sparse, dtype=np.int32)
    Ap     = A_csc.indptr.astype(np.int32)
    Ai     = A_csc.indices.astype(np.int32)
    P      = np.zeros(n, dtype=np.int32)
    Ctrl   = (ctypes.c_double * 5)()
    Info   = (ctypes.c_double * 20)()
    _amd.amd_defaults(Ctrl)
    _amd.amd_order(
        ctypes.c_int(n),
        Ap.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        Ai.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        P.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        Ctrl, Info,
    )
    return P.astype(int)


def _nd_order_matrix(A_sparse):
    """METIS nested dissection ordering for symmetric sparse matrix A."""
    n      = A_sparse.shape[0]
    A_csr  = csr_matrix(A_sparse, dtype=np.int32)
    xadj   = A_csr.indptr.astype(np.int32)
    adjncy = A_csr.indices.astype(np.int32)
    nvtxs  = _idx_t(n)
    opts   = (_idx_t * _METIS_NOPTIONS)()
    _metis.METIS_SetDefaultOptions(opts)
    perm   = np.zeros(n, dtype=np.int32)
    iperm  = np.zeros(n, dtype=np.int32)
    ret = _metis.METIS_NodeND(
        ctypes.byref(nvtxs),
        xadj.ctypes.data_as(ctypes.POINTER(_idx_t)),
        adjncy.ctypes.data_as(ctypes.POINTER(_idx_t)),
        None, opts,
        perm.ctypes.data_as(ctypes.POINTER(_idx_t)),
        iperm.ctypes.data_as(ctypes.POINTER(_idx_t)),
    )
    assert ret == _METIS_OK, f"METIS_NodeND returned {ret}"
    return perm.astype(int)


def _row_row_adj(H):
    """Sparse row-row adjacency matrix: A[i,j]=1 if rows i,j share a column."""
    H_sp = csr_matrix(H.astype(np.int32))
    A    = (H_sp @ H_sp.T).astype(bool).astype(np.int32)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def apply_amd_row(H):
    """AMD row permutation via row-row adjacency."""
    A = _row_row_adj(H)
    if A.nnz == 0:
        return H, np.arange(H.shape[0])
    order = _amd_order_matrix(A)
    return H[order, :], order


def apply_col_weight(H):
    """Column weight ordering: sparsest columns first."""
    order = np.argsort(H.sum(axis=0), kind='stable')
    return H[:, order], order


def apply_dm(H):
    """
    Dulmage-Mendelsohn via maximum bipartite matching.
    Matched rows/cols first, unmatched (structural free vars) last.
    Returns (H_perm, row_order, col_order, n_matched).
    """
    H_sp      = csr_matrix(H.astype(np.int32))
    col_match = maximum_bipartite_matching(H_sp, perm_type='column')
    matched_rows   = np.where(col_match >= 0)[0].astype(int)
    matched_cols   = col_match[matched_rows].astype(int)
    matched_col_set = set(matched_cols.tolist())
    unmatched_rows = np.where(col_match < 0)[0].astype(int)
    unmatched_cols = np.array(
        [c for c in range(H.shape[1]) if c not in matched_col_set], dtype=int
    )
    row_order = np.concatenate([matched_rows, unmatched_rows])
    col_order = np.concatenate([matched_cols, unmatched_cols])
    return H[np.ix_(row_order, col_order)], row_order, col_order, len(matched_cols)


def apply_nd_row(H):
    """Nested dissection row permutation via METIS NodeND."""
    A = _row_row_adj(H)
    if A.nnz == 0:
        return H, np.arange(H.shape[0])
    order = _nd_order_matrix(A)
    return H[order, :], order


def apply_rcm_row(H):
    """
    Reverse Cuthill-McKee row permutation via row-row adjacency.
    Minimises matrix bandwidth — reduces cache-miss pattern in GE.
    Uses scipy sparse multiply: O(m x w^2) not O(m^2 x n).
    """
    A = _row_row_adj(H)
    if A.nnz == 0:
        return H, np.arange(H.shape[0], dtype=int)
    order = reverse_cuthill_mckee(A, symmetric_mode=True).astype(int)
    return H[order, :], order


# ── HGP utilities ──────────────────────────────────────────────────────────

def load_classical_H(filepath):
    """Load classical H from Connolly et al. txt file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")
    with open(filepath) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    m, n = int(lines[0].split()[0]), int(lines[0].split()[1])
    H    = np.zeros((m, n), dtype=int)
    for row_idx, line in enumerate(lines[1:]):
        for col in map(int, line.split()):
            H[row_idx, col] = 1
    return H


def build_hgp_sparse(H_cl):
    """Build sparse HGP Hx using scipy sparse kron."""
    from scipy.sparse import eye as speye, kron as spkron, hstack as sphstack
    m, n  = H_cl.shape
    H_sp  = csr_matrix(H_cl, dtype=np.int8)
    Im    = speye(m, dtype=np.int8, format="csr")
    In    = speye(n, dtype=np.int8, format="csr")
    Hx    = sphstack([spkron(H_sp, In), spkron(Im, H_sp.T)], format="csr")
    return Hx


def peeling_decoder(H, s, erasure_index_set):
    """Peeling decoder for classical linear code over BEC."""
    n_vars   = H.shape[1]
    solution = np.zeros(n_vars, dtype=int)
    check_to_vars = {}
    var_to_checks = {j: set() for j in erasure_index_set}

    if hasattr(H, "tocsr"):
        rows_nz, cols_nz = H.tocsr().nonzero()
    else:
        rows_nz, cols_nz = np.where(H == 1)

    for i, j in zip(rows_nz, cols_nz):
        if j not in erasure_index_set:
            continue
        if i not in check_to_vars:
            check_to_vars[i] = set()
        check_to_vars[i].add(int(j))
        var_to_checks[j].add(int(i))

    syndrome = {i: int(s[i]) for i in check_to_vars}
    dangling  = {i for i, nbrs in check_to_vars.items() if len(nbrs) == 1}

    while dangling:
        check = dangling.pop()
        if check not in check_to_vars or len(check_to_vars[check]) != 1:
            continue
        var           = next(iter(check_to_vars[check]))
        var_value     = syndrome[check]
        solution[var] = var_value
        for nb in var_to_checks[var]:
            if nb == check or nb not in check_to_vars:
                continue
            syndrome[nb] ^= var_value
            check_to_vars[nb].discard(var)
            if len(check_to_vars[nb]) == 1:
                dangling.add(nb)
            elif len(check_to_vars[nb]) == 0:
                del check_to_vars[nb]
                del syndrome[nb]
        del var_to_checks[var]
        del check_to_vars[check]
        del syndrome[check]

    return solution, set(var_to_checks.keys()), syndrome


def extract_residual_submatrix(H, residual_erasure):
    """
    Extract dense submatrix of H restricted to residual stopping set.
    Returns (H_sub, active_rows, col_map).
    """
    col_map = np.array(sorted(residual_erasure), dtype=int)
    if hasattr(H, "tocsr"):
        H_csr = H.tocsr()
        rows_nz, cols_nz = H_csr.nonzero()
    else:
        rows_nz, cols_nz = np.where(H == 1)

    residual_set = residual_erasure
    active_set   = set()
    for i, j in zip(rows_nz, cols_nz):
        if j in residual_set:
            active_set.add(int(i))

    active_rows = np.array(sorted(active_set), dtype=int)
    n_active    = len(active_rows)
    n_cols      = len(col_map)
    H_sub       = np.zeros((n_active, n_cols), dtype=int)
    col_to_sub  = {int(orig): sub for sub, orig in enumerate(col_map)}

    for sub_row, orig_row in enumerate(active_rows):
        if hasattr(H, "tocsr"):
            _, orig_cols = H_csr[int(orig_row)].nonzero()
        else:
            orig_cols = np.where(H[int(orig_row)] == 1)[0]
        for orig_col in orig_cols:
            oc = int(orig_col)
            if oc in col_to_sub:
                H_sub[sub_row, col_to_sub[oc]] = 1

    return H_sub, active_rows, col_map


# ── Core per-trial measurement ─────────────────────────────────────────────

def measure_trial(H_sub, s_sub):
    """
    Run all strategies on one (H_sub, s_sub) instance.
    Returns dict of {strategy: {"t_order_ms", "t_ge_ms", "t_total_ms"}}.
    All strategies run on identical input — fair comparison.
    """
    n_cols      = H_sub.shape[1]
    sub_erasure = set(range(n_cols))
    results     = {}

    # ── none: M4RI on original H_sub ──────────────────────────────────────
    t0   = time.perf_counter()
    ge_f2_m4ri(H_sub, s_sub)
    t_ge = (time.perf_counter() - t0) * 1000
    results["none"] = {"t_order_ms": 0.0, "t_ge_ms": t_ge,
                       "t_total_ms": t_ge}

    # ── amd_row ───────────────────────────────────────────────────────────
    if _AMD_AVAILABLE:
        t0 = time.perf_counter()
        H_p, _ = apply_amd_row(H_sub)
        t_ord  = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        ge_f2_m4ri(H_p, s_sub)
        t_ge   = (time.perf_counter() - t0) * 1000
        results["amd_row"] = {"t_order_ms": t_ord, "t_ge_ms": t_ge,
                               "t_total_ms": t_ord + t_ge}

    # ── col_w ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    H_p, col_order = apply_col_weight(H_sub)
    t_ord = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    ge_f2_m4ri(H_p, s_sub)
    t_ge = (time.perf_counter() - t0) * 1000
    results["col_w"] = {"t_order_ms": t_ord, "t_ge_ms": t_ge,
                        "t_total_ms": t_ord + t_ge}

    # ── dm ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    H_p, row_order, col_order, n_matched = apply_dm(H_sub)
    t_ord = (time.perf_counter() - t0) * 1000
    H_m   = H_p[:, :n_matched]
    s_m   = s_sub[row_order]
    t0 = time.perf_counter()
    ge_f2_m4ri(H_m, s_m)
    t_ge = (time.perf_counter() - t0) * 1000
    results["dm"] = {"t_order_ms": t_ord, "t_ge_ms": t_ge,
                     "t_total_ms": t_ord + t_ge}

    # ── nd_row ────────────────────────────────────────────────────────────
    if _METIS_AVAILABLE:
        t0 = time.perf_counter()
        H_p, _ = apply_nd_row(H_sub)
        t_ord  = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        ge_f2_m4ri(H_p, s_sub)
        t_ge   = (time.perf_counter() - t0) * 1000
        results["nd_row"] = {"t_order_ms": t_ord, "t_ge_ms": t_ge,
                              "t_total_ms": t_ord + t_ge}

    # ── rcm_row ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    H_p, _ = apply_rcm_row(H_sub)
    t_ord  = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    ge_f2_m4ri(H_p, s_sub)
    t_ge   = (time.perf_counter() - t0) * 1000
    results["rcm_row"] = {"t_order_ms": t_ord, "t_ge_ms": t_ge,
                          "t_total_ms": t_ord + t_ge}

    return results


# ── Benchmark ──────────────────────────────────────────────────────────────

def run_experiment(
    erasure_rates = DEFAULT_ERASURE_RATES,
    n_trials      = DEFAULT_TRIALS,
    random_seed   = DEFAULT_SEED,
    data_dir      = ".",
    stat_file     = DEFAULT_STAT_FILE,
    debug         = False,
):
    """
    Run reordering strategy comparison on all four HGP code families.

    For each (family, erasure_rate):
        1. Sample n_trials erasure patterns
        2. Run peeling → extract residual submatrix H_sub
        3. Run measure_trial(H_sub, s_sub) for each GE-needed trial
        4. Record t_order_ms and t_ge_ms per strategy

    Stat file schema:
    {
      "params": {...},
      "results": {
        "n625": {
          "label": str,
          "N": int,
          "erasure_rates": [...],
          "per_rate": {
            "0.10": {
              "n_trials": int,
              "n_ge_trials": int,
              "residual_sizes": [float,...],
              "strategies": {
                "none":    {"t_order_ms":[...], "t_ge_ms":[...], "t_total_ms":[...]},
                "amd_row": {...},
                ...
              }
            }, ...
          }
        }, ...
      }
    }
    """
    rng = np.random.default_rng(random_seed)

    print("Reorder GE Experiment — Four HGP Code Families")
    print("═" * 50)
    print(f"  erasure rates : {erasure_rates[0]:.2f} → {erasure_rates[-1]:.2f} "
          f"({len(erasure_rates)} points)")
    print(f"  trials        : {n_trials}")
    print(f"  seed          : {random_seed}")
    print(f"  data dir      : {os.path.abspath(data_dir)}")
    print(f"  stat file     : {stat_file}")
    print(f"  debug         : {debug}")
    avail = []
    if _AMD_AVAILABLE:   avail.append("AMD")
    if _METIS_AVAILABLE: avail.append("METIS/ND")
    print(f"  libraries     : M4RI + {', '.join(avail) if avail else 'none'}")
    print()

    all_results = {}

    for code_name, meta in CODE_FAMILIES.items():
        filepath = os.path.join(data_dir, meta["file"])
        label    = meta["label"]

        print(f"── {label} ──────────────────────────")
        try:
            H_cl = load_classical_H(filepath)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}\n")
            continue

        Hx     = build_hgp_sparse(H_cl)
        N      = Hx.shape[1]
        n_rows = Hx.shape[0]
        print(f"  Hx shape : {Hx.shape}   N={N}")

        code_result = {
            "label"        : label,
            "N"            : N,
            "erasure_rates": erasure_rates,
            "per_rate"     : {},
        }

        sx = np.zeros(n_rows, dtype=int)

        for rate in erasure_rates:
            n_erased     = int(N * rate)
            n_ge_trials  = 0
            residual_sizes = []

            # Per-strategy raw lists (GE-needed trials only)
            strat_data = {
                s: {"t_order_ms": [], "t_ge_ms": [], "t_total_ms": []}
                for s in STRATEGY_META
            }

            for trial in range(n_trials):
                erased_bits = rng.choice(N, size=n_erased, replace=False)
                erasure_set = set(erased_bits.tolist())

                _, residual, res_syn = peeling_decoder(Hx, sx, erasure_set)

                if not residual:
                    continue   # peeling succeeded — GE not needed

                s_res = np.array(
                    [res_syn.get(i, 0) for i in range(n_rows)], dtype=int
                )
                H_sub, active_rows, col_map = extract_residual_submatrix(
                    Hx, residual
                )
                s_sub = s_res[active_rows]

                # Run all strategies on this H_sub
                trial_results = measure_trial(H_sub, s_sub)

                n_ge_trials += 1
                residual_sizes.append(len(residual))

                for sname, data in trial_results.items():
                    for key in ("t_order_ms", "t_ge_ms", "t_total_ms"):
                        strat_data[sname][key].append(data[key])

            # Compute means for console output
            def mean_ms(lst):
                return float(np.mean(lst)) if lst else 0.0

            t_none_total = mean_ms(strat_data["none"]["t_total_ms"])
            res_mean     = float(np.mean(residual_sizes)) if residual_sizes else 0.0

            # Console line
            parts = (
                f"rate={rate:.2f}  |ε|={res_mean:.0f}  "
                f"n_ge={n_ge_trials}/{n_trials}  "
                f"t_none={t_none_total:.1f}ms"
            )
            if n_ge_trials > 0:
                for sname in ["amd_row", "col_w", "dm", "nd_row"]:
                    if strat_data[sname]["t_total_ms"]:
                        t_s = mean_ms(strat_data[sname]["t_total_ms"])
                        sp  = t_none_total / t_s if t_s > 0 else 0.0
                        parts += f"  {sname}={sp:.2f}x"
            print(f"  {parts}")

            if debug and n_ge_trials > 0:
                print(f"    [debug] H_sub shape: "
                      f"({int(np.mean([len(residual_sizes)]))}...)")
                for sname in STRATEGY_META:
                    if strat_data[sname]["t_order_ms"]:
                        t_ord = mean_ms(strat_data[sname]["t_order_ms"])
                        t_ge  = mean_ms(strat_data[sname]["t_ge_ms"])
                        t_tot = mean_ms(strat_data[sname]["t_total_ms"])
                        t_none_ge = mean_ms(strat_data["none"]["t_ge_ms"])
                        ge_sp = t_none_ge / t_ge if t_ge > 0 else 0.0
                        print(f"      {sname:<10} "
                              f"order={t_ord:7.2f}ms  "
                              f"ge={t_ge:7.2f}ms  "
                              f"total={t_tot:7.2f}ms  "
                              f"ge_speedup={ge_sp:.2f}x")

            code_result["per_rate"][str(rate)] = {
                "n_trials"      : n_trials,
                "n_ge_trials"   : n_ge_trials,
                "residual_sizes": residual_sizes,
                "strategies"    : {
                    sname: {
                        k: [float(x) for x in v]
                        for k, v in strat_data[sname].items()
                    }
                    for sname in STRATEGY_META
                    if strat_data[sname]["t_total_ms"]
                },
            }

        all_results[code_name] = code_result
        print()

    # Save
    output = {
        "params": {
            "erasure_rates": erasure_rates,
            "n_trials"     : n_trials,
            "random_seed"  : random_seed,
        },
        "results": all_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(stat_file)), exist_ok=True)
    with open(stat_file, "wb") as f:
        msgpack.pack(output, f)
    print(f"Stats saved → {stat_file}")
    return output


# ── Plot ───────────────────────────────────────────────────────────────────

def plot_experiment(
    stat_file = DEFAULT_STAT_FILE,
    plot_file = DEFAULT_PLOT_FILE,
):
    """
    Plot reordering speedup vs erasure rate for all four code families.

    Layout: 4 rows × 2 columns
        Left  — overall speedup (t_none_total / t_strategy_total)
                includes ordering cost — answers "is it worth it in practice?"
        Right — GE-only speedup (t_none_ge / t_strategy_ge)
                excludes ordering cost — answers "does reordering help M4RI?"

    Horizontal reference line at 1.0 in both columns.
    """
    if not os.path.exists(stat_file):
        raise FileNotFoundError(
            f"Stat file not found: '{stat_file}'. Run --benchmark first."
        )
    with open(stat_file, "rb") as f:
        data = msgpack.unpack(f)

    results = data["results"]
    codes   = [k for k in CODE_FAMILIES if k in results]

    if not codes:
        print("No results found in stat file.")
        return

    n_codes = len(codes)
    fig, axes = plt.subplots(
        n_codes, 2,
        figsize=(13, 4.5 * n_codes),
        squeeze=False,
    )
    fig.suptitle(
        "GE Reordering Strategies vs Bare M4RI — Four HGP Code Families\n"
        "Left: overall speedup (includes ordering cost)   |   "
        "Right: GE-only speedup (M4RI on reordered H vs original H)\n"
        "Reference line at 1.0 — above = reordering helps",
        fontsize=10,
    )

    def mean_or_nan(lst):
        return float(np.mean(lst)) if lst else float("nan")

    for row_idx, code_name in enumerate(codes):
        cr    = results[code_name]
        label = cr["label"]
        rates = cr["erasure_rates"]
        pr    = cr["per_rate"]

        ax_left  = axes[row_idx][0]
        ax_right = axes[row_idx][1]

        for ax in (ax_left, ax_right):
            ax.axhline(y=1.0, color="#bdc3c7", linewidth=1,
                       linestyle="-", label="_nolegend_")
            ax.set_xticks(rates)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlabel("Erasure rate", fontsize=9)

        ax_left.set_title(f"{label} — Overall speedup\n"
                          f"(t_none_total / t_strategy_total)", fontsize=9)
        ax_left.set_ylabel("Speedup (×)", fontsize=9)

        ax_right.set_title(f"{label} — GE-only speedup\n"
                           f"(t_none_GE / t_strategy_GE)", fontsize=9)
        ax_right.set_ylabel("GE speedup (×)", fontsize=9)

        # Gather data per strategy
        for sname, smeta in STRATEGY_META.items():
            overall_sp = []
            ge_sp      = []
            valid_rates = []

            for rate in rates:
                rkey = str(rate)
                if rkey not in pr:
                    continue
                rd = pr[rkey]
                if rd["n_ge_trials"] == 0:
                    continue
                strats = rd["strategies"]
                if "none" not in strats or sname not in strats:
                    continue

                t_none_total = mean_or_nan(strats["none"]["t_total_ms"])
                t_none_ge    = mean_or_nan(strats["none"]["t_ge_ms"])
                t_s_total    = mean_or_nan(strats[sname]["t_total_ms"])
                t_s_ge       = mean_or_nan(strats[sname]["t_ge_ms"])

                if t_s_total > 0 and t_s_ge > 0:
                    overall_sp.append(t_none_total / t_s_total)
                    ge_sp.append(t_none_ge / t_s_ge)
                    valid_rates.append(rate)

            if not valid_rates:
                continue

            kw = dict(
                color=smeta["color"],
                linestyle=smeta["linestyle"],
                marker=smeta["marker"],
                linewidth=1.2, markersize=4,
                label=smeta["label"],
            )
            ax_left.plot(valid_rates, overall_sp,  **kw)
            ax_right.plot(valid_rates, ge_sp, **kw)

        for ax in (ax_left, ax_right):
            ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved → {plot_file}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare GE reordering strategies on four HGP code families.\n"
            "Default (no flags): runs benchmark then plots."
        )
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run experiment and save stats."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Load stats and produce plot."
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help=f"Trials per (code, erasure rate) point (default: {DEFAULT_TRIALS})."
    )
    parser.add_argument(
        "--erasure-rates", nargs="+", type=float,
        default=DEFAULT_ERASURE_RATES,
        help="Erasure rates to sweep (default: 0.10 to 0.50 step 0.05)."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})."
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing classical H txt files (default: .)."
    )
    parser.add_argument(
        "--stat-file", default=DEFAULT_STAT_FILE,
        help=f"Output msgpack path (default: {DEFAULT_STAT_FILE})."
    )
    parser.add_argument(
        "--plot-file", default=DEFAULT_PLOT_FILE,
        help=f"Output PNG path (default: {DEFAULT_PLOT_FILE})."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "Print per-strategy timing breakdown "
            "(t_order, t_ge, t_total, ge_speedup) for each erasure rate. "
            "Turn off for production runs."
        )
    )
    args = parser.parse_args()

    if not args.benchmark and not args.plot:
        args.benchmark = True
        args.plot      = True

    if args.benchmark:
        run_experiment(
            erasure_rates = [round(r, 4) for r in args.erasure_rates],
            n_trials      = args.trials,
            random_seed   = args.seed,
            data_dir      = args.data_dir,
            stat_file     = args.stat_file,
            debug         = args.debug,
        )

    if args.plot:
        plot_experiment(
            stat_file = args.stat_file,
            plot_file = args.plot_file,
        )