# peeling_reorder_benchmark.py
#
# Benchmark the full decoding pipeline across three reordering strategies:
#   none — original Hx, no reordering
#   dfs  — DFS post-order reordering of Tanner graph
#   rcm  — Reverse Cuthill-McKee reordering (bandwidth minimisation)
#
# Pipeline per trial:
#   1. reorder applied ONCE per code (timed separately as t_reorder_ms)
#   2. peeling_decoder(H_strategy, ...)          — per trial, all trials
#   3. sparse_ge_v3(H_strategy, ...) if needed   — per trial, residual only
#
# Raw stats saved per code family as msgpack binary files.
#
# Plot: 2x2 grid (one subplot per code family)
#   Left Y-axis  — avg time per call (ms): peeling + GE for all strategies
#   Right Y-axis — peeling success rate (%) as single line
#                  (strategy-independent since only rows are permuted)
#
# Usage
# ─────
#   Run benchmark + plot (default):
#       python peeling_reorder_benchmark.py
#
#   Benchmark only:
#       python peeling_reorder_benchmark.py --benchmark
#
#   Plot only (requires stat files):
#       python peeling_reorder_benchmark.py --plot
#
#   Single code family:
#       python peeling_reorder_benchmark.py --code n625
#
#   Specific strategies only:
#       python peeling_reorder_benchmark.py --strategies none dfs
#
#   Custom directories:
#       python peeling_reorder_benchmark.py --data-dir ./codes/ --stat-dir ./stats/
#
# From a notebook or another script:
#   from peeling_reorder_benchmark import run_benchmark, plot_benchmark
#   run_benchmark(code_names=["n625"], data_dir=".")
#   plot_benchmark(code_names=["n625"], stat_dir=".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import time
import os
import argparse
import msgpack
import networkx as nx
from scipy.sparse import (
    csr_matrix, eye as speye, kron as spkron,
    hstack as sphstack, bmat as spbmat,
)
from scipy.sparse.csgraph import reverse_cuthill_mckee, depth_first_order
from core.sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3

# ── Experiment parameters ──────────────────────────────────────────────────
ERASURE_RATES = [round(r, 2) for r in np.arange(0.05, 0.51, 0.02)]
N_TRIALS      = 50
RANDOM_SEED   = 42

CODE_FAMILIES = {
    "n625"  : {
        "file"  : "PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt",
        "label" : "[[625,25]]",
    },
    "n1225" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt",
        "label" : "[[1225,65]]",
    },
    "n1600" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt",
        "label" : "[[1600,64]]",
    },
    "n2025" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt",
        "label" : "[[2025,81]]",
    },
}

# ── Visual encoding ────────────────────────────────────────────────────────
STRATEGY_STYLE = {
    "none": {"linestyle": "-",  "marker": "o", "label": "none (original)"},
    "dfs" : {"linestyle": "--", "marker": "s", "label": "DFS"},
    "rcm" : {"linestyle": ":",  "marker": "D", "label": "RCM"},
}
COLOR_PEEL    = "#e74c3c"   # red  — peeling time
COLOR_GE      = "#3498db"   # blue — GE time
COLOR_SUCCESS = "#95a5a6"   # gray — peeling success rate (single line)


# ── Reordering strategies ──────────────────────────────────────────────────
def no_reorder(H):
    """
    Identity — returns H unchanged with identity row permutation.
    t_reorder will be effectively zero.

    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)

    Returns:
        H:             same numpy array (not copied)
        cons_ordering: list of int, [0, 1, ..., m-1]
    """
    return H, list(range(H.shape[0]))


def dfs_reorder(H):
    """
    Reorder rows of H using DFS post-order traversal of the Tanner graph.
    Only rows are permuted — columns stay in original order so that
    erasure_index_set indices remain valid.

    Uses scipy sparse graph DFS — avoids NetworkX Python overhead.
    Accepts both dense numpy arrays and scipy sparse matrices.

    Strategy:
        Build bipartite Tanner graph as sparse block matrix:
            [0    H  ]
            [H.T  0  ]
        DFS from node 0 visits check nodes (first num_rows nodes)
        in a traversal-order permutation.

    Inputs:
        H: numpy 2D array or scipy sparse matrix, shape (m, n)

    Returns:
        H_reordered:   same type as H, shape (m, n)
        cons_ordering: list of int, row permutation applied
    """
    H_sp     = csr_matrix(H)
    num_rows = H_sp.shape[0]
    num_cols = H_sp.shape[1]

    # Build bipartite adjacency: block [[0, H], [H.T, 0]]
    zero_rr = csr_matrix((num_rows, num_rows), dtype=np.int8)
    zero_cc = csr_matrix((num_cols, num_cols), dtype=np.int8)
    H_int   = H_sp.astype(np.int8)
    tanner  = spbmat(
        [[zero_rr, H_int], [H_int.T, zero_cc]], format="csr"
    )

    # DFS from node 0 — scipy C-level, much faster than NetworkX
    node_order, _ = depth_first_order(tanner, i_start=0, directed=False)

    # Check nodes are indices 0..num_rows-1 in the block layout
    cons_ordering = [i for i in node_order if i < num_rows]

    # Handle disconnected components — any check not reached by DFS
    visited = set(cons_ordering)
    for i in range(num_rows):
        if i not in visited:
            cons_ordering.append(i)

    if hasattr(H, "toarray"):
        return H_sp[cons_ordering, :], cons_ordering
    return np.asarray(H)[cons_ordering, :], cons_ordering


def cm_reorder(H):
    """
    Reorder rows of H using Reverse Cuthill-McKee (RCM) algorithm.

    RCM minimises matrix bandwidth — nonzeros move closer to the diagonal,
    reducing fill-in propagation during GE. Only rows are permuted so that
    erasure_index_set indices remain valid.

    Uses sparse matrix multiply throughout — avoids O(m²n) dense computation.
    Accepts both dense numpy arrays and scipy sparse matrices.

    Strategy:
        1. Build sparse row-row adjacency A = (H @ H.T > 0) — two rows
           are adjacent if they share at least one nonzero column.
           For (3,4)-regular codes: nnz(A) ≈ m × w² ≪ m²
        2. Apply scipy RCM to sparse A.
        3. Permute rows of H accordingly.

    Inputs:
        H: numpy 2D array or scipy sparse matrix, shape (m, n)

    Returns:
        H_reordered:   same type as H, shape (m, n)
        cons_ordering: list of int, row permutation applied
    """
    H_sp     = csr_matrix(H)
    num_rows = H_sp.shape[0]

    # Sparse multiply — O(m × w²) not O(m² × n)
    A_sparse = (H_sp @ H_sp.T).astype(bool)
    A_sparse.setdiag(0)
    A_sparse.eliminate_zeros()

    # Guard — if adjacency has no nonzeros (all rows disjoint) RCM has
    # nothing to do. Return identity permutation immediately.
    if A_sparse.nnz == 0:
        cons_ordering = list(range(num_rows))
        if hasattr(H, "toarray"):
            return H_sp, cons_ordering
        return np.asarray(H), cons_ordering

    perm          = reverse_cuthill_mckee(A_sparse, symmetric_mode=True)
    cons_ordering = perm.tolist()

    if hasattr(H, "toarray"):
        return H_sp[cons_ordering, :], cons_ordering
    return np.asarray(H)[cons_ordering, :], cons_ordering


# Registry — add new strategies here without changing any other code
REORDER_STRATEGIES = {
    "none": no_reorder,
    "dfs" : dfs_reorder,
    "rcm" : cm_reorder,
}


# ── HGP utilities ──────────────────────────────────────────────────────────
def load_classical_H(filepath):
    """
    Load classical H from Connolly et al. txt file.
    Line 1: "m n" header. Remaining lines: nonzero column indices per row.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: '{filepath}'\n"
            f"Download from: https://github.com/Nicholas-Connolly/"
            f"Pruned-Peeling-and-VH-Decoder"
        )
    with open(filepath, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    m, n = int(lines[0].split()[0]), int(lines[0].split()[1])
    H    = np.zeros((m, n), dtype=int)
    for row_idx, line in enumerate(lines[1:]):
        for col in [int(x) for x in line.split()]:
            H[row_idx, col] = 1
    return H


def build_hgp(H_cl):
    """Build HGP CSS code matrices Hx and Hz from classical H_cl."""
    m, n = H_cl.shape
    Im   = np.eye(m, dtype=int)
    In   = np.eye(n, dtype=int)
    Hx   = np.hstack([np.kron(H_cl, In),  np.kron(Im, H_cl.T)])
    Hz   = np.hstack([np.kron(In, H_cl),  np.kron(H_cl.T, Im)])
    return Hx, Hz


def build_hgp_sparse(H_cl):
    """
    Build HGP CSS code using scipy sparse matrices.
    Avoids dense np.kron which allocates O(m*n * N) memory.

    Memory: O(nnz) ≈ O(row_weight × N) instead of O(m*n × N).
    For (3,4)-regular codes at N=27689: ~1.8 MB vs ~6 GB dense.

    Accepts dense numpy H_cl and returns scipy csr_matrix Hx, Hz.

    Inputs:
        H_cl: numpy 2D array, dtype=int, shape (m, n)

    Returns:
        Hx: scipy csr_matrix, shape (m*n, N)
        Hz: scipy csr_matrix, shape (n*m, N)
    """
    m, n  = H_cl.shape
    H_sp  = csr_matrix(H_cl, dtype=np.int8)
    Im    = speye(m, dtype=np.int8, format="csr")
    In    = speye(n, dtype=np.int8, format="csr")

    Hx = sphstack([spkron(H_sp, In), spkron(Im, H_sp.T)], format="csr")
    Hz = sphstack([spkron(In, H_sp), spkron(H_sp.T, Im)], format="csr")
    return Hx, Hz



def build_expander(n_vars, left_degree, seed):
    """
    Build a random left-regular bipartite expander graph.

    Square construction: n_L = n_R = n_vars.
    Each left (variable) node has exactly left_degree edges.
    Right (check) nodes have varying degree (avg = left_degree).

    Returns sparse adjacency matrix G of shape (n_R, n_L) where
    G[r, l] = 1 if right node r is connected to left node l.

    Inputs:
        n_vars:      int, number of left (variable) nodes = n_L = n_R
        left_degree: int, number of edges per left node
        seed:        int, random seed

    Returns:
        G: scipy csr_matrix, dtype=int8, shape (n_vars, n_vars)
    """
    rng     = np.random.default_rng(seed)
    n_L     = n_vars
    n_R     = n_vars

    # Build edge list: each left node l connects to left_degree right nodes
    rows = []   # right node indices
    cols = []   # left node indices
    for l in range(n_L):
        targets = rng.choice(n_R, size=left_degree, replace=False)
        for r in targets:
            rows.append(int(r))
            cols.append(int(l))

    data = np.ones(len(rows), dtype=np.int8)
    G    = csr_matrix(
        (data, (rows, cols)),
        shape=(n_R, n_L),
        dtype=np.int8,
    )
    return G


def ael_amplify(H_cl, G):
    """
    AEL (Alon-Edmonds-Luby / Sipser-Spielman) amplification of a
    classical LDPC code using a bipartite expander graph.

    For each right vertex r of G:
        Let N(r) = left neighbours of r  (size = deg(r))
        Apply H_cl locally to the variables indexed by N(r)
        This contributes m rows to H_amp

    If deg(r) != n_cl (the number of columns of H_cl), we pad or
    truncate N(r) to match. In the square left-regular construction
    with left_degree = row_weight, most right nodes see exactly
    left_degree neighbours which equals n_cl only for small codes.

    For practical use we require deg(r) == H_cl.shape[1] for all r.
    The build_expander function with left_degree = H_cl.shape[1] / 2
    and appropriate n_vars satisfies this. Here we use the simpler
    approach: subsample or pad N(r) to exactly n_cl entries.

    H_amp shape: (n_R × m, n_L)
    CSS orthogonality: automatic when H_amp is used in build_hgp_sparse.

    Inputs:
        H_cl: numpy 2D array, dtype=int, shape (m, n_cl)
        G:    scipy sparse matrix, shape (n_R, n_L) — expander adjacency

    Returns:
        H_amp: numpy 2D array, dtype=int, shape (n_R * m, n_L)
    """
    m, n_cl = H_cl.shape
    n_R, n_L = G.shape
    G_csr    = G.tocsr()

    H_amp = np.zeros((n_R * m, n_L), dtype=int)

    for r in range(n_R):
        # Neighbours of right node r = column indices of row r in G
        nbrs = G_csr[r].indices.tolist()

        # Pad or subsample to exactly n_cl neighbours
        rng_local = np.random.default_rng(r)
        if len(nbrs) < n_cl:
            # Pad with random additional left nodes not already in nbrs
            candidates = [l for l in range(n_L) if l not in set(nbrs)]
            extra = rng_local.choice(
                candidates,
                size=min(n_cl - len(nbrs), len(candidates)),
                replace=False,
            ).tolist()
            nbrs = nbrs + extra
        if len(nbrs) > n_cl:
            nbrs = rng_local.choice(nbrs, size=n_cl, replace=False).tolist()

        if len(nbrs) != n_cl:
            # Degenerate case — skip this right vertex
            continue

        nbrs_sorted = sorted(nbrs)

        # Apply H_cl to variables at nbrs_sorted
        # H_amp[r*m : (r+1)*m, nbrs_sorted] = H_cl
        row_start = r * m
        row_end   = row_start + m
        for local_col, global_col in enumerate(nbrs_sorted):
            H_amp[row_start:row_end, global_col] = H_cl[:, local_col]

    return H_amp


def get_row_nonzeros(H, i):
    """
    Return nonzero column indices of row i.
    Works on both dense numpy arrays and scipy sparse matrices.

    Inputs:
        H: numpy 2D array or scipy sparse matrix
        i: int, row index

    Returns:
        indices: numpy 1D array of int, nonzero column positions
    """
    row = H[i]
    if hasattr(row, "toarray"):          # scipy sparse row
        return row.toarray().ravel().nonzero()[0]
    return np.where(row == 1)[0]         # dense numpy row


# ── Peeling decoder ────────────────────────────────────────────────────────
def peeling_decoder(H, s, erasure_index_set):
    """
    Peeling decoder for classical linear code over the BEC.

    Inputs:
        H:                 numpy 2D array, dtype=int, shape (m, n)
        s:                 numpy 1D array, dtype=int, shape (m,)
        erasure_index_set: set of int

    Returns:
        solution:          numpy 1D array, dtype=int, shape (n,)
        residual_erasure:  set of int — unresolved erased bits
        residual_syndrome: dict mapping check_index -> syndrome_bit
    """
    n_vars   = H.shape[1]
    solution = np.zeros(n_vars, dtype=int)

    check_to_vars = {}
    var_to_checks = {j: set() for j in erasure_index_set}

    # Extract all nonzero positions once — O(nnz), no per-row overhead.
    # Avoids 13,300+ individual sparse row accesses at large code sizes.
    if hasattr(H, "tocsr"):
        rows_nz, cols_nz = H.tocsr().nonzero()
    else:
        rows_nz, cols_nz = np.where(H == 1)

    # Build adjacency restricted to erased columns only
    for i, j in zip(rows_nz, cols_nz):
        if j not in erasure_index_set:
            continue
        if i not in check_to_vars:
            check_to_vars[i] = set()
        check_to_vars[i].add(j)
        var_to_checks[j].add(i)

    syndrome = {i: int(s[i]) for i in check_to_vars}
    dangling  = {i for i, nbrs in check_to_vars.items() if len(nbrs) == 1}

    while dangling:
        check = dangling.pop()
        if check not in check_to_vars:
            continue
        if len(check_to_vars[check]) != 1:
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


# ── Residual submatrix extraction ─────────────────────────────────────────
def extract_residual_submatrix(H, residual_erasure):
    """
    Extract the submatrix of H relevant to the residual stopping set.

    Restricts H to:
        Columns : residual_erasure only (sorted)
        Rows    : only rows with at least one nonzero in residual_erasure

    The result is a dense numpy array — residuals are small (tens to hundreds
    of bits) so dense representation is appropriate and avoids sparse overhead.

    Inputs:
        H:                numpy 2D array or scipy sparse matrix, shape (m, n)
        residual_erasure: set of int, column indices of unresolved erased bits

    Returns:
        H_sub:       numpy 2D array, dtype=int, shape (n_active, |residual|)
        active_rows: numpy 1D array of int, original row indices in H_sub
                     used to extract s_sub = s[active_rows]
        col_map:     numpy 1D array of int, original column indices
                     col_map[j] = original column index for H_sub column j
    """
    col_map = np.array(sorted(residual_erasure), dtype=int)

    # Find active rows via one-shot nonzero extraction
    if hasattr(H, "tocsr"):
        H_csr              = H.tocsr()
        rows_nz, cols_nz   = H_csr.nonzero()
    else:
        rows_nz, cols_nz   = np.where(H == 1)

    residual_set = residual_erasure
    active_set   = set()
    for i, j in zip(rows_nz, cols_nz):
        if j in residual_set:
            active_set.add(int(i))

    active_rows = np.array(sorted(active_set), dtype=int)

    # Build dense submatrix — small for typical residuals
    n_active   = len(active_rows)
    n_cols     = len(col_map)
    H_sub      = np.zeros((n_active, n_cols), dtype=int)

    # Reverse mapping: original column -> submatrix column index
    col_to_sub = {int(orig): sub for sub, orig in enumerate(col_map)}

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


# ── Stat file I/O ──────────────────────────────────────────────────────────
def stat_filename(code_name, stat_dir):
    return os.path.join(
        stat_dir, f"stats_peeling_reorder_{code_name}.msgpack"
    )


def save_stats(stats, code_name, stat_dir):
    """Serialize stats dict to msgpack binary file."""
    os.makedirs(stat_dir, exist_ok=True)
    fpath = stat_filename(code_name, stat_dir)
    with open(fpath, "wb") as f:
        msgpack.pack(stats, f)
    print(f"    Stats saved → {fpath}")


def load_stats(code_name, stat_dir):
    """Load stats dict from msgpack binary file. Returns None if not found."""
    fpath = stat_filename(code_name, stat_dir)
    if not os.path.exists(fpath):
        return None
    with open(fpath, "rb") as f:
        return msgpack.unpack(f)


# ── Benchmark ──────────────────────────────────────────────────────────────
def run_benchmark(
    code_names       = None,
    strategy_names   = None,
    data_dir         = ".",
    stat_dir         = ".",
    erasure_rates    = ERASURE_RATES,
    n_trials         = N_TRIALS,
    random_seed      = RANDOM_SEED,
):
    """
    Benchmark peeling + GE pipeline for each reordering strategy.

    For each strategy:
        - Reorder Hx ONCE, time recorded as t_reorder_ms
        - Per trial: peeling on H_strategy, GE on residual if needed
        - GE time averaged only over trials where GE was needed
        - Same erasure pattern used for all strategies per trial

    Peeling success rate is strategy-independent (row permutation only)
    but stored per strategy for verification. The "none" strategy is used
    as the reference line in the plot.

    Stat file schema per code family:
    {
      "code_name", "label", "erasure_rates", "n_trials", "random_seed",
      "strategies": {
        "none" | "dfs" | "rcm": {
          "t_reorder_ms": float,
          "per_rate": {
            "0.05": {
              "t_peel"     : [float,...],  all n_trials
              "t_ge"       : [float,...],  GE-needed trials only
              "n_ge_trials": int,
              "n_peel_only": int,
            }, ...
          },
          "summary": {
            "t_peel"      : {"mean":[...], "std":[...]},
            "t_ge"        : {"mean":[...], "std":[...]},
            "n_ge_trials" : [...],
            "n_peel_only" : [...],
          }
        }, ...
      }
    }

    Inputs:
        code_names:     list of str or None (None = all families)
        strategy_names: list of str or None (None = all strategies)
        data_dir:       str
        stat_dir:       str
        erasure_rates:  list of float
        n_trials:       int
        random_seed:    int
    """
    # from ge_decoder import erasure_decode_sparse_v3

    if code_names     is None:
        code_names     = list(CODE_FAMILIES.keys())
    if strategy_names is None:
        strategy_names = list(REORDER_STRATEGIES.keys())

    print("Peeling + Reorder Benchmark")
    print("───────────────────────────")
    print(f"  code families : {code_names}")
    print(f"  strategies    : {strategy_names}")
    print(f"  erasure rates : {erasure_rates[0]:.2f} to {erasure_rates[-1]:.2f} "
          f"step 0.02 ({len(erasure_rates)} points)")
    print(f"  trials        : {n_trials}")
    print(f"  random seed   : {random_seed}")
    print(f"  data dir      : {os.path.abspath(data_dir)}")
    print(f"  stat dir      : {os.path.abspath(stat_dir)}")
    print()

    for code_name in code_names:
        meta     = CODE_FAMILIES[code_name]
        filepath = os.path.join(data_dir, meta["file"])
        label    = meta["label"]

        print(f"── {label} ──────────────────────────────────")

        try:
            H_cl = load_classical_H(filepath)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}\n")
            continue

        Hx, _ = build_hgp(H_cl)
        N      = Hx.shape[1]
        print(f"  Hx shape : {Hx.shape}   N={N}")

        # ── Apply and time each reordering strategy once ───────────────────
        reordered = {}
        for sname in strategy_names:
            fn = REORDER_STRATEGIES[sname]
            t0 = time.perf_counter()
            H_strat, _ = fn(Hx)
            t_ms = (time.perf_counter() - t0) * 1000
            reordered[sname] = {"H": H_strat, "t_reorder_ms": t_ms}
            print(f"  t_reorder [{sname:4s}] : {t_ms:.2f} ms")
        print()

        rng = np.random.default_rng(random_seed)

        # ── Raw storage initialisation ─────────────────────────────────────
        strategy_raw = {
            sname: {
                str(r): {
                    "t_peel"     : [],
                    "t_ge"       : [],
                    "n_ge_trials": 0,
                    "n_peel_only": 0,
                }
                for r in erasure_rates
            }
            for sname in strategy_names
        }

        # ── Trial loop ─────────────────────────────────────────────────────
        for rate in erasure_rates:
            n_erased = int(N * rate)
            sx       = np.zeros(Hx.shape[0], dtype=int)

            print(f"  rate={rate:.2f}  |ε|={n_erased:4d}", end="  ")

            for trial in range(n_trials):
                # Same erasure pattern for all strategies — fair comparison
                erased_bits = rng.choice(N, size=n_erased, replace=False)
                erasure_set = set(erased_bits.tolist())

                for sname in strategy_names:
                    H_strat = reordered[sname]["H"]
                    bucket  = strategy_raw[sname][str(rate)]

                    # Peeling — timed for all trials
                    t0 = time.perf_counter()
                    _, residual, res_syn = peeling_decoder(
                        H_strat, sx, erasure_set
                    )
                    bucket["t_peel"].append(
                        (time.perf_counter() - t0) * 1000
                    )

                    # GE fallback — timed only when residual non-empty
                    if residual:
                        s_res = np.array(
                            [res_syn.get(i, 0) for i in range(Hx.shape[0])],
                            dtype=int
                        )
                        t0 = time.perf_counter()
                        erasure_decode_sparse_v3(H_strat, s_res, residual)
                        bucket["t_ge"].append(
                            (time.perf_counter() - t0) * 1000
                        )
                        bucket["n_ge_trials"] += 1
                    else:
                        bucket["n_peel_only"] += 1

            # Console summary per strategy per rate
            for sname in strategy_names:
                b         = strategy_raw[sname][str(rate)]
                peel_mean = float(np.mean(b["t_peel"]))
                ge_mean   = float(np.mean(b["t_ge"])) if b["t_ge"] else 0.0
                peel_pct  = b["n_peel_only"] / n_trials * 100
                print(
                    f"[{sname}] "
                    f"peel={peel_mean:.3f}ms "
                    f"ge={ge_mean:.3f}ms(n={b['n_ge_trials']}) "
                    f"peel%={peel_pct:.0f}%",
                    end="  "
                )
            print()

        # ── Summary statistics ─────────────────────────────────────────────
        def summarise(raw_per_rate, key, rates):
            means, stds = [], []
            for r in rates:
                vals = raw_per_rate[str(r)][key]
                means.append(float(np.mean(vals)) if vals else 0.0)
                stds.append( float(np.std(vals))  if vals else 0.0)
            return {"mean": means, "std": stds}

        strategies_block = {}
        for sname in strategy_names:
            raw = strategy_raw[sname]
            strategies_block[sname] = {
                "t_reorder_ms": float(reordered[sname]["t_reorder_ms"]),
                "per_rate"    : raw,
                "summary"     : {
                    "t_peel"     : summarise(raw, "t_peel", erasure_rates),
                    "t_ge"       : summarise(raw, "t_ge",   erasure_rates),
                    "n_ge_trials": [raw[str(r)]["n_ge_trials"]
                                    for r in erasure_rates],
                    "n_peel_only": [raw[str(r)]["n_peel_only"]
                                    for r in erasure_rates],
                },
            }

        stats = {
            "code_name"    : code_name,
            "label"        : label,
            "erasure_rates": erasure_rates,
            "n_trials"     : n_trials,
            "random_seed"  : random_seed,
            "strategies"   : strategies_block,
        }
        save_stats(stats, code_name, stat_dir)
        print()

    print("Benchmark complete.")


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_benchmark(
    code_names = None,
    stat_dir   = ".",
    plot_file  = "peeling_reorder_benchmark.png",
):
    """
    Load per-family msgpack stat files and produce a 2×2 figure.

    One subplot per code family:
        Left Y-axis  — avg time per call (ms): peeling + GE for all strategies
        Right Y-axis — peeling success rate (%) as single gray line
                       uses "none" strategy as reference (strategy-independent)

    Visual encoding:
        Color      : red=peeling time, blue=GE time, gray=success rate
        Line style : solid=none, dashed=dfs, dotted=rcm
        Marker     : o=none, s=dfs, D=rcm

    X-axis ticks: major at 0.10 intervals, minor at every 0.02 point.

    Raw data for all strategies preserved in msgpack files.

    Inputs:
        code_names: list of str or None (None = all with stat files)
        stat_dir:   str
        plot_file:  str, output PNG path
    """
    if code_names is None:
        code_names = list(CODE_FAMILIES.keys())

    loaded = {}
    for code_name in code_names:
        stats = load_stats(code_name, stat_dir)
        if stats is None:
            print(f"  SKIP {code_name}: stat file not found — "
                  f"run --benchmark first")
            continue
        loaded[code_name] = stats

    if not loaded:
        raise FileNotFoundError(
            f"No stat files found in '{stat_dir}'. Run --benchmark first."
        )

    n_codes = len(loaded)
    ncols   = 2
    nrows   = (n_codes + 1) // 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(13, 5 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        "Peeling + Sparse GE v3 — Reordering Strategy Comparison\n"
        "Left Y: avg time per call (ms)   |   "
        "Right Y: peeling success rate (%) — strategy-independent\n"
        "Peeling time = all trials   |   "
        "GE time = GE-needed trials only   |   "
        "t_reorder = one-time cost",
        fontsize=10,
    )

    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax_idx, (code_name, stats) in enumerate(loaded.items()):
        ax      = axes_flat[ax_idx]
        ax_twin = ax.twinx()

        erasure_rates  = stats["erasure_rates"]
        n_trials       = stats["n_trials"]
        label          = stats["label"]
        strategies     = stats["strategies"]
        strategy_names = list(strategies.keys())

        # ── Left Y-axis — timing ───────────────────────────────────────────
        for sname in strategy_names:
            style   = STRATEGY_STYLE[sname]
            summary = strategies[sname]["summary"]

            peel_mean = np.array(summary["t_peel"]["mean"])
            peel_std  = np.array(summary["t_peel"]["std"])
            ge_mean   = np.array(summary["t_ge"]["mean"])
            ge_std    = np.array(summary["t_ge"]["std"])
            n_ge      = np.array(summary["n_ge_trials"])

            # Peeling time — all trials
            ax.plot(
                erasure_rates, peel_mean,
                color=COLOR_PEEL,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1, markersize=3,
                label=f"peel [{style['label']}]",
            )
            ax.fill_between(
                erasure_rates,
                np.maximum(peel_mean - peel_std, 0),
                peel_mean + peel_std,
                color=COLOR_PEEL, alpha=0.08,
            )

            # GE time — GE-needed trials only, masked where n_ge == 0
            ge_mask  = n_ge > 0
            ge_rates = [erasure_rates[i]
                        for i in range(len(erasure_rates)) if ge_mask[i]]
            ge_vals  = ge_mean[ge_mask]
            ge_err   = ge_std[ge_mask]
            ge_n     = n_ge[ge_mask]

            if ge_rates:
                ax.plot(
                    ge_rates, ge_vals,
                    color=COLOR_GE,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=1, markersize=3,
                    label=f"GE [{style['label']}]",
                )
                ax.fill_between(
                    ge_rates,
                    np.maximum(ge_vals - ge_err, 0),
                    ge_vals + ge_err,
                    color=COLOR_GE, alpha=0.08,
                )
                # Annotate n= on each GE data point
                for r, v, n in zip(ge_rates, ge_vals, ge_n):
                    ax.annotate(
                        f"n={n}",
                        xy=(r, v), xytext=(0, 5),
                        textcoords="offset points",
                        fontsize=5, color=COLOR_GE, ha="center",
                    )

        # ── Right Y-axis — peeling success rate (single line) ─────────────
        # Use "none" as reference — strategy-independent since only rows
        # are permuted and peeling success depends on column connectivity only
        ref_strategy = "none" if "none" in strategies else strategy_names[0]
        n_peel   = np.array(strategies[ref_strategy]["summary"]["n_peel_only"])
        peel_pct = n_peel / n_trials * 100

        ax_twin.plot(
            erasure_rates, peel_pct,
            color=COLOR_SUCCESS,
            linestyle="-", linewidth=1.2,
            label="peeling success %",
        )
        ax_twin.set_ylabel("Peeling success rate (%)", fontsize=8,
                           color=COLOR_SUCCESS)
        ax_twin.tick_params(axis="y", labelcolor=COLOR_SUCCESS, labelsize=7)
        ax_twin.set_ylim(-5, 105)
        ax_twin.yaxis.set_major_formatter(mticker.PercentFormatter())

        # ── t_reorder annotation — text box inside plot ────────────────────
        t_lines = "\n".join(
            f"{sname}: {strategies[sname]['t_reorder_ms']:.1f}ms"
            for sname in strategy_names
        )
        ax.text(
            0.02, 0.98,
            f"t_reorder (one-time)\n{t_lines}",
            transform=ax.transAxes,
            fontsize=6, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8),
        )

        # ── Axes formatting ────────────────────────────────────────────────
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Erasure rate", fontsize=9)
        ax.set_ylabel("Avg time per call (ms)", fontsize=9)

        # Major ticks at 0.10 intervals, minor ticks at every 0.02 point
        ax.set_xticks([0.10, 0.20, 0.30, 0.40, 0.50])
        ax.set_xticks(erasure_rates, minor=True)
        ax.tick_params(axis="x", which="major", labelsize=8)
        ax.tick_params(axis="x", which="minor", length=3)
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":",  alpha=0.2)

        # ── Combined legend ────────────────────────────────────────────────
        lines_l, labels_l = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_twin.get_legend_handles_labels()
        ax.legend(
            lines_l + lines_r,
            labels_l + labels_r,
            fontsize=7, loc="upper left",
            ncol=2,
        )

    # Hide unused subplots if n_codes is odd
    for ax_idx in range(len(loaded), nrows * ncols):
        axes_flat[ax_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {plot_file}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark peeling + Sparse GE v3 across reordering strategies.\n"
            "Default (no flags): runs benchmark then plots."
        )
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark and save stats to msgpack files."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Load stats and produce plot."
    )
    parser.add_argument(
        "--code", default="all",
        choices=list(CODE_FAMILIES.keys()) + ["all"],
        help="Code family to benchmark (default: all)."
    )
    parser.add_argument(
        "--strategies", nargs="+",
        default=list(REORDER_STRATEGIES.keys()),
        choices=list(REORDER_STRATEGIES.keys()),
        help="Reordering strategies to run (default: all)."
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing classical H txt files (default: .)."
    )
    parser.add_argument(
        "--stat-dir", default=".",
        help="Directory for msgpack stat files (default: .)."
    )
    parser.add_argument(
        "--plot-file", default="peeling_reorder_benchmark.png",
        help="Output PNG filename (default: peeling_reorder_benchmark.png)."
    )
    parser.add_argument(
        "--trials", type=int, default=N_TRIALS,
        help=f"Trials per (code, erasure rate) point (default: {N_TRIALS})."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})."
    )
    args = parser.parse_args()

    if not args.benchmark and not args.plot:
        args.benchmark = True
        args.plot      = True

    code_names = (
        list(CODE_FAMILIES.keys())
        if args.code == "all"
        else [args.code]
    )

    if args.benchmark:
        run_benchmark(
            code_names     = code_names,
            strategy_names = args.strategies,
            data_dir       = args.data_dir,
            stat_dir       = args.stat_dir,
            n_trials       = args.trials,
            random_seed    = args.seed,
        )

    if args.plot:
        plot_benchmark(
            code_names = code_names,
            stat_dir   = args.stat_dir,
            plot_file  = args.plot_file,
        )