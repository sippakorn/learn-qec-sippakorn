"""Demo: record an F₂ Gaussian elimination session on a random binary matrix.

Run from the project root:
    python record_replay/main_f2.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scipy.sparse as sp

from recorder import Recorder
from generator_f2 import F2GaussianEliminationGenerator


# ── Experiment parameters ──────────────────────────────────────────────────
ERASURE_RATES = [round(r, 2) for r in np.arange(0.05, 0.51, 0.05)]
N_TRIALS      = 50
RANDOM_SEED   = 42

CODE_FAMILIES = {
    "n625"  : {
        "file"  : "PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt",
        "label" : "[[625,25]]",
        "color" : "#e74c3c",
    },
    "n1225" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt",
        "label" : "[[1225,65]]",
        "color" : "#3498db",
    },
    "n1600" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt",
        "label" : "[[1600,64]]",
        "color" : "#2ecc71",
    },
    "n2025" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt",
        "label" : "[[2025,81]]",
        "color" : "#9b59b6",
    },
}


# ── HGP utilities (mirrors run_hgp_tests.py) ──────────────────────────────
def load_classical_H(filepath):
    """
    Load classical H from Connolly et al. txt file.
    Format: line 1 = "m n", remaining lines = nonzero column indices per row.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: '{filepath}'\n"
            f"Download from: https://github.com/Nicholas-Connolly/"
            f"Pruned-Peeling-and-VH-Decoder"
        )

    with open(filepath, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    header = lines[0].split()
    m, n   = int(header[0]), int(header[1])

    H = np.zeros((m, n), dtype=int)
    for row_idx, line in enumerate(lines[1:]):
        for col in [int(x) for x in line.split()]:
            H[row_idx, col] = 1
    return H


def build_hgp(H_cl):
    """Build HGP CSS code: Hx = [H⊗I_n | I_m⊗H^T], Hz = [I_n⊗H | H^T⊗I_m]."""
    m, n = H_cl.shape
    Im   = np.eye(m, dtype=int)
    In   = np.eye(n, dtype=int)
    Hx   = np.hstack([np.kron(H_cl, In),  np.kron(Im, H_cl.T)])
    Hz   = np.hstack([np.kron(In, H_cl),  np.kron(H_cl.T, Im)])
    return Hx, Hz


# ── Stat file I/O using msgpack ────────────────────────────────────────────
def stat_filename(code_name, stat_dir):
    """Return path to msgpack stat file for a given code family."""
    return os.path.join(stat_dir, f"stats_reorder_{code_name}.msgpack")


def save_stats(stats, code_name, stat_dir):
    """
    Serialize stats dict to a msgpack binary file.

    Stats format:
        {
          "code_name":    str,
          "label":        str,
          "erasure_rates": [float, ...],
          "n_trials":     int,
          "random_seed":  int,
          "original": {
              "mean": [float, ...],   # one per erasure rate
              "std":  [float, ...]
          },
          "reordered": {
              "mean": [float, ...],
              "std":  [float, ...]
          }
        }
    """
    os.makedirs(stat_dir, exist_ok=True)
    fpath = stat_filename(code_name, stat_dir)
    with open(fpath, "wb") as f:
        msgpack.pack(stats, f)
    print(f"    Stats saved → {fpath}")


def load_stats(code_name, stat_dir):
    """
    Load stats dict from msgpack binary file.
    Returns None if file does not exist.
    """
    fpath = stat_filename(code_name, stat_dir)
    if not os.path.exists(fpath):
        return None
    with open(fpath, "rb") as f:
        stats = msgpack.unpack(f)
    return stats



# ── Plot ───────────────────────────────────────────────────────────────────
def plot_benchmark(
    code_names = None,
    stat_dir   = ".",
    plot_file  = "reorder_benchmark.png",
):
    """
    Load per-family msgpack stat files and produce a single plot.

    Visual encoding:
        Color     = code family  (one color per family)
        Solid     = original Hx
        Dashed    = DFS-reordered Hx
        Shading   = ± 1 std band

    Inputs:
        code_names: list of str or None (None = all families with stat files)
        stat_dir:   str, directory containing msgpack stat files
        plot_file:  str, output PNG path

    Raises:
        FileNotFoundError if no stat files are found
    """
    if code_names is None:
        code_names = list(CODE_FAMILIES.keys())

    # Load available stats
    loaded = {}
    for code_name in code_names:
        stats = load_stats(code_name, stat_dir)
        if stats is None:
            print(f"  SKIP {code_name}: stat file not found — run --benchmark first")
            continue
        loaded[code_name] = stats

    if not loaded:
        raise FileNotFoundError(
            f"No stat files found in '{stat_dir}'. "
            f"Run with --benchmark first."
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    for code_name, stats in loaded.items():
        color         = CODE_FAMILIES[code_name]["color"]
        label         = stats["label"]
        erasure_rates = stats["erasure_rates"]

        for version, linestyle in [("original", "-"), ("reordered", "--")]:
            mean = np.array(stats[version]["mean"])
            std  = np.array(stats[version]["std"])

            ax.plot(
                erasure_rates, mean,
                color=color, linestyle=linestyle,
                linewidth=1, markersize=3,
                marker="o" if version == "original" else "s",
                label=f"{label} {version}",
            )
            ax.fill_between(
                erasure_rates,
                mean - std, mean + std,
                color=color, alpha=0.10,
            )

    # ── Legend — two-part: colors for families, styles for versions ───────
    family_handles = [
        mlines.Line2D([], [],
                      color=CODE_FAMILIES[cn]["color"],
                      linewidth=2,
                      label=CODE_FAMILIES[cn]["label"])
        for cn in loaded
    ]
    version_handles = [
        mlines.Line2D([], [], color="gray", linestyle="-",
                      linewidth=1.5, label="original"),
        mlines.Line2D([], [], color="gray", linestyle="--",
                      linewidth=1.5, label="reordered"),
    ]

    legend1 = ax.legend(
        handles=family_handles,
        title="Code family",
        loc="upper left",
        fontsize=9, title_fontsize=9,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=version_handles,
        title="Version",
        loc="upper center",
        fontsize=9, title_fontsize=9,
    )

    ax.set_xlabel("Erasure rate", fontsize=11)
    ax.set_ylabel("Avg decoding time per call (ms)", fontsize=11)
    ax.set_title(
        "Sparse GE v3 — Original vs DFS-Reordered HGP\n"
        f"({stats['n_trials']} trials per point, reorder time excluded)",
        fontsize=11,
    )
    ax.set_xticks(erasure_rates)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {plot_file}")


def dfs_reorder(H):
    import networkx as nx

    # build bipartite Tanner graph from H, using NetworkX graph representation
    # each variable (column) becomes a node, each constraint (row) is also a node;
    num_rows, num_cols = H.shape
    G = nx.Graph()
    # Add variable nodes (columns) and check nodes (rows) with distinct labels
    var_nodes = [('v', j) for j in range(num_cols)]
    chk_nodes = [('c', i) for i in range(num_rows)]
    G.add_nodes_from(var_nodes)
    G.add_nodes_from(chk_nodes)
    # Add edges wherever H[i, j] == 1
    for i in range(num_rows):
        for j in range(num_cols):
            if H[i, j]:
                G.add_edge(('c', i), ('v', j))


    # Compute a DFS ordering of the variable nodes
    dfs_ordering = list(nx.dfs_postorder_nodes(G))

    var_ordering = [node[1] for node in dfs_ordering if node[0] == 'v']
    cons_ordering = [node[1] for node in dfs_ordering if node[0] == 'c']
    
    row_dfs_ordering = True

    if row_dfs_ordering:
        H_reordered = H[np.ix_(cons_ordering, var_ordering)]
    else:
        # order rows by indices of its first non-zero column
        cons_ordering = range(num_rows)
        H2 = H[np.ix_(cons_ordering, var_ordering)]

        first_col = []
        for i in range(num_rows):
            mn = num_rows+100
            for j in range(num_cols):
                if H2[i, j]:
                    mn = j
                    break
            first_col.append((mn, i))

        first_col.sort()
        new_row_order = [row for _, row in first_col]
        cons_ordering = new_row_order
        H_reordered = H2[np.ix_(new_row_order, range(num_cols))]

    return H_reordered, cons_ordering, var_ordering

def peeling_decoder(H, s, erasure_index_set):
    """
    Peeling decoder for classical linear code over the binary erasure channel.

    Iteratively resolves erased variables by finding dangling checks —
    check nodes with exactly one erased variable neighbour. When no
    dangling check exists, peeling is stuck and returns the residual.

    Inputs:
        H:                 numpy 2D array, dtype=int, shape (m, n)
        s:                 numpy 1D array, dtype=int, shape (m,)
        erasure_index_set: set of int, indices of erased bits

    Returns:
        solution:         numpy 1D array, dtype=int, shape (n,)
                          resolved bits set to their values,
                          unresolved bits set to 0
        residual_erasure: set of int
                          erased bits not resolved by peeling
                          empty set means peeling fully succeeded
        residual_syndrome: dict mapping check_index -> syndrome_bit
                           syndrome of checks still connected to
                           residual erasure — needed for GE fallback
    """
    n_vars   = H.shape[1]
    solution = np.zeros(n_vars, dtype=int)

    # ── Step 1 — Build adjacency structures ───────────────────────────────
    # check_to_vars[i] = set of erased variable indices connected to check i
    # var_to_checks[j] = set of check indices connected to erased variable j
    check_to_vars = {}
    var_to_checks = {j: set() for j in erasure_index_set}

    for i in range(H.shape[0]):
        neighbours = set(
            j for j in np.where(H[i] == 1)[0]
            if j in erasure_index_set
        )
        if neighbours:                      # skip checks with no erased neighbours
            check_to_vars[i] = neighbours
            for j in neighbours:
                var_to_checks[j].add(i)

    # ── Step 2 — Working syndrome (mutable copy, only active checks) ──────
    syndrome = {i: int(s[i]) for i in check_to_vars}

    # ── Step 3 — Initialise dangling queue ────────────────────────────────
    # Use a set for O(1) membership test and removal
    dangling = {i for i, nbrs in check_to_vars.items() if len(nbrs) == 1}

    # ── Step 4 — Peeling loop ─────────────────────────────────────────────
    while dangling:

        # Pop one dangling check
        check = dangling.pop()

        # Guard: check may have been invalidated by an earlier peel step
        # (can happen if two dangling checks shared a variable)
        if check not in check_to_vars:
            continue
        if len(check_to_vars[check]) != 1:
            continue

        # Identify and resolve the single erased variable
        var          = next(iter(check_to_vars[check]))
        var_value    = syndrome[check]
        solution[var] = var_value

        # ── Step 5 — Propagate to neighbouring checks ─────────────────────
        for neighbour_check in var_to_checks[var]:
            if neighbour_check == check:
                continue
            if neighbour_check not in check_to_vars:
                continue

            # Update syndrome
            syndrome[neighbour_check] ^= var_value

            # Remove resolved variable from neighbour
            check_to_vars[neighbour_check].discard(var)

            # Check if neighbour became dangling
            if len(check_to_vars[neighbour_check]) == 1:
                dangling.add(neighbour_check)

            # Check if neighbour became empty (all its variables resolved)
            elif len(check_to_vars[neighbour_check]) == 0:
                del check_to_vars[neighbour_check]
                del syndrome[neighbour_check]

        # ── Step 6 — Remove resolved variable and check from graph ────────
        del var_to_checks[var]
        del check_to_vars[check]
        del syndrome[check]

    # ── Step 7 — Collect residual ─────────────────────────────────────────
    residual_erasure = set(var_to_checks.keys())

    return solution, residual_erasure, syndrome


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    meta     = CODE_FAMILIES["n625"]
    filepath = os.path.join("codes", meta["file"])
    label    = meta["label"]

    print(f"── {label} ──────────────────────────────────")

    # ------------------------------------------------------------------ #
    # 1. Load classical H and build HGP                                   #
    # ------------------------------------------------------------------ #
    try:
        H_cl = load_classical_H(filepath)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}\n")
        return

    Hx, _ = build_hgp(H_cl)
    M, N  = Hx.shape
    rate  = 0.40

    # ------------------------------------------------------------------ #
    # 2. Random column erasure                                            #
    # ------------------------------------------------------------------ #
    n_erased    = int(N * rate)
    erased_bits = rng.choice(N, size=n_erased, replace=False)
    erasure_set = set(erased_bits.tolist())
    s           = rng.integers(0, 2, size=M, dtype=np.int32)

    print(f"Matrix shape   : {Hx.shape}, erased bits: {n_erased}/{N} ({rate:.0%})")

    # ------------------------------------------------------------------ #
    # 3. Peeling decoder on the erased matrix                             #
    # ------------------------------------------------------------------ #
    _, residual_erasure, residual_syndrome = peeling_decoder(Hx, s, erasure_set)
    print(f"After peeling  : {len(residual_erasure)} residual bits, "
          f"{len(residual_syndrome)} active checks")

    if not residual_erasure:
        print("Peeling fully resolved — nothing left for GE.")
        return

    # ------------------------------------------------------------------ #
    # 4. Extract residual submatrix                                       #
    # ------------------------------------------------------------------ #
    row_indices = sorted(residual_syndrome.keys())
    col_indices = sorted(residual_erasure)
    H_sub = Hx[np.ix_(row_indices, col_indices)]
    s_sub = np.array([residual_syndrome[i] for i in row_indices], dtype=np.int32)

    # ------------------------------------------------------------------ #
    # 5. DFS reorder residual submatrix → H_active                       #
    # ------------------------------------------------------------------ #
    H_active, row_order, _ = dfs_reorder(H_sub)
    s_active = s_sub[np.array(row_order)]

    print(f"After reorder  : {H_active.shape}")

    # ------------------------------------------------------------------ #
    # Start recording session — initial state is the augmented matrix     #
    # ------------------------------------------------------------------ #
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    recorder = Recorder(data_dir=data_dir)

    # The generator builds H_aug internally; we store it as the baseline
    H_aug_initial = np.hstack((H_active, s_active[:, np.newaxis])).astype(np.float64)
    session_id = recorder.start_session(sp.csr_matrix(H_aug_initial))
    print(f"Session started: {session_id}")

    # ------------------------------------------------------------------ #
    # Run F₂ GE, emitting events into the recorder                        #
    # ------------------------------------------------------------------ #
    gen = F2GaussianEliminationGenerator(recorder, H_active, s_active)
    pivot_cols, free_cols = gen.run()

    # ------------------------------------------------------------------ #
    # Close and report                                                     #
    # ------------------------------------------------------------------ #
    summary = recorder.close()
    total_bytes = sum(summary["file_sizes"].values())

    print()
    print("=" * 48)
    print("  Session Summary")
    print("=" * 48)
    print(f"  Session ID      : {summary['session_id']}")
    print(f"  Total steps     : {summary['total_events']:,}")
    print(f"  Checkpoints     : {summary['total_checkpoints']}")
    print(f"  Total on disk   : {_human_bytes(total_bytes)}")
    print(f"  Pivot columns   : {len(pivot_cols)}")
    print(f"  Free columns    : {len(free_cols)}")
    print()
    print("  Files:")
    for fname, size in summary["file_sizes"].items():
        print(f"    {fname:<35s}  {_human_bytes(size):>10s}")
    print("=" * 48)
    print()
    print(f"View with: python record_replay/main_replay.py --session {summary['session_id']}")


if __name__ == "__main__":
    main()
