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
        H_reordered = H2[np.ix_(new_row_order, range(num_cols))]

    return H_reordered


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    # ------------------------------------------------------------------ #
    # Build random F₂ matrix and syndrome vector                          #
    # ------------------------------------------------------------------ #
    m, n = 80, 100          # rows × columns
    density = 0.25
    rng = np.random.default_rng(42)

    H = rng.integers(0, 2, size=(m, n), dtype=np.int32)
    # Make it sparse-ish to keep the session manageable
    mask = rng.random(size=(m, n)) < density
    H = (H * mask).astype(np.int32)
    s = rng.integers(0, 2, size=m, dtype=np.int32)

    nnz = int(np.count_nonzero(H))
    print(f"Initial matrix : {m}×{n}, density≈{density:.0%}, nnz={nnz}")
    print(f"Syndrome vector: {s}")


    meta     = CODE_FAMILIES["n625"]
    filepath = os.path.join("codes", meta["file"])
    label    = meta["label"]

    print(f"── {label} ──────────────────────────────────")

    # Load and build
    try:
        H_cl = load_classical_H(filepath)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}\n")

    Hx, _      = build_hgp(H_cl)
    Hx_reorder = dfs_reorder(Hx)
    N          = Hx_reorder.shape[1]
    rate       = 0.35

    # Erasure decoding setup: zero out non-erased columns → H_active
    n_erased    = int(N * rate)
    erased_bits = rng.choice(N, size=n_erased, replace=False)
    erasure_set = set(erased_bits.tolist())

    H_active = Hx_reorder.copy()
    for bit_idx in range(N):
        if bit_idx not in erasure_set:
            H_active[:, bit_idx] = 0

    s = (rng.random(size=Hx_reorder.shape[0]) < rate).astype(np.int32)

    print(f"Matrix shape   : {Hx_reorder.shape}, erased bits: {n_erased}/{N} ({rate:.0%})")

    # ------------------------------------------------------------------ #
    # Start recording session — initial state is the augmented matrix     #
    # ------------------------------------------------------------------ #
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    recorder = Recorder(data_dir=data_dir)

    # The generator builds H_aug internally; we store it as the baseline
    H_aug_initial = np.hstack((H_active, s[:, np.newaxis])).astype(np.float64)
    session_id = recorder.start_session(sp.csr_matrix(H_aug_initial))
    print(f"Session started: {session_id}")

    # ------------------------------------------------------------------ #
    # Run F₂ GE, emitting events into the recorder                        #
    # ------------------------------------------------------------------ #
    gen = F2GaussianEliminationGenerator(recorder, H_active, s)
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
