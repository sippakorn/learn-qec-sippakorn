# reorder_benchmark.py
#
# Compare decoding time of Sparse GE v3 on original vs DFS-reordered
# HGP parity-check matrices across code families and erasure rates.
#
# X-axis : erasure rate (0.05 to 0.50)
# Y-axis : average decoding time per call (ms, linear scale)
# Lines  : 4 code families × 2 versions (original / reordered) = 8 lines
#          Color = code family,  line style = original (solid) / reordered (dashed)
#
# Raw stats saved per code family as msgpack binary files:
#   stats_reorder_n625.msgpack
#   stats_reorder_n1225.msgpack
#   stats_reorder_n1600.msgpack
#   stats_reorder_n2025.msgpack
#
# Usage
# ─────
#   Run benchmark + plot (default):
#       python reorder_benchmark.py
#
#   Run benchmark only:
#       python reorder_benchmark.py --benchmark
#
#   Plot only (requires stat files to exist):
#       python reorder_benchmark.py --plot
#
#   Single code family:
#       python reorder_benchmark.py --code n625
#
#   Custom paths:
#       python reorder_benchmark.py --data-dir ./codes/ --stat-dir ./stats/
#
# From a notebook or another script:
#   from reorder_benchmark import run_benchmark, plot_benchmark
#   run_benchmark(code_names=["n625"], data_dir=".")
#   plot_benchmark(code_names=["n625"], stat_dir=".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time
import os
import argparse
import msgpack
import networkx as nx

from sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3

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


# ── DFS reorder (from utility.py) ─────────────────────────────────────────
def dfs_reorder(H):
    """
    Reorder rows of H using DFS post-order on the Tanner graph.
    Only rows are permuted — columns stay in original order so that
    erasure_index_set indices remain valid after reordering.

    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)

    Returns:
        H_reordered: numpy 2D array, dtype=int, shape (m, n)
        cons_ordering: list of int, row permutation applied
    """
    import networkx as nx

    num_rows, num_cols = H.shape
    G = nx.Graph()

    var_nodes = [('v', j) for j in range(num_cols)]
    chk_nodes = [('c', i) for i in range(num_rows)]
    G.add_nodes_from(var_nodes)
    G.add_nodes_from(chk_nodes)

    # Fix 1 — use np.where instead of double loop
    rows_idx, cols_idx = np.where(H == 1)
    for i, j in zip(rows_idx, cols_idx):
        G.add_edge(('c', i), ('v', j))

    dfs_ordering  = list(nx.dfs_postorder_nodes(G))
    cons_ordering = [node[1] for node in dfs_ordering if node[0] == 'c']

    # Fix 2 — reorder rows only, leave columns unchanged
    H_reordered = H[cons_ordering, :]

    return H_reordered, cons_ordering


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


# ── Benchmark ──────────────────────────────────────────────────────────────
def run_benchmark(
    code_names    = None,
    data_dir      = ".",
    stat_dir      = ".",
    erasure_rates = ERASURE_RATES,
    n_trials      = N_TRIALS,
    random_seed   = RANDOM_SEED,
):
    """
    Benchmark Sparse GE v3 on original vs DFS-reordered Hx
    across erasure rates for each code family.

    Saves per-family stats to msgpack files in stat_dir.
    Reorder time is excluded — only the decode call is timed.

    Inputs:
        code_names:    list of str or None (None = all families)
        data_dir:      str, directory containing classical H txt files
        stat_dir:      str, directory to write msgpack stat files
        erasure_rates: list of float
        n_trials:      int, erasure patterns per (version, rate) point
        random_seed:   int
    """
    # from ge_decoder import erasure_decode_sparse_v3

    if code_names is None:
        code_names = list(CODE_FAMILIES.keys())

    print("Reorder Benchmark")
    print("─────────────────")
    print(f"  code families : {code_names}")
    print(f"  erasure rates : {erasure_rates[0]} to {erasure_rates[-1]}")
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

        # Load and build
        try:
            H_cl = load_classical_H(filepath)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}\n")
            continue

        Hx, _      = build_hgp(H_cl)
        Hx_reorder, row_perm = dfs_reorder(Hx)

        assert sorted(row_perm) == list(range(Hx.shape[0])), "Not a valid permutation"

        N          = Hx.shape[1]

        print(f"  Hx shape     : {Hx.shape}   N={N}")
        print(f"  Reorder done : Hx_reorder shape {Hx_reorder.shape}")
        print()

        rng = np.random.default_rng(random_seed)

        # Storage: raw per-trial times in ms
        raw = {
            "original" : {str(r): [] for r in erasure_rates},
            "reordered": {str(r): [] for r in erasure_rates},
        }

        for rate in erasure_rates:
            n_erased    = int(N * rate)
            sx          = np.zeros(Hx.shape[0], dtype=int)

            orig_times  = []
            reord_times = []

            for trial in range(n_trials):
                # Same erasure pattern for both versions — fair comparison
                erased_bits = rng.choice(N, size=n_erased, replace=False)
                erasure_set = set(erased_bits.tolist())

                # Time original
                t0 = time.perf_counter()
                erasure_decode_sparse_v3(Hx, sx, erasure_set)
                orig_times.append((time.perf_counter() - t0) * 1000)

                # Time reordered
                t0 = time.perf_counter()
                erasure_decode_sparse_v3(Hx_reorder, sx, erasure_set)
                reord_times.append((time.perf_counter() - t0) * 1000)

            raw["original" ][str(rate)] = orig_times
            raw["reordered"][str(rate)] = reord_times

            orig_mean  = float(np.mean(orig_times))
            reord_mean = float(np.mean(reord_times))
            print(f"  rate={rate:.2f}  |ε|={n_erased:4d}  "
                  f"original={orig_mean:.3f}ms  "
                  f"reordered={reord_mean:.3f}ms  "
                  f"ratio={orig_mean/reord_mean:.2f}x")

        # Compute summary statistics
        summary = {}
        for version in ("original", "reordered"):
            means = [float(np.mean(raw[version][str(r)])) for r in erasure_rates]
            stds  = [float(np.std( raw[version][str(r)])) for r in erasure_rates]
            summary[version] = {"mean": means, "std": stds}

        # Save to msgpack
        stats = {
            "code_name"    : code_name,
            "label"        : label,
            "erasure_rates": erasure_rates,
            "n_trials"     : n_trials,
            "random_seed"  : random_seed,
            "original"     : summary["original"],
            "reordered"    : summary["reordered"],
        }
        save_stats(stats, code_name, stat_dir)
        print()

    print("Benchmark complete.")


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


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Sparse GE v3 on original vs DFS-reordered HGP codes.\n"
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
        "--data-dir", default=".",
        help="Directory containing classical H txt files (default: .)."
    )
    parser.add_argument(
        "--stat-dir", default=".",
        help="Directory for msgpack stat files (default: .)."
    )
    parser.add_argument(
        "--plot-file", default="reorder_benchmark.png",
        help="Output PNG filename (default: reorder_benchmark.png)."
    )
    parser.add_argument(
        "--trials", type=int, default=N_TRIALS,
        help=f"Trials per (version, erasure rate) point (default: {N_TRIALS})."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})."
    )
    args = parser.parse_args()

    # Default: run both if neither flag given
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
            code_names    = code_names,
            data_dir      = args.data_dir,
            stat_dir      = args.stat_dir,
            n_trials      = args.trials,
            random_seed   = args.seed,
        )

    if args.plot:
        plot_benchmark(
            code_names = code_names,
            stat_dir   = args.stat_dir,
            plot_file  = args.plot_file,
        )