# peeling_reorder_benchmark.py
#
# Benchmark the full decoding pipeline on DFS-reordered HGP codes:
#   1. dfs_reorder(Hx)                          — one-time cost per code
#   2. peeling_decoder(Hx_reordered, ...)        — per trial
#   3. sparse_ge_v3(Hx_reordered, ...) if needed — per trial, residual only
#
# Raw stats saved per code family as msgpack binary files.
# Plot shows peeling time, GE time, and peeling success rate per erasure rate.
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
        "color_peel" : "#e74c3c",
        "color_ge"   : "#e67e22",
    },
    "n1225" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt",
        "label" : "[[1225,65]]",
        "color_peel" : "#3498db",
        "color_ge"   : "#1abc9c",
    },
    "n1600" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt",
        "label" : "[[1600,64]]",
        "color_peel" : "#9b59b6",
        "color_ge"   : "#8e44ad",
    },
    "n2025" : {
        "file"  : "PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt",
        "label" : "[[2025,81]]",
        "color_peel" : "#27ae60",
        "color_ge"   : "#f39c12",
    },
}


# ── DFS reorder ────────────────────────────────────────────────────────────
def dfs_reorder(H):
    """
    Reorder rows of H using DFS post-order traversal of the Tanner graph.
    Only rows are permuted — columns stay in original order so that
    erasure_index_set indices remain valid after reordering.

    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)

    Returns:
        H_reordered:   numpy 2D array, dtype=int, shape (m, n)
        cons_ordering: list of int, row permutation applied
    """
    num_rows, num_cols = H.shape
    G = nx.Graph()

    G.add_nodes_from([('v', j) for j in range(num_cols)])
    G.add_nodes_from([('c', i) for i in range(num_rows)])

    # Use np.where — avoids O(m*n) double loop
    rows_idx, cols_idx = np.where(H == 1)
    for i, j in zip(rows_idx, cols_idx):
        G.add_edge(('c', i), ('v', j))

    dfs_ordering  = list(nx.dfs_postorder_nodes(G))
    cons_ordering = [node[1] for node in dfs_ordering if node[0] == 'c']

    # Rows only — columns unchanged so erasure indices remain valid
    H_reordered = H[cons_ordering, :]
    return H_reordered, cons_ordering


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

    # Build adjacency structures
    check_to_vars = {}
    var_to_checks = {j: set() for j in erasure_index_set}

    for i in range(H.shape[0]):
        neighbours = set(
            j for j in np.where(H[i] == 1)[0]
            if j in erasure_index_set
        )
        if neighbours:
            check_to_vars[i] = neighbours
            for j in neighbours:
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

        for neighbour_check in var_to_checks[var]:
            if neighbour_check == check:
                continue
            if neighbour_check not in check_to_vars:
                continue

            syndrome[neighbour_check] ^= var_value
            check_to_vars[neighbour_check].discard(var)

            if len(check_to_vars[neighbour_check]) == 1:
                dangling.add(neighbour_check)
            elif len(check_to_vars[neighbour_check]) == 0:
                del check_to_vars[neighbour_check]
                del syndrome[neighbour_check]

        del var_to_checks[var]
        del check_to_vars[check]
        del syndrome[check]

    residual_erasure = set(var_to_checks.keys())
    return solution, residual_erasure, syndrome


# ── Stat file I/O ──────────────────────────────────────────────────────────
def stat_filename(code_name, stat_dir):
    return os.path.join(stat_dir, f"stats_peeling_reorder_{code_name}.msgpack")


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
    code_names    = None,
    data_dir      = ".",
    stat_dir      = ".",
    erasure_rates = ERASURE_RATES,
    n_trials      = N_TRIALS,
    random_seed   = RANDOM_SEED,
):
    """
    Benchmark full decoding pipeline on DFS-reordered HGP codes.

    Pipeline per trial:
        peeling_decoder(Hx_reordered, ...)
        → if residual: sparse_ge_v3(Hx_reordered, s_residual, residual_erasure)

    Timing breakdown recorded separately:
        t_reorder : one-time DFS reorder cost (ms)
        t_peel    : peeling time per trial (all trials)
        t_ge      : GE time per trial (only trials where GE was needed)

    Raw data also stored for original Hx pipeline (no reorder) for reference.

    Inputs:
        code_names:    list of str or None (None = all families)
        data_dir:      str, directory containing classical H txt files
        stat_dir:      str, directory to write msgpack stat files
        erasure_rates: list of float
        n_trials:      int, trials per (code, erasure rate) point
        random_seed:   int
    """
    # from ge_decoder import erasure_decode_sparse_v3

    if code_names is None:
        code_names = list(CODE_FAMILIES.keys())

    print("Peeling + Reorder Benchmark")
    print("───────────────────────────")
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

        try:
            H_cl = load_classical_H(filepath)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}\n")
            continue

        Hx, _ = build_hgp(H_cl)
        N      = Hx.shape[1]
        print(f"  Hx shape : {Hx.shape}   N={N}")

        # ── One-time reorder ──────────────────────────────────────────────
        t0 = time.perf_counter()
        Hx_reordered, _ = dfs_reorder(Hx)
        t_reorder_ms = (time.perf_counter() - t0) * 1000
        print(f"  t_reorder : {t_reorder_ms:.2f} ms")
        print()

        rng = np.random.default_rng(random_seed)

        # Raw storage per erasure rate
        # Each key maps to a list of per-trial times in ms
        per_rate_reordered = {}
        per_rate_original  = {}

        for rate in erasure_rates:
            n_erased = int(N * rate)
            sx       = np.zeros(Hx.shape[0], dtype=int)

            # Per-trial raw times
            peel_times_reord  = []
            ge_times_reord    = []
            peel_times_orig   = []
            ge_times_orig     = []
            n_ge_reord        = 0
            n_ge_orig         = 0
            n_peel_only_reord = 0
            n_peel_only_orig  = 0

            for trial in range(n_trials):
                erased_bits = rng.choice(N, size=n_erased, replace=False)
                erasure_set = set(erased_bits.tolist())

                # ── Reordered pipeline ────────────────────────────────────
                t0 = time.perf_counter()
                sol_p, residual, res_syn = peeling_decoder(
                    Hx_reordered, sx, erasure_set
                )
                peel_times_reord.append((time.perf_counter() - t0) * 1000)

                if residual:
                    s_res = np.array(
                        [res_syn.get(i, 0) for i in range(Hx.shape[0])],
                        dtype=int
                    )
                    t0 = time.perf_counter()
                    erasure_decode_sparse_v3(Hx_reordered, s_res, residual)
                    ge_times_reord.append((time.perf_counter() - t0) * 1000)
                    n_ge_reord += 1
                else:
                    n_peel_only_reord += 1

                # ── Original pipeline (no reorder) ────────────────────────
                t0 = time.perf_counter()
                sol_p, residual, res_syn = peeling_decoder(
                    Hx, sx, erasure_set
                )
                peel_times_orig.append((time.perf_counter() - t0) * 1000)

                if residual:
                    s_res = np.array(
                        [res_syn.get(i, 0) for i in range(Hx.shape[0])],
                        dtype=int
                    )
                    t0 = time.perf_counter()
                    erasure_decode_sparse_v3(Hx, s_res, residual)
                    ge_times_orig.append((time.perf_counter() - t0) * 1000)
                    n_ge_orig += 1
                else:
                    n_peel_only_orig += 1

            per_rate_reordered[str(rate)] = {
                "t_peel"        : peel_times_reord,
                "t_ge"          : ge_times_reord,
                "n_ge_trials"   : n_ge_reord,
                "n_peel_only"   : n_peel_only_reord,
            }
            per_rate_original[str(rate)] = {
                "t_peel"        : peel_times_orig,
                "t_ge"          : ge_times_orig,
                "n_ge_trials"   : n_ge_orig,
                "n_peel_only"   : n_peel_only_orig,
            }

            # Console summary
            peel_mean_r = float(np.mean(peel_times_reord))
            ge_mean_r   = float(np.mean(ge_times_reord)) if ge_times_reord else 0.0
            peel_pct_r  = n_peel_only_reord / n_trials * 100
            print(f"  rate={rate:.2f}  |ε|={n_erased:4d}  "
                  f"peel={peel_mean_r:.3f}ms  "
                  f"ge={ge_mean_r:.3f}ms (n={n_ge_reord})  "
                  f"peel_success={peel_pct_r:.0f}%")

        # ── Compute summary statistics ─────────────────────────────────────
        def summarise(per_rate, key, rates):
            means, stds = [], []
            for r in rates:
                vals = per_rate[str(r)][key]
                means.append(float(np.mean(vals)) if vals else 0.0)
                stds.append( float(np.std(vals))  if vals else 0.0)
            return {"mean": means, "std": stds}

        stats = {
            "code_name"    : code_name,
            "label"        : label,
            "erasure_rates": erasure_rates,
            "n_trials"     : n_trials,
            "random_seed"  : random_seed,
            "t_reorder_ms" : float(t_reorder_ms),
            "reordered": {
                "t_peel"       : summarise(per_rate_reordered, "t_peel",  erasure_rates),
                "t_ge"         : summarise(per_rate_reordered, "t_ge",    erasure_rates),
                "n_ge_trials"  : [per_rate_reordered[str(r)]["n_ge_trials"]
                                  for r in erasure_rates],
                "n_peel_only"  : [per_rate_reordered[str(r)]["n_peel_only"]
                                  for r in erasure_rates],
                "raw"          : per_rate_reordered,
            },
            "original": {
                "t_peel"       : summarise(per_rate_original, "t_peel",   erasure_rates),
                "t_ge"         : summarise(per_rate_original, "t_ge",     erasure_rates),
                "n_ge_trials"  : [per_rate_original[str(r)]["n_ge_trials"]
                                  for r in erasure_rates],
                "n_peel_only"  : [per_rate_original[str(r)]["n_peel_only"]
                                  for r in erasure_rates],
                "raw"          : per_rate_original,
            },
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
    Load per-family msgpack stat files and produce a 2×2 subplot figure.

    Each subplot (one per code family) shows:
        Left Y-axis  — peeling time (ms) and GE time (ms), linear scale
        Right Y-axis — peeling success rate (%)
        Title        — code label + t_reorder one-time cost

    Only the reordered pipeline is plotted.
    Raw data for both pipelines is preserved in the msgpack files.

    Inputs:
        code_names: list of str or None (None = all families with stat files)
        stat_dir:   str, directory containing msgpack stat files
        plot_file:  str, output PNG path
    """
    if code_names is None:
        code_names = list(CODE_FAMILIES.keys())

    loaded = {}
    for code_name in code_names:
        stats = load_stats(code_name, stat_dir)
        if stats is None:
            print(f"  SKIP {code_name}: stat file not found")
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
        "Peeling + Sparse GE v3 on DFS-Reordered HGP Codes\n"
        "Peeling time = all trials   |   GE time = GE-needed trials only   |"
        "   t_reorder = one-time cost",
        fontsize=11,
    )

    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax_idx, (code_name, stats) in enumerate(loaded.items()):
        ax      = axes_flat[ax_idx]
        ax_twin = ax.twinx()

        meta          = CODE_FAMILIES[code_name]
        color_peel    = meta["color_peel"]
        color_ge      = meta["color_ge"]
        color_success = "#7f8c8d"

        erasure_rates = stats["erasure_rates"]
        reord         = stats["reordered"]
        n_trials      = stats["n_trials"]
        t_reorder_ms  = stats["t_reorder_ms"]

        peel_mean = np.array(reord["t_peel"]["mean"])
        peel_std  = np.array(reord["t_peel"]["std"])
        ge_mean   = np.array(reord["t_ge"]["mean"])
        ge_std    = np.array(reord["t_ge"]["std"])
        n_ge      = np.array(reord["n_ge_trials"])
        n_peel    = np.array(reord["n_peel_only"])
        success_pct = n_peel / n_trials * 100

        # ── Peeling time ──────────────────────────────────────────────────
        ax.plot(erasure_rates, peel_mean,
                color=color_peel, linewidth=1, marker="o", markersize=3,
                label="Peeling time")
        ax.fill_between(erasure_rates,
                        np.maximum(peel_mean - peel_std, 0),
                        peel_mean + peel_std,
                        color=color_peel, alpha=0.12)

        # ── GE time (only where GE was needed) ────────────────────────────
        # Mask points where no GE was run (n_ge == 0)
        ge_mask  = n_ge > 0
        ge_rates = [erasure_rates[i] for i in range(len(erasure_rates))
                    if ge_mask[i]]
        ge_vals  = ge_mean[ge_mask]
        ge_err   = ge_std[ge_mask]

        if ge_rates:
            ax.plot(ge_rates, ge_vals,
                    color=color_ge, linewidth=1, marker="s", markersize=3,
                    linestyle="--", label=f"GE time (GE-needed trials only)")
            ax.fill_between(ge_rates,
                            np.maximum(ge_vals - ge_err, 0),
                            ge_vals + ge_err,
                            color=color_ge, alpha=0.12)

        # ── Peeling success rate (right axis) ─────────────────────────────
        ax_twin.plot(erasure_rates, success_pct,
                     color=color_success, linewidth=1,
                     marker="^", markersize=3, linestyle=":",
                     label="Peeling success %")
        ax_twin.set_ylabel("Peeling success rate (%)", fontsize=9,
                           color=color_success)
        ax_twin.tick_params(axis="y", labelcolor=color_success)
        ax_twin.set_ylim(-5, 105)
        ax_twin.yaxis.set_major_formatter(mticker.PercentFormatter())

        # ── Axes formatting ───────────────────────────────────────────────
        ax.set_xlabel("Erasure rate", fontsize=9)
        ax.set_ylabel("Avg time per call (ms)", fontsize=9)
        ax.set_title(
            f"{stats['label']}   "
            f"t_reorder = {t_reorder_ms:.1f} ms (one-time)",
            fontsize=10,
        )
        ax.set_xticks(erasure_rates)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        # ── n_ge annotation per point ─────────────────────────────────────
        for i, (r, n) in enumerate(zip(erasure_rates, n_ge)):
            if n > 0:
                ax.annotate(
                    f"n={n}",
                    xy=(r, ge_mean[i]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    fontsize=6,
                    color=color_ge,
                    ha="center",
                )

        # ── Combined legend ───────────────────────────────────────────────
        lines_left,  labels_left  = ax.get_legend_handles_labels()
        lines_right, labels_right = ax_twin.get_legend_handles_labels()
        ax.legend(
            lines_left + lines_right,
            labels_left + labels_right,
            fontsize=8, loc="upper left",
        )

    # Hide unused subplots
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
            "Benchmark peeling + Sparse GE v3 on DFS-reordered HGP codes.\n"
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
            code_names  = code_names,
            data_dir    = args.data_dir,
            stat_dir    = args.stat_dir,
            n_trials    = args.trials,
            random_seed = args.seed,
        )

    if args.plot:
        plot_benchmark(
            code_names = code_names,
            stat_dir   = args.stat_dir,
            plot_file  = args.plot_file,
        )