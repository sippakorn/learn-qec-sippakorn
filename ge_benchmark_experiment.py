# ge_benchmark_experiment.py
#
# Benchmark and plot four GE decoder versions for sparse LDPC erasure decoding.
#
# Usage
# ─────
#   Run benchmark + plot (default):
#       python ge_benchmark_experiment.py
#
#   Run benchmark only (save stats, no plot):
#       python ge_benchmark_experiment.py --benchmark
#
#   Plot only (load existing stats file):
#       python ge_benchmark_experiment.py --plot
#
#   Custom file paths:
#       python ge_benchmark_experiment.py --benchmark --stats-file my_run.json
#       python ge_benchmark_experiment.py --plot --stats-file my_run.json --plot-file my_plot.png
#
#   From a notebook or another script:
#       from ge_benchmark_experiment import run_benchmark, plot_benchmark
#       run_benchmark(stats_file="ge_benchmark_stats.json")
#       plot_benchmark(stats_file="ge_benchmark_stats.json")
#
#   Inspect raw trial data:
#       import json
#       with open("ge_benchmark_stats.json") as f:
#           data = json.load(f)
#       # data["raw"]["Dense"]["150"] → list of 50 trial times (ms) at n=150

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from gaussian_elimination import erasure_decode_f2
from sparse_gaussian_elimination import erasure_decode_sparse
from sparse_gaussian_elimination_v2 import erasure_decode_sparse_v2
from sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3

# ── Experiment parameters ──────────────────────────────────────────────────
N_START      = 150
N_STOP       = 2550
N_STEP       = 200
N_RANGE      = list(range(N_START, N_STOP + 1, N_STEP))
ROW_WEIGHT   = 4
ERASURE_RATE = 0.4
N_TRIALS     = 50
RANDOM_SEED  = 42
STATS_FILE   = "ge_benchmark_stats.json"
PLOT_FILE    = "ge_benchmark.png"

VERSION_NAMES = ["Dense", "Sparse v1", "Sparse v2", "Sparse v3"]
COLORS = {
    "Dense"     : "#e74c3c",
    "Sparse v1" : "#e67e22",
    "Sparse v2" : "#2ecc71",
    "Sparse v3" : "#3498db",
}
MARKERS = {
    "Dense"     : "o",
    "Sparse v1" : "s",
    "Sparse v2" : "^",
    "Sparse v3" : "D",
}


# ── Helper — generate one random LDPC matrix ──────────────────────────────
def make_random_ldpc(n, row_weight, seed):
    """
    Generate a random LDPC parity-check matrix H of shape (m, n).
    Each row has exactly row_weight nonzeros chosen uniformly at random.

    Inputs:
        n:          int, number of variable nodes
        row_weight: int, number of ones per row
        seed:       int, random seed for reproducibility

    Returns:
        H: numpy 2D array, dtype=int, shape (m, n)
    """
    rng        = np.random.default_rng(seed)
    col_weight = 3
    m          = n * row_weight // col_weight
    H          = np.zeros((m, n), dtype=int)
    for i in range(m):
        cols       = rng.choice(n, size=row_weight, replace=False)
        H[i, cols] = 1
    return H


# ── Benchmark ──────────────────────────────────────────────────────────────
def run_benchmark(
    n_range      = N_RANGE,
    row_weight   = ROW_WEIGHT,
    erasure_rate = ERASURE_RATE,
    n_trials     = N_TRIALS,
    random_seed  = RANDOM_SEED,
    stats_file   = STATS_FILE,
):
    """
    Run timing benchmark for all four GE decoder versions.
    Saves raw per-trial timing data and summary statistics to a JSON file.

    Inputs:
        n_range:      list of int, code lengths to benchmark
        row_weight:   int, LDPC row weight
        erasure_rate: float, fraction of bits erased
        n_trials:     int, number of trials per (version, n) combination
        random_seed:  int, base random seed — trial k uses seed + k
        stats_file:   str, path to output JSON file

    Output JSON format:
        {
          "params":  { ... experiment parameters ... },
          "n_range": [...],
          "raw": {
              "Dense":     { "150": [t1, t2, ...], "350": [...], ... },
              "Sparse v1": { ... },
              ...
          },
          "summary": {
              "Dense":     { "mean": [...], "std": [...] },
              ...
          }
        }
    """
    # from ge_decoder import (
    #     erasure_decode_f2,
    #     erasure_decode_sparse,
    #     erasure_decode_sparse_v2,
    #     erasure_decode_sparse_v3,
    # )

    versions = {
        "Dense"     : erasure_decode_f2,
        "Sparse v1" : erasure_decode_sparse,
        "Sparse v2" : erasure_decode_sparse_v2,
        "Sparse v3" : erasure_decode_sparse_v3,
    }

    raw = {name: {str(n): [] for n in n_range} for name in versions}

    print("Benchmark parameters:")
    print(f"  n range      : {n_range[0]} to {n_range[-1]}"
          f" step {n_range[1] - n_range[0]}")
    print(f"  row weight   : {row_weight}")
    print(f"  erasure rate : {erasure_rate}")
    print(f"  trials       : {n_trials}")
    print(f"  random seed  : {random_seed}")
    print(f"  output file  : {stats_file}")
    print()

    for n in n_range:
        m           = n * row_weight // 3
        n_erased    = int(n * erasure_rate)
        erasure_set = set(range(n_erased))
        s           = np.zeros(m, dtype=int)

        print(f"  n={n:4d}  m={m:4d}  |erasure|={n_erased:3d}", end="  ")

        for trial in range(n_trials):
            H = make_random_ldpc(n, row_weight, seed=random_seed + trial)

            for name, fn in versions.items():
                start      = time.perf_counter()
                fn(H, s, erasure_set)
                elapsed_ms = (time.perf_counter() - start) * 1000
                raw[name][str(n)].append(elapsed_ms)

        for name in versions:
            mean = np.mean(raw[name][str(n)])
            print(f"{name}: {mean:.3f}ms", end="  ")
        print()

    # Compute summary statistics
    summary = {}
    for name in versions:
        means = [float(np.mean(raw[name][str(n)])) for n in n_range]
        stds  = [float(np.std( raw[name][str(n)])) for n in n_range]
        summary[name] = {"mean": means, "std": stds}

    output = {
        "params": {
            "n_range"     : n_range,
            "row_weight"  : row_weight,
            "erasure_rate": erasure_rate,
            "n_trials"    : n_trials,
            "random_seed" : random_seed,
        },
        "n_range": n_range,
        "raw"    : raw,
        "summary": summary,
    }

    with open(stats_file, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Stats saved to {stats_file}")
    return output


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_benchmark(
    stats_file = STATS_FILE,
    plot_file  = PLOT_FILE,
):
    """
    Load benchmark stats from JSON and produce a two-panel vertical plot:
      Top    — raw decoding time per call (ms, log scale) with std band
      Bottom — speedup ratio relative to dense version

    Can be run independently of run_benchmark() as long as stats_file exists.

    Inputs:
        stats_file: str, path to JSON file produced by run_benchmark()
        plot_file:  str, path to save the output PNG

    Raises:
        FileNotFoundError if stats_file does not exist
    """
    if not os.path.exists(stats_file):
        raise FileNotFoundError(
            f"Stats file '{stats_file}' not found. "
            f"Run run_benchmark() first to generate it."
        )

    with open(stats_file, "r") as f:
        data = json.load(f)

    n_range       = data["n_range"]
    summary       = data["summary"]
    params        = data["params"]
    version_names = [v for v in VERSION_NAMES if v in summary]

    means = {name: np.array(summary[name]["mean"]) for name in version_names}
    stds  = {name: np.array(summary[name]["std"])  for name in version_names}

    dense_mean = means["Dense"]
    speedups   = {name: dense_mean / means[name] for name in version_names}

    # ── Figure — two panels stacked vertically ────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    fig.suptitle(
        "GE Erasure Decoder — Dense vs Sparse Implementations\n"
        f"(3,4)-regular random LDPC  |  "
        f"erasure rate={params['erasure_rate']}  |  "
        f"row weight={params['row_weight']}  |  "
        f"{params['n_trials']} trials per point",
        fontsize=11,
    )

    # ── Top panel: raw time ───────────────────────────────────────────────
    ax = axes[0]
    for name in version_names:
        mean = means[name]
        std  = stds[name]
        ax.plot(
            n_range, mean,
            color=COLORS[name], marker=MARKERS[name],
            linewidth=1, markersize=4, label=name,
        )
        ax.fill_between(
            n_range,
            np.maximum(mean - std, 1e-6),
            mean + std,
            color=COLORS[name], alpha=0.12,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Code length n", fontsize=10)
    ax.set_ylabel("Time per call (ms, log scale)", fontsize=10)
    ax.set_title("Raw Decoding Time", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(n_range[::2])
    ax.tick_params(axis="x", rotation=30)

    # ── Bottom panel: speedup ─────────────────────────────────────────────
    ax = axes[1]
    for name in version_names:
        ax.plot(
            n_range, speedups[name],
            color=COLORS[name], marker=MARKERS[name],
            linewidth=1, markersize=4, label=name,
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Code length n", fontsize=10)
    ax.set_ylabel("Speedup vs Dense (×)", fontsize=10)
    ax.set_title("Speedup Relative to Dense", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(n_range[::2])
    ax.tick_params(axis="x", rotation=30)

    # Annotate final speedup values at largest n
    last_idx = len(n_range) - 1
    for name in version_names:
        final = speedups[name][last_idx]
        ax.annotate(
            f"{final:.1f}×",
            xy=(n_range[last_idx], final),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=COLORS[name],
        )

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {plot_file}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GE decoder versions and/or plot results.\n"
            "Default (no flags): runs benchmark then plots."
        )
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run the benchmark and save stats to JSON."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Load stats from JSON and produce the plot."
    )
    parser.add_argument(
        "--stats-file", default=STATS_FILE,
        help=f"Path to stats JSON file (default: {STATS_FILE})."
    )
    parser.add_argument(
        "--plot-file", default=PLOT_FILE,
        help=f"Path to output PNG file (default: {PLOT_FILE})."
    )
    args = parser.parse_args()

    # Default: run both if neither flag given
    if not args.benchmark and not args.plot:
        args.benchmark = True
        args.plot      = True

    if args.benchmark:
        run_benchmark(stats_file=args.stats_file)

    if args.plot:
        plot_benchmark(stats_file=args.stats_file, plot_file=args.plot_file)