# scaling_experiment.py
#
# Demonstrate that reordering (DFS, RCM) provides increasing GE speedup
# as HGP code size grows. Uses random (3,4)-regular classical seed codes
# to build HGP codes across a wide range of N (physical qubit count).
#
# Design:
#   - 9 code sizes: N ≈ 625 to 35000
#   - 10 random seed codes per size (k=0..9)
#   - Fixed erasure rate 0.30
#   - 50 trials per (code, strategy)
#   - Speedup ratio = mean_t_ge[none] / mean_t_ge[strategy]
#   - GE time averaged over GE-needed trials only
#
# Output:
#   stats_scaling_experiment.msgpack  — all raw data
#   scaling_experiment.png            — speedup ratio vs N plot
#
# Usage
# ─────
#   Run benchmark + plot (default):
#       python scaling_experiment.py
#
#   Benchmark only:
#       python scaling_experiment.py --benchmark
#
#   Plot only (requires stat file):
#       python scaling_experiment.py --plot
#
#   Custom parameters:
#       python scaling_experiment.py --trials 30 --seeds 5
#
#   Custom output paths:
#       python scaling_experiment.py --stat-file ./stats/scaling.msgpack
#                                    --plot-file ./plots/scaling.png
#
# From a notebook or another script:
#   from scaling_experiment import run_scaling_experiment, plot_scaling_experiment
#   run_scaling_experiment()
#   plot_scaling_experiment()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time
import os
import argparse
import msgpack
from sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3 

# Import reorder functions and decoders from existing scripts
from peeling_reorder_benchmark import (
    no_reorder,
    dfs_reorder,
    cm_reorder,
    peeling_decoder,
    build_hgp,
)
# from ge_decoder import erasure_decode_sparse_v3

# ── Experiment parameters ──────────────────────────────────────────────────

# Classical seed code lengths → HGP N values
# n=20  → N=625,   n=30  → N=1384,  n=40  → N=2500
# n=50  → N=3869,  n=60  → N=5625,  n=80  → N=10000
# n=100 → N=15625, n=120 → N=22500, n=150 → N=35044
CLASSICAL_SIZES = [20, 30, 40, 50, 60, 80, 100, 120, 150]
ROW_WEIGHT      = 4
COL_WEIGHT      = 3
ERASURE_RATE    = 0.35
N_TRIALS        = 10
N_SEEDS         = 5
RANDOM_SEED     = 42       # base seed — seed k uses RANDOM_SEED + k
STAT_FILE       = "stats_scaling_experiment.msgpack"
PLOT_FILE       = "scaling_experiment.png"

REORDER_STRATEGIES = {
    "none": no_reorder,
    "dfs" : dfs_reorder,
    "rcm" : cm_reorder,
}

STRATEGY_STYLE = {
    "dfs": {
        "color"    : "#27ae60",
        "linestyle": "--",
        "marker"   : "s",
        "label"    : "DFS",
    },
    "rcm": {
        "color"    : "#9b59b6",
        "linestyle": ":",
        "marker"   : "D",
        "label"    : "RCM",
    },
}


# ── Random LDPC generator ──────────────────────────────────────────────────
def make_random_ldpc(n, row_weight, seed):
    """
    Generate a random (row_weight, col_weight)-regular classical LDPC
    parity-check matrix H of shape (m, n) where m = n * row_weight // col_weight.
    Each row has exactly row_weight nonzeros chosen uniformly at random.

    Inputs:
        n:          int, number of variable nodes
        row_weight: int, number of ones per row
        seed:       int, random seed

    Returns:
        H: numpy 2D array, dtype=int, shape (m, n)
    """
    rng = np.random.default_rng(seed)
    m   = n * row_weight // COL_WEIGHT
    H   = np.zeros((m, n), dtype=int)
    for i in range(m):
        cols       = rng.choice(n, size=row_weight, replace=False)
        H[i, cols] = 1
    return H


def hgp_N(n):
    """Return physical qubit count N for HGP built from classical code of length n."""
    m = n * ROW_WEIGHT // COL_WEIGHT
    return n * n + m * m


# ── Stat file I/O ──────────────────────────────────────────────────────────
def save_stats(stats, stat_file):
    """Serialize stats dict to msgpack binary file."""
    os.makedirs(os.path.dirname(stat_file) or ".", exist_ok=True)
    with open(stat_file, "wb") as f:
        msgpack.pack(stats, f)
    print(f"Stats saved → {stat_file}")


def load_stats(stat_file):
    """Load stats dict from msgpack binary file. Returns None if not found."""
    if not os.path.exists(stat_file):
        return None
    with open(stat_file, "rb") as f:
        return msgpack.unpack(f)


# ── Benchmark ──────────────────────────────────────────────────────────────
def run_scaling_experiment(
    classical_sizes = CLASSICAL_SIZES,
    row_weight      = ROW_WEIGHT,
    erasure_rate    = ERASURE_RATE,
    n_trials        = N_TRIALS,
    n_seeds         = N_SEEDS,
    random_seed     = RANDOM_SEED,
    stat_file       = STAT_FILE,
):
    """
    Benchmark GE speedup from reordering across HGP code sizes.

    For each classical code size n and each random seed k:
        1. Generate random (3,4)-regular classical H
        2. Build HGP Hx
        3. For each strategy (none, dfs, rcm):
               Apply reorder ONCE — record t_reorder_ms
               Run 50 trials at erasure_rate=0.30:
                   peeling → if residual: Sparse GE v3
                   Record t_ge for GE-needed trials only
        4. Compute speedup_dfs = t_ge_none / t_ge_dfs
                  speedup_rcm = t_ge_none / t_ge_rcm

    Stat file schema:
    {
      "params": {
        "classical_sizes", "row_weight", "erasure_rate",
        "n_trials", "n_seeds", "random_seed"
      },
      "results": [
        {
          "n_classical"  : int,
          "N"            : int,
          "seed_k"       : int,
          "strategies"   : {
            "none" | "dfs" | "rcm": {
              "t_reorder_ms" : float,
              "t_ge"         : [float, ...],   GE-needed trials only
              "n_ge_trials"  : int,
              "n_peel_only"  : int,
              "t_ge_mean"    : float,
            }
          }
        }, ...
      ]
    }

    Inputs:
        classical_sizes: list of int, classical code lengths
        row_weight:      int
        erasure_rate:    float
        n_trials:        int, erasure patterns per (code, strategy)
        n_seeds:         int, random seed codes per size
        random_seed:     int, base seed — seed k uses random_seed + k*100
        stat_file:       str, output msgpack path
    """
    strategy_names = list(REORDER_STRATEGIES.keys())

    print("Scaling Experiment — Reorder Speedup vs N")
    print("──────────────────────────────────────────")
    print(f"  classical sizes : {classical_sizes}")
    print(f"  N values        : {[hgp_N(n) for n in classical_sizes]}")
    print(f"  row weight      : {row_weight}")
    print(f"  erasure rate    : {erasure_rate}")
    print(f"  trials          : {n_trials}")
    print(f"  seeds per size  : {n_seeds}")
    print(f"  base seed       : {random_seed}")
    print(f"  stat file       : {stat_file}")
    print()

    results = []

    for n_cl in classical_sizes:
        m_cl = n_cl * row_weight // COL_WEIGHT
        N    = n_cl * n_cl + m_cl * m_cl
        print(f"── n={n_cl:3d}  m={m_cl:3d}  N={N:6d} ──────────────────")

        for k in range(n_seeds):
            seed = random_seed + k * 100
            H_cl = make_random_ldpc(n_cl, row_weight, seed=seed)
            Hx, _  = build_hgp(H_cl)
            n_rows = Hx.shape[0]
            rng    = np.random.default_rng(seed)

            print(f"  seed k={k}  Hx={Hx.shape}", end="  ")

            # Apply and time each reordering strategy once
            reordered = {}
            for sname in strategy_names:
                fn = REORDER_STRATEGIES[sname]
                t0 = time.perf_counter()
                H_strat, _ = fn(Hx)
                t_ms = (time.perf_counter() - t0) * 1000
                reordered[sname] = {
                    "H"           : H_strat,
                    "t_reorder_ms": t_ms,
                }

            # Per-strategy raw storage
            raw = {
                sname: {
                    "t_ge"        : [],
                    "n_ge_trials" : 0,
                    "n_peel_only" : 0,
                }
                for sname in strategy_names
            }

            sx = np.zeros(n_rows, dtype=int)

            # Trial loop — same erasure pattern for all strategies
            n_erased = int(N * erasure_rate)
            for trial in range(n_trials):
                erased_bits = rng.choice(N, size=n_erased, replace=False)
                erasure_set = set(erased_bits.tolist())

                for sname in strategy_names:
                    H_strat = reordered[sname]["H"]
                    bucket  = raw[sname]

                    # Peeling — NOT timed (reordering-independent)
                    _, residual, res_syn = peeling_decoder(
                        H_strat, sx, erasure_set
                    )

                    if residual:
                        s_res = np.array(
                            [res_syn.get(i, 0) for i in range(n_rows)],
                            dtype=int
                        )
                        # GE fallback — timed in isolation
                        t0 = time.perf_counter()
                        erasure_decode_sparse_v3(H_strat, s_res, residual)
                        bucket["t_ge"].append(
                            (time.perf_counter() - t0) * 1000
                        )
                        bucket["n_ge_trials"] += 1
                    else:
                        bucket["n_peel_only"] += 1

            # Compute per-strategy means
            strategy_block = {}
            for sname in strategy_names:
                b        = raw[sname]
                t_ge_mean = float(np.mean(b["t_ge"])) if b["t_ge"] else 0.0
                strategy_block[sname] = {
                    "t_reorder_ms": float(reordered[sname]["t_reorder_ms"]),
                    "t_ge"        : [float(x) for x in b["t_ge"]],
                    "n_ge_trials" : b["n_ge_trials"],
                    "n_peel_only" : b["n_peel_only"],
                    "t_ge_mean"   : t_ge_mean,
                }

            # Compute speedup ratios for this seed
            t_none = strategy_block["none"]["t_ge_mean"]
            speedups = {}
            for sname in ["dfs", "rcm"]:
                t_s = strategy_block[sname]["t_ge_mean"]
                speedups[sname] = (
                    float(t_none / t_s) if t_s > 0 else 1.0
                )

            results.append({
                "n_classical": n_cl,
                "N"          : N,
                "seed_k"     : k,
                "strategies" : strategy_block,
                "speedups"   : speedups,
            })

            # Console summary
            t_none_ms = strategy_block["none"]["t_ge_mean"]
            n_ge      = strategy_block["none"]["n_ge_trials"]
            print(
                f"t_ge_none={t_none_ms:.2f}ms(n={n_ge})  "
                f"speedup_dfs={speedups['dfs']:.2f}x  "
                f"speedup_rcm={speedups['rcm']:.2f}x"
            )

        print()

    # Save all results
    output = {
        "params": {
            "classical_sizes": classical_sizes,
            "row_weight"     : row_weight,
            "erasure_rate"   : erasure_rate,
            "n_trials"       : n_trials,
            "n_seeds"        : n_seeds,
            "random_seed"    : random_seed,
        },
        "results": results,
    }
    save_stats(output, stat_file)
    return output


# ── Plot ───────────────────────────────────────────────────────────────────
def plot_scaling_experiment(
    stat_file = STAT_FILE,
    plot_file = PLOT_FILE,
):
    """
    Load scaling experiment stats and produce speedup ratio vs N plot.

    For each N, aggregates speedup across n_seeds random codes:
        mean speedup ± 1 std shown as line + shaded band

    X-axis: N (physical qubit count), log scale
    Y-axis: speedup ratio vs no-reorder (linear scale)
    Lines:  DFS speedup, RCM speedup
    Ref:    horizontal line at 1.0 (no improvement baseline)

    Also annotates t_reorder cost at selected N points for context.

    Inputs:
        stat_file: str, path to msgpack file from run_scaling_experiment
        plot_file: str, output PNG path

    Raises:
        FileNotFoundError if stat_file does not exist
    """
    if not os.path.exists(stat_file):
        raise FileNotFoundError(
            f"Stat file not found: '{stat_file}'. "
            f"Run --benchmark first."
        )

    with open(stat_file, "rb") as f:
        data = msgpack.unpack(f)

    params  = data["params"]
    results = data["results"]

    # Aggregate speedups per N
    from collections import defaultdict
    speedups_by_N = defaultdict(lambda: {"dfs": [], "rcm": []})
    t_reorder_by_N = defaultdict(lambda: {"dfs": [], "rcm": [], "none": []})
    N_values = []

    for entry in results:
        N    = entry["N"]
        sp   = entry["speedups"]
        strats = entry["strategies"]
        speedups_by_N[N]["dfs"].append(sp["dfs"])
        speedups_by_N[N]["rcm"].append(sp["rcm"])
        for sname in ["none", "dfs", "rcm"]:
            if sname in strats:
                t_reorder_by_N[N][sname].append(
                    strats[sname]["t_reorder_ms"]
                )

    N_sorted = sorted(speedups_by_N.keys())

    # Compute mean and std per N
    summary = {}
    for N in N_sorted:
        summary[N] = {}
        for sname in ["dfs", "rcm"]:
            vals = speedups_by_N[N][sname]
            summary[N][sname] = {
                "mean": float(np.mean(vals)),
                "std" : float(np.std(vals)),
            }

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(
        y=1.0, color="#bdc3c7", linestyle="-",
        linewidth=1, label="no improvement (1.0×)",
    )

    for sname in ["dfs", "rcm"]:
        style = STRATEGY_STYLE[sname]
        means = np.array([summary[N][sname]["mean"] for N in N_sorted])
        stds  = np.array([summary[N][sname]["std"]  for N in N_sorted])

        ax.plot(
            N_sorted, means,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1, markersize=4,
            label=f"{style['label']} speedup",
        )
        ax.fill_between(
            N_sorted,
            np.maximum(means - stds, 0),
            means + stds,
            color=style["color"], alpha=0.12,
        )

        # Annotate final speedup value at largest N
        ax.annotate(
            f"{means[-1]:.1f}×",
            xy=(N_sorted[-1], means[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9, color=style["color"],
        )

    # Annotate t_reorder for RCM at selected N points
    # Choose 3 evenly spaced points to avoid clutter
    annotate_indices = [0, len(N_sorted)//2, len(N_sorted)-1]
    for idx in annotate_indices:
        N = N_sorted[idx]
        t_rcm_vals = t_reorder_by_N[N]["rcm"]
        if t_rcm_vals:
            t_rcm_mean = float(np.mean(t_rcm_vals))
            rcm_mean   = summary[N]["rcm"]["mean"]
            ax.annotate(
                f"t_rcm={t_rcm_mean:.0f}ms",
                xy=(N, rcm_mean),
                xytext=(0, -14),
                textcoords="offset points",
                fontsize=6, color=STRATEGY_STYLE["rcm"]["color"],
                ha="center",
                arrowprops=dict(arrowstyle="-", color="gray",
                                lw=0.5),
            )

    ax.set_xscale("log")
    ax.set_xlabel("N (physical qubit count, log scale)", fontsize=11)
    ax.set_ylabel("Speedup vs no-reorder (×)", fontsize=11)
    ax.set_title(
        "GE Speedup from Reordering vs HGP Code Size\n"
        f"Random (3,4)-regular HGP codes  |  "
        f"erasure rate={params['erasure_rate']}  |  "
        f"{params['n_seeds']} seeds × {params['n_trials']} trials per point  |  "
        f"GE-needed trials only",
        fontsize=10,
    )

    # Mark N values from Connolly et al. families
    connolly_N = {625: "[[625,25]]", 1600: "[[1600,64]]", 2025: "[[2025,81]]"}
    for N_ref, lbl in connolly_N.items():
        if N_ref in N_sorted:
            ax.axvline(
                x=N_ref, color="#e74c3c",
                linestyle="--", linewidth=0.7, alpha=0.5,
            )
            ax.text(
                N_ref, ax.get_ylim()[0],
                f" {lbl}",
                fontsize=6, color="#e74c3c",
                rotation=90, va="bottom",
            )

    # X-axis ticks at the actual N values
    ax.set_xticks(N_sorted)
    ax.set_xticklabels(
        [f"{N:,}" for N in N_sorted],
        rotation=30, ha="right", fontsize=8,
    )
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.15)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {plot_file}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark GE speedup from reordering vs HGP code size.\n"
            "Default (no flags): runs benchmark then plots."
        )
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run scaling experiment and save stats to msgpack file."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Load stats and produce speedup plot."
    )
    parser.add_argument(
        "--stat-file", default=STAT_FILE,
        help=f"Path to msgpack stat file (default: {STAT_FILE})."
    )
    parser.add_argument(
        "--plot-file", default=PLOT_FILE,
        help=f"Path to output PNG (default: {PLOT_FILE})."
    )
    parser.add_argument(
        "--trials", type=int, default=N_TRIALS,
        help=f"Trials per (code, strategy) (default: {N_TRIALS})."
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help=f"Random seed codes per size (default: {N_SEEDS})."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Base random seed (default: {RANDOM_SEED})."
    )
    args = parser.parse_args()

    if not args.benchmark and not args.plot:
        args.benchmark = True
        args.plot      = True

    if args.benchmark:
        run_scaling_experiment(
            n_trials    = args.trials,
            n_seeds     = args.seeds,
            random_seed = args.seed,
            stat_file   = args.stat_file,
        )

    if args.plot:
        plot_scaling_experiment(
            stat_file = args.stat_file,
            plot_file = args.plot_file,
        )