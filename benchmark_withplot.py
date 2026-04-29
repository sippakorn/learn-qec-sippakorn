import numpy as np
import matplotlib.pyplot as plt
import time
from gaussian_elimination import erasure_decode_f2
from sparse_gaussian_elimination import erasure_decode_sparse
from sparse_gaussian_elimination_v2 import erasure_decode_sparse_v2
from sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3

# ── Experiment parameters ──────────────────────────────────────────────────
N_START    = 150
N_STOP     = 2550
N_STEP     = 200
N_RANGE    = list(range(N_START, N_STOP + 1, N_STEP))
ROW_WEIGHT = 4          # (3,4)-regular LDPC
ERASURE_RATE = 0.4      # 40% of n erased
N_TRIALS   = 50         # trials per (version, n) combination
RANDOM_SEED = 42

VERSIONS = {
    "Dense"     : erasure_decode_f2,
    "Sparse v1" : erasure_decode_sparse,
    "Sparse v2" : erasure_decode_sparse_v2,
    "Sparse v3" : erasure_decode_sparse_v3,
}

# ── Helper — generate one random (3,4)-regular-like LDPC matrix ───────────
def make_random_ldpc(n, row_weight, seed):
    """
    Generate a random LDPC parity-check matrix H of shape (m, n)
    where m = n * row_weight // col_weight with col_weight = 3.
    Each row has exactly row_weight nonzeros chosen uniformly at random.

    Inputs:
        n:          int, number of variable nodes
        row_weight: int, number of ones per row
        seed:       int, random seed for reproducibility

    Returns:
        H: numpy 2D array, dtype=int, shape (m, n)
    """
    rng = np.random.default_rng(seed)
    col_weight = 3
    m = n * row_weight // col_weight
    H = np.zeros((m, n), dtype=int)
    for i in range(m):
        cols = rng.choice(n, size=row_weight, replace=False)
        H[i, cols] = 1
    return H


# ── Main benchmark loop ────────────────────────────────────────────────────
# results[version_name][n] = list of per-trial times in milliseconds
results = {name: {n: [] for n in N_RANGE} for name in VERSIONS}

print(f"Running benchmark: n = {N_RANGE[0]} to {N_RANGE[-1]}, "
      f"{N_TRIALS} trials each, erasure rate = {ERASURE_RATE}")
print()

for n_idx, n in enumerate(N_RANGE):
    m              = n * ROW_WEIGHT // 3
    n_erased       = int(n * ERASURE_RATE)
    erasure_set    = set(range(n_erased))   # erase first 40% — deterministic
    s              = np.zeros(m, dtype=int) # all-zero syndrome (zero codeword)

    print(f"  n={n:4d}  m={m:4d}  |erasure|={n_erased:3d}", end="  ")

    for trial in range(N_TRIALS):
        # Fresh random H per trial — different seed each time
        H = make_random_ldpc(n, ROW_WEIGHT, seed=RANDOM_SEED + trial)

        for name, fn in VERSIONS.items():
            start = time.perf_counter()
            fn(H, s, erasure_set)
            elapsed_ms = (time.perf_counter() - start) * 1000
            results[name][n].append(elapsed_ms)

    # Print mean times for this n as a sanity check
    for name in VERSIONS:
        mean = np.mean(results[name][n])
        print(f"{name}: {mean:.3f}ms", end="  ")
    print()

print()
print("Benchmark complete.")


# ── Compute statistics ─────────────────────────────────────────────────────
stats = {}
for name in VERSIONS:
    means = np.array([np.mean(results[name][n]) for n in N_RANGE])
    stds  = np.array([np.std( results[name][n]) for n in N_RANGE])
    stats[name] = {"mean": means, "std": stds}

# Speedup relative to dense
dense_mean = stats["Dense"]["mean"]
speedups = {}
speedup_stds = {}
for name in VERSIONS:
    ratio     = dense_mean / stats[name]["mean"]
    speedups[name] = ratio


# ── Plot 1 — Raw timing (log scale) ───────────────────────────────────────
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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "GE Erasure Decoder — Dense vs Sparse Implementations\n"
    f"(3,4)-regular random LDPC, erasure rate={ERASURE_RATE}, "
    f"{N_TRIALS} trials per point",
    fontsize=12
)

# ── Left: raw time ─────────────────────────────────────────────────────────
ax = axes[0]
for name in VERSIONS:
    mean = stats[name]["mean"]
    std  = stats[name]["std"]
    ax.plot(N_RANGE, mean,
            color=COLORS[name], marker=MARKERS[name],
            linewidth=2, markersize=5, label=name)
    ax.fill_between(N_RANGE,
                    mean - std, mean + std,
                    color=COLORS[name], alpha=0.15)

ax.set_yscale("log")
ax.set_xlabel("Code length n", fontsize=11)
ax.set_ylabel("Time per call (ms, log scale)", fontsize=11)
ax.set_title("Raw Decoding Time", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.set_xticks(N_RANGE[::2])
ax.tick_params(axis='x', rotation=30)

# ── Right: speedup ratio ───────────────────────────────────────────────────
ax = axes[1]
for name in VERSIONS:
    ax.plot(N_RANGE, speedups[name],
            color=COLORS[name], marker=MARKERS[name],
            linewidth=2, markersize=5, label=name)

ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlabel("Code length n", fontsize=11)
ax.set_ylabel("Speedup vs Dense (×)", fontsize=11)
ax.set_title("Speedup Relative to Dense", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xticks(N_RANGE[::2])
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("ge_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to ge_benchmark.png")