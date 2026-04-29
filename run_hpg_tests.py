# run_hgp_tests.py
#
# Load HGP code families from Connolly et al. txt files and verify
# that all four GE decoder versions produce correct, consistent results.
#
# Usage
# ─────
#   Run all available code families:
#       python run_hgp_tests.py
#
#   Run one specific family:
#       python run_hgp_tests.py --code n625
#
#   Custom data directory:
#       python run_hgp_tests.py --data-dir ./codes/
#
#   List available families:
#       python run_hgp_tests.py --list
#
#   Custom erasure rate and trial count:
#       python run_hgp_tests.py --erasure-rate 0.2 --trials 50
#
# Required data files (download from Connolly et al. repo)
# ─────────────────────────────────────────────────────────
#   Place these txt files in the same directory as this script,
#   or pass their location via --data-dir:
#
#   PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt
#   PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt
#   PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt
#   PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt
#
#   Source: https://github.com/Nicholas-Connolly/Pruned-Peeling-and-VH-Decoder
#
# From a notebook or another script:
#   from run_hgp_tests import load_classical_H, build_hgp, run_hgp_tests
#   H_cl    = load_classical_H("PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt")
#   Hx, Hz  = build_hgp(H_cl)
#   results = run_hgp_tests(Hx, Hz, label="[[625,25]]")

import numpy as np
import os
import argparse

from gaussian_elimination import erasure_decode_f2
from sparse_gaussian_elimination import erasure_decode_sparse
from sparse_gaussian_elimination_v2 import erasure_decode_sparse_v2
from sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3

# ── Code family registry ───────────────────────────────────────────────────
# Maps short name -> (filename, quantum parameters [[N, K]])
CODE_FAMILIES = {
    "n625"  : (
        "PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt",
        {"N": 625,  "K": 25},
    ),
    "n1225" : (
        "PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt",
        {"N": 1225, "K": 65},
    ),
    "n1600" : (
        "PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt",
        {"N": 1600, "K": 64},
    ),
    "n2025" : (
        "PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt",
        {"N": 2025, "K": 81},
    ),
}

# ── Default experiment parameters ─────────────────────────────────────────
ERASURE_RATE    = 0.3   # 30% — below threshold for these codes
N_DECODE_TRIALS = 20
RANDOM_SEED     = 42


# ── File loading ───────────────────────────────────────────────────────────
def load_classical_H(filepath):
    """
    Load a classical parity-check matrix H from a Connolly et al. txt file.

    File format:
        Line 1:         header — two integers "m n" (rows, columns)
        Lines 2 to m+1: one row per line — space-separated nonzero column
                        indices (sparse adjacency list, not full binary row)

    Example (15 rows, 20 columns):
        15 20
        0 9 5 15
        0 16 10 6
        0 7 11 17 19
        16 1 11 5
        ...

    Inputs:
        filepath: str, path to the txt file

    Returns:
        H: numpy 2D array, dtype=int, shape (m, n)

    Raises:
        FileNotFoundError if filepath does not exist
        ValueError if header is malformed, column indices out of range,
                   or number of data lines does not match header
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: '{filepath}'\n"
            f"Download from: https://github.com/Nicholas-Connolly/"
            f"Pruned-Peeling-and-VH-Decoder"
        )

    with open(filepath, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"File is empty: '{filepath}'")

    # Parse header
    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(
            f"Expected header 'm n' on line 1, got: '{lines[0]}'"
        )
    m, n = int(header[0]), int(header[1])

    # Parse data lines
    data_lines = lines[1:]
    if len(data_lines) != m:
        raise ValueError(
            f"Header says m={m} rows but found {len(data_lines)} "
            f"data lines in '{filepath}'"
        )

    H = np.zeros((m, n), dtype=int)
    for row_idx, line in enumerate(data_lines):
        try:
            col_indices = [int(x) for x in line.split()]
        except ValueError:
            raise ValueError(
                f"Non-integer entry at data line {row_idx + 1}: '{line}'"
            )
        for col in col_indices:
            if col < 0 or col >= n:
                raise ValueError(
                    f"Column index {col} out of range [0, {n-1}] "
                    f"at data line {row_idx + 1}: '{line}'"
                )
            H[row_idx, col] = 1

    return H


# ── HGP construction ───────────────────────────────────────────────────────
def build_hgp(H_cl):
    """
    Build a hypergraph product (HGP) CSS code from a classical
    parity-check matrix H_cl of shape (m, n).

    Construction (Tillich-Zemor 2014):
        Hx = [ H ⊗ I_n  |  I_m ⊗ H^T ]   shape: (m*n, n²+m²)
        Hz = [ I_n ⊗ H  |  H^T ⊗ I_m ]   shape: (n*m, n²+m²)

    CSS orthogonality: Hx @ Hz.T = 0  over F2.

    Quantum code parameters:
        N = n² + m²   physical qubits
        K = k²        logical qubits   (k = n - rank(H_cl))
        D = d         distance         (d = distance of classical code)

    Inputs:
        H_cl: numpy 2D array, dtype=int, shape (m, n)

    Returns:
        Hx: numpy 2D array, dtype=int, shape (m*n, N)
        Hz: numpy 2D array, dtype=int, shape (n*m, N)
    """
    m, n = H_cl.shape
    Im   = np.eye(m, dtype=int)
    In   = np.eye(n, dtype=int)

    Hx = np.hstack([np.kron(H_cl, In),  np.kron(Im, H_cl.T)])
    Hz = np.hstack([np.kron(In, H_cl),  np.kron(H_cl.T, Im)])

    return Hx, Hz


def verify_css_orthogonality(Hx, Hz):
    """
    Check CSS orthogonality condition: Hx @ Hz.T = 0 over F2.

    Inputs:
        Hx: numpy 2D array, dtype=int
        Hz: numpy 2D array, dtype=int

    Returns:
        bool — True if condition holds
    """
    return bool(np.all((Hx @ Hz.T) % 2 == 0))


# ── Agreement check ────────────────────────────────────────────────────────
def _check_agreement(trial_results, Hx, sx, verbose):
    """
    Determine whether all four decoder versions agree on a single trial.

    Agreement rules (in order of severity):
        1. Consistency verdict (ok) must match across all versions.
        2. If consistent, every solution must satisfy Hx @ sol = sx over F2.
        3. If consistent, solution vectors must be identical across versions.
        4. free_cols differences are acceptable when solutions are identical
           — different pivot orderings can produce different free variable
           sets while arriving at the same solution vector.

    Inputs:
        trial_results: dict  name -> {"sol", "ok", "free"}
        Hx:            numpy 2D array
        sx:            numpy 1D array, syndrome
        verbose:       bool, print detail on failure

    Returns:
        trial_agree: bool
        failure_reason: str or None
    """
    ref  = trial_results["Dense"]
    names = ["Sparse v1", "Sparse v2", "Sparse v3"]

    for name in names:
        r = trial_results[name]

        # Rule 1 — consistency must agree
        if r["ok"] != ref["ok"]:
            if verbose:
                print(f"    FAIL [{name}]: consistency mismatch  "
                      f"Dense={ref['ok']}  {name}={r['ok']}")
            return False, "consistency mismatch"

        if not ref["ok"]:
            continue   # both failed — nothing more to check

        # Rule 2 — solution must satisfy syndrome
        for n2, res in [(name, r), ("Dense", ref)]:
            residual = (Hx @ res["sol"]) % 2
            if not np.array_equal(residual, sx):
                if verbose:
                    bad = np.where(residual != sx)[0]
                    print(f"    FAIL [{n2}]: solution does not satisfy syndrome  "
                          f"residual nonzeros at {bad[:5].tolist()}")
                return False, f"{n2} solution fails syndrome check"

        # Rule 3 — solution vectors must be identical
        if not np.array_equal(r["sol"], ref["sol"]):
            diff = np.where(r["sol"] != ref["sol"])[0]
            if verbose:
                print(f"    FAIL [{name}]: solutions differ at "
                      f"{len(diff)} positions  "
                      f"first diffs: {diff[:5].tolist()}")
            return False, "solution mismatch"

        # Rule 4 — free_cols mismatch with identical solutions is acceptable
        if r["free"] != ref["free"] and verbose:
            print(f"    INFO [{name}]: free_cols differ but solutions match "
                  f"— acceptable (different pivot ordering)")

    return True, None


# ── Main test function ─────────────────────────────────────────────────────
def run_hgp_tests(
    Hx,
    Hz,
    label        = "",
    erasure_rate = ERASURE_RATE,
    n_trials     = N_DECODE_TRIALS,
    random_seed  = RANDOM_SEED,
    verbose      = True,
):
    """
    Run decoding correctness tests on an HGP CSS code.

    For each trial:
        1. Sample a random erasure pattern at erasure_rate
        2. Use zero syndrome (all-zero codeword, zero error)
        3. Run all four GE decoders on Hx (X-check decoding)
        4. Verify agreement using the three-rule check in _check_agreement

    Correctness criteria:
        - All versions must agree on whether the syndrome is consistent
        - All solutions must satisfy Hx @ sol = sx over F2
        - All solution vectors must be identical
        - free_cols differences with identical solutions are acceptable

    Inputs:
        Hx:           numpy 2D array, dtype=int — X parity-check matrix
        Hz:           numpy 2D array, dtype=int — Z parity-check matrix
                      (used only for CSS orthogonality check, not decoding)
        label:        str, display name for this code
        erasure_rate: float, fraction of qubits erased per trial
        n_trials:     int, number of random erasure patterns to test
        random_seed:  int, reproducibility seed
        verbose:      bool, print per-trial results

    Returns:
        dict with keys:
            "all_pass"     : bool   — True if all trials passed
            "n_consistent" : int    — trials with consistent syndrome
            "n_free"       : int    — consistent trials with free variables
            "n_fail"       : int    — trials with inconsistent syndrome
    """
    # from ge_decoder import (
    #     erasure_decode_f2,
    #     erasure_decode_sparse,
    #     erasure_decode_sparse_v2,
    #     erasure_decode_sparse_v3,
    # )

    decoders = {
        "Dense"     : erasure_decode_f2,
        "Sparse v1" : erasure_decode_sparse,
        "Sparse v2" : erasure_decode_sparse_v2,
        "Sparse v3" : erasure_decode_sparse_v3,
    }

    N            = Hx.shape[1]
    rng          = np.random.default_rng(random_seed)
    all_pass     = True
    n_consistent = 0
    n_free       = 0
    n_fail       = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  HGP code {label}   N={N} qubits   "
              f"erasure_rate={erasure_rate}   {n_trials} trials")
        print(f"{'='*60}")

    for trial in range(n_trials):

        # Sample erasure pattern
        n_erased    = int(N * erasure_rate)
        erased_bits = rng.choice(N, size=n_erased, replace=False)
        erasure_set = set(erased_bits.tolist())

        # Zero syndrome — all-zero codeword, no errors on erased qubits
        sx = np.zeros(Hx.shape[0], dtype=int)

        # Run all four decoders
        trial_results = {}
        for name, fn in decoders.items():
            sol, ok, free = fn(Hx, sx, erasure_set)
            trial_results[name] = {
                "sol" : sol,
                "ok"  : ok,
                "free": sorted(free) if free else [],
            }

        # Check agreement
        trial_pass, _ = _check_agreement(
            trial_results, Hx, sx, verbose=verbose
        )
        if not trial_pass:
            all_pass = False

        # Tally outcomes (based on Dense reference)
        ref = trial_results["Dense"]
        if ref["ok"]:
            n_consistent += 1
            if len(ref["free"]) > 0:
                n_free += 1
        else:
            n_fail += 1

        if verbose:
            status = (
                "FAIL (inconsistent)" if not ref["ok"]
                else f"OK  free={len(ref['free'])}"
            )
            mark = "✓" if trial_pass else "✗"
            print(f"  trial {trial+1:2d}  |erasure|={n_erased:4d}  "
                  f"{status:<25}  {mark}")

    if verbose:
        print()
        print(f"  Summary:")
        print(f"    consistent   : {n_consistent}/{n_trials}")
        print(f"    with free var: {n_free}/{n_trials}")
        print(f"    inconsistent : {n_fail}/{n_trials}")
        print(f"    result       : "
              f"{'PASS' if all_pass else 'FAIL — see above'}")

    return {
        "all_pass"     : all_pass,
        "n_consistent" : n_consistent,
        "n_free"       : n_free,
        "n_fail"       : n_fail,
    }


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Load HGP code families from Connolly et al. txt files and\n"
            "verify all four GE decoder versions produce correct results."
        )
    )
    parser.add_argument(
        "--code", default="all",
        choices=list(CODE_FAMILIES.keys()) + ["all"],
        help="Code family to test (default: all)."
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing the classical H txt files (default: .)."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available code families and exit."
    )
    parser.add_argument(
        "--erasure-rate", type=float, default=ERASURE_RATE,
        help=f"Fraction of qubits erased per trial (default: {ERASURE_RATE})."
    )
    parser.add_argument(
        "--trials", type=int, default=N_DECODE_TRIALS,
        help=f"Number of erasure patterns per code (default: {N_DECODE_TRIALS})."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})."
    )
    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────────────────
    if args.list:
        print("\nAvailable code families:")
        print(f"  {'Name':<8}  {'Parameters':<12}  File")
        print(f"  {'-'*8}  {'-'*12}  {'-'*55}")
        for name, (fname, params) in CODE_FAMILIES.items():
            print(f"  {name:<8}  "
                  f"[[{params['N']},{params['K']}]]{'':4}  {fname}")
        print()
        raise SystemExit(0)

    # ── Select families ────────────────────────────────────────────────────
    families = (
        list(CODE_FAMILIES.items())
        if args.code == "all"
        else [(args.code, CODE_FAMILIES[args.code])]
    )

    # ── Header ────────────────────────────────────────────────────────────
    print()
    print("HGP Code Test Suite")
    print("───────────────────")
    print(f"  data dir     : {os.path.abspath(args.data_dir)}")
    print(f"  erasure rate : {args.erasure_rate}")
    print(f"  trials       : {args.trials}")
    print(f"  random seed  : {args.seed}")
    print()

    overall_pass = True

    for code_name, (fname, params) in families:

        filepath = os.path.join(args.data_dir, fname)
        label    = f"[[{params['N']},{params['K']}]]"

        # ── Load ──────────────────────────────────────────────────────────
        try:
            H_cl = load_classical_H(filepath)
        except FileNotFoundError as e:
            print(f"  SKIP {label}: {e}")
            print()
            continue

        print(f"Loaded {label}:")
        print(f"  Classical H  : shape={H_cl.shape}")
        print(f"  Row weight   : "
              f"avg={H_cl.sum(axis=1).mean():.1f}  "
              f"min={H_cl.sum(axis=1).min()}  "
              f"max={H_cl.sum(axis=1).max()}")
        print(f"  Col weight   : "
              f"avg={H_cl.sum(axis=0).mean():.1f}  "
              f"min={H_cl.sum(axis=0).min()}  "
              f"max={H_cl.sum(axis=0).max()}")

        # ── Build HGP ─────────────────────────────────────────────────────
        Hx, Hz = build_hgp(H_cl)
        print(f"  Hx shape     : {Hx.shape}")
        print(f"  Hz shape     : {Hz.shape}")

        # ── CSS orthogonality ─────────────────────────────────────────────
        css_ok = verify_css_orthogonality(Hx, Hz)
        print(f"  CSS check    : {'PASS' if css_ok else 'FAIL'}")
        if not css_ok:
            print(f"  ERROR: CSS orthogonality failed — skipping decoding tests")
            overall_pass = False
            continue

        # ── Decoding tests ─────────────────────────────────────────────────
        results = run_hgp_tests(
            Hx, Hz,
            label        = label,
            erasure_rate = args.erasure_rate,
            n_trials     = args.trials,
            random_seed  = args.seed,
            verbose      = True,
        )

        if not results["all_pass"]:
            overall_pass = False

    # ── Final verdict ──────────────────────────────────────────────────────
    print()
    print("─" * 50)
    if overall_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — check output above for details")
    print()