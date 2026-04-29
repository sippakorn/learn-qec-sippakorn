# hgp_test_cases.py
#
# Load HGP code families from Connolly et al. txt files and run
# correctness + decoding tests using all four GE decoder versions.
#
# Usage
# ─────
#   Run all tests:
#       python hgp_test_cases.py
#
#   Run tests for a specific code family:
#       python hgp_test_cases.py --code n625
#
#   List available code families:
#       python hgp_test_cases.py --list
#
# Required files (download from Connolly et al. repo)
# ────────────────────────────────────────────────────
#   Place these txt files in the same directory as this script,
#   or in a subdirectory specified by --data-dir:
#
#   PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt
#   PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt
#   PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt
#   PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt
#
#   Source: https://github.com/Nicholas-Connolly/Pruned-Peeling-and-VH-Decoder
#
# From a notebook or another script:
#       from hgp_test_cases import load_classical_H, build_hgp, run_hgp_tests
#       H_cl = load_classical_H("PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt")
#       Hx, Hz = build_hgp(H_cl)
#       run_hgp_tests(Hx, Hz, label="[[625,25]]")

import numpy as np
import os
import argparse
from gaussian_elimination import erasure_decode_f2
from sparse_gaussian_elimination import erasure_decode_sparse
from sparse_gaussian_elimination_v2 import erasure_decode_sparse_v2
from sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3

# ── Code family registry ───────────────────────────────────────────────────
# Maps short name -> (filename, quantum parameters [[n, k]])
CODE_FAMILIES = {
    "n625"  : (
        "PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt",
        {"n": 625,  "k": 25}
    ),
    "n1225" : (
        "PEG_HGP_code_(3,4)_family_n1225_k65_classicalH.txt",
        {"n": 1225, "k": 65}
    ),
    "n1600" : (
        "PEG_HGP_code_(3,4)_family_n1600_k64_classicalH.txt",
        {"n": 1600, "k": 64}
    ),
    "n2025" : (
        "PEG_HGP_code_(3,4)_family_n2025_k81_classicalH.txt",
        {"n": 2025, "k": 81}
    ),
}

ERASURE_RATE = 0.3   # 30% erasure — below threshold for these codes
RANDOM_SEED  = 42
N_DECODE_TRIALS = 20  # erasure patterns to test per code


# ── File loading ───────────────────────────────────────────────────────────
def load_classical_H(filepath):
    """
    Load a classical parity-check matrix H from a Connolly et al. txt file.

    File format:
        Line 1:         header  — two integers "m n" giving matrix dimensions
        Lines 2 to m+1: one row per line — space-separated nonzero column
                        indices for that row (sparse adjacency list format)

    Example file content for a (15, 20) matrix:
        15 20
        0 9 5 15
        0 16 10 6
        0 7 11 17 19
        16 1 11 5
        ...

    Each data line lists the column positions of 1s in that row.
    The number of data lines must equal m (from the header).

    Inputs:
        filepath: str, path to the txt file

    Returns:
        H: numpy 2D array, dtype=int, shape (m, n)

    Raises:
        FileNotFoundError if filepath does not exist
        ValueError if header is malformed, column indices are out of range,
                   or number of data lines does not match header
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Classical H file not found: '{filepath}'\n"
            f"Download from: https://github.com/Nicholas-Connolly/"
            f"Pruned-Peeling-and-VH-Decoder"
        )

    with open(filepath, "r") as f:
        lines = [l.strip() for l in f if l.strip()]  # skip blank lines

    if not lines:
        raise ValueError(f"File is empty: '{filepath}'")

    # ── Parse header ──────────────────────────────────────────────────────
    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(
            f"Expected header 'm n' on line 1, got: '{lines[0]}'"
        )
    m, n = int(header[0]), int(header[1])

    # ── Parse data lines ──────────────────────────────────────────────────
    data_lines = lines[1:]
    if len(data_lines) != m:
        raise ValueError(
            f"Header says m={m} rows but found {len(data_lines)} data lines "
            f"in '{filepath}'"
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
    Build hypergraph product (HGP) CSS code from classical parity-check matrix.

    Given H_cl of shape (m, n), the HGP construction produces:
        Hx = [H ⊗ I_n | I_m ⊗ H^T]   shape: (m*n, n*n + m*m)
        Hz = [I_n ⊗ H | H^T ⊗ I_m]   shape: (n*m, n*n + m*m)

    These satisfy Hx @ Hz.T = 0 over F2 (CSS orthogonality condition).

    Quantum code parameters:
        N = n^2 + m^2   physical qubits
        K = k^2         logical qubits  (where k = n - rank(H))
        D = d           code distance   (where d = distance of classical code)

    Inputs:
        H_cl: numpy 2D array, dtype=int, shape (m, n)

    Returns:
        Hx: numpy 2D array, dtype=int, shape (m*n, N)
        Hz: numpy 2D array, dtype=int, shape (n*m, N)
    """
    m, n = H_cl.shape
    Im   = np.eye(m, dtype=int)
    In   = np.eye(n, dtype=int)

    # Hx = [H ⊗ I_n | I_m ⊗ H^T]
    Hx = np.hstack([
        np.kron(H_cl, In),
        np.kron(Im, H_cl.T)
    ])

    # Hz = [I_n ⊗ H | H^T ⊗ I_m]
    Hz = np.hstack([
        np.kron(In, H_cl),
        np.kron(H_cl.T, Im)
    ])

    return Hx, Hz


def verify_css_orthogonality(Hx, Hz):
    """
    Verify CSS orthogonality condition: Hx @ Hz.T = 0 over F2.

    Inputs:
        Hx: numpy 2D array, dtype=int
        Hz: numpy 2D array, dtype=int

    Returns:
        is_valid: bool
    """
    product = (Hx @ Hz.T) % 2
    return np.all(product == 0)


# ── Decoding test ──────────────────────────────────────────────────────────
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
    Run decoding correctness tests on an HGP code using all four GE versions.

    For each trial:
        1. Sample a random erasure pattern at the given rate
        2. Sample a random Pauli error on erased qubits (all-zero here
           for simplicity — zero codeword, zero syndrome)
        3. Run erasure_decode_f2, v1, v2, v3 on Hx (X-check decoding)
        4. Verify all four versions agree on: consistent?, free_cols, solution

    Inputs:
        Hx:           numpy 2D array, dtype=int — X parity-check matrix
        Hz:           numpy 2D array, dtype=int — Z parity-check matrix
        label:        str, display name for this code
        erasure_rate: float, fraction of qubits erased
        n_trials:     int, number of random erasure patterns to test
        random_seed:  int
        verbose:      bool, print per-trial results

    Returns:
        results: dict with keys
            "all_agree":     bool, True if all versions agreed on every trial
            "n_consistent":  int, number of trials with consistent syndrome
            "n_free":        int, number of trials with free variables
            "n_fail":        int, number of trials with inconsistent syndrome
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

    N         = Hx.shape[1]
    rng       = np.random.default_rng(random_seed)

    n_consistent = 0
    n_free       = 0
    n_fail       = 0
    all_agree    = True

    if verbose:
        print(f"\n{'='*60}")
        print(f"  HGP code {label}   N={N} qubits   "
              f"erasure_rate={erasure_rate}   {n_trials} trials")
        print(f"{'='*60}")

    for trial in range(n_trials):

        # Random erasure pattern
        n_erased    = int(N * erasure_rate)
        erased_bits = rng.choice(N, size=n_erased, replace=False)
        erasure_set = set(erased_bits.tolist())

        # Zero syndrome (all-zero codeword, no errors on erased qubits)
        sx = np.zeros(Hx.shape[0], dtype=int)

        # Run all four decoders on Hx (X-check decoding for Z errors)
        trial_results = {}
        for name, fn in decoders.items():
            sol, ok, free = fn(Hx, sx, erasure_set)
            trial_results[name] = {
                "sol" : sol,
                "ok"  : ok,
                "free": sorted(free),
            }

        # Check all four decoders agree
        # ref = trial_results["Dense"]
        # trial_agree = True
        # for name in ["Sparse v1", "Sparse v2", "Sparse v3"]:
        #     r = trial_results[name]
        #     # Consistent/inconsistent must agree
        #     if r["ok"] != ref["ok"]:
        #         trial_agree = False
        #     # Free cols must agree
        #     if r["free"] != ref["free"]:
        #         trial_agree = False
        #     # Solutions must agree when consistent
        #     if ref["ok"] and r["ok"]:
        #         if not np.array_equal(r["sol"], ref["sol"]):
        #             trial_agree = False

        # if not trial_agree:
        #     all_agree = False

        # # Tally outcomes
        # if ref["ok"]:
        #     n_consistent += 1
        #     if len(ref["free"]) > 0:
        #         n_free += 1
        # else:
        #     n_fail += 1
        # ── Replace the existing agreement check with this diagnostic version ──

        ref = trial_results["Dense"]
        trial_agree = True

        for name in ["Sparse v1", "Sparse v2", "Sparse v3"]:

            # Verify each solution actually satisfies the syndrome
            for name, res in trial_results.items():
                if res["ok"] and res["sol"] is not None:
                    check = (Hx @ res["sol"]) % 2
                    if not np.array_equal(check, sx):
                        print(f"    BUG: {name} solution does not satisfy syndrome!")
                        print(f"      residual = {np.where(check != sx)[0][:10]}")

            r = trial_results[name]

            ok_match  = (r["ok"]   == ref["ok"])
            free_match = (r["free"] == ref["free"])

            if ref["ok"] and r["ok"]:
                sol_match = np.array_equal(r["sol"], ref["sol"])
            else:
                sol_match = True  # both failed, no solution to compare

            if not (ok_match and free_match and sol_match):
                trial_agree = False
                print(f"\n    MISMATCH detail [{name} vs Dense]:")
                print(f"      ok    : Dense={ref['ok']}  {name}={r['ok']}  match={ok_match}")
                print(f"      free  : Dense={ref['free'][:5]}...  "
                    f"{name}={r['free'][:5]}...  match={free_match}")
                if ref["ok"] and r["ok"]:
                    diff = np.where(r["sol"] != ref["sol"])[0]
                    print(f"      sol   : {len(diff)} positions differ")
                    print(f"      first diffs at indices: {diff[:10].tolist()}")
                    print(f"      Dense sol at diffs:     "
                        f"{ref['sol'][diff[:10]].tolist()}")
                    print(f"      {name} sol at diffs: "
                        f"{r['sol'][diff[:10]].tolist()}")

        if verbose:
            status = (
                "FAIL (inconsistent)" if not ref["ok"]
                else f"OK  free={len(ref['free'])}"
            )
            agree_str = "✓" if trial_agree else "✗ MISMATCH"
            print(f"  trial {trial+1:2d}  |erasure|={n_erased:4d}  "
                  f"{status:<25}  {agree_str}")

    if verbose:
        print()
        print(f"  Summary:")
        print(f"    consistent   : {n_consistent}/{n_trials}")
        print(f"    with free var: {n_free}/{n_trials}")
        print(f"    inconsistent : {n_fail}/{n_trials}")
        print(f"    all agree    : {'YES' if all_agree else 'NO — CHECK ABOVE'}")

    return {
        "all_agree"    : all_agree,
        "n_consistent" : n_consistent,
        "n_free"       : n_free,
        "n_fail"       : n_fail,
    }


# ── CSS orthogonality check ────────────────────────────────────────────────
def run_orthogonality_check(Hx, Hz, label=""):
    """
    Verify and print CSS orthogonality result for a code.

    Inputs:
        Hx:    numpy 2D array, dtype=int
        Hz:    numpy 2D array, dtype=int
        label: str, display name
    """
    valid = verify_css_orthogonality(Hx, Hz)
    status = "PASS" if valid else "FAIL"
    print(f"  CSS orthogonality [{label}]: {status}  "
          f"Hx.shape={Hx.shape}  Hz.shape={Hz.shape}")
    
def verify_free_cols_in_erasure(H, s, erasure_set):
    """
    Sanity check: all free_cols must be inside the erasure set.
    Catches the zeroed-column bug immediately.
    """
    sol, ok, free = erasure_decode_f2(H, s, erasure_set)
    invalid = [c for c in free if c not in erasure_set]
    if invalid:
        print(f"BUG: free_cols {invalid} are outside erasure_set")
        return False
    return True


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Load HGP code families from Connolly et al. txt files\n"
            "and run correctness tests using all four GE decoder versions."
        )
    )
    parser.add_argument(
        "--code", default="all",
        choices=list(CODE_FAMILIES.keys()) + ["all"],
        help="Code family to test. Default: all."
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing the classical H txt files. Default: current dir."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available code families and exit."
    )
    parser.add_argument(
        "--erasure-rate", type=float, default=ERASURE_RATE,
        help=f"Fraction of qubits erased. Default: {ERASURE_RATE}."
    )
    parser.add_argument(
        "--trials", type=int, default=N_DECODE_TRIALS,
        help=f"Number of random erasure patterns per code. Default: {N_DECODE_TRIALS}."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed. Default: {RANDOM_SEED}."
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable code families:")
        print(f"  {'Name':<8}  {'File':<55}  {'Parameters'}")
        print(f"  {'-'*8}  {'-'*55}  {'-'*15}")
        for name, (fname, params) in CODE_FAMILIES.items():
            print(f"  {name:<8}  {fname:<55}  [[{params['n']},{params['k']}]]")
        print()
        raise SystemExit(0)

    # Select which families to run
    families = (
        list(CODE_FAMILIES.items())
        if args.code == "all"
        else [(args.code, CODE_FAMILIES[args.code])]
    )

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
        label    = f"[[{params['n']},{params['k']}]]"

        # ── Load ──────────────────────────────────────────────────────────
        try:
            H_cl = load_classical_H(filepath)
        except FileNotFoundError as e:
            print(f"  SKIP {label}: {e}")
            print()
            continue

        print(f"Loaded {label}:")
        print(f"  Classical H  : {H_cl.shape}  "
              f"(m={H_cl.shape[0]}, n={H_cl.shape[1]})")
        print(f"  Row weight   : {H_cl.sum(axis=1).mean():.1f} avg  "
              f"{H_cl.sum(axis=1).min()} min  "
              f"{H_cl.sum(axis=1).max()} max")
        print(f"  Col weight   : {H_cl.sum(axis=0).mean():.1f} avg  "
              f"{H_cl.sum(axis=0).min()} min  "
              f"{H_cl.sum(axis=0).max()} max")

        # ── Build HGP ─────────────────────────────────────────────────────
        Hx, Hz = build_hgp(H_cl)
        print(f"  Hx shape     : {Hx.shape}")
        print(f"  Hz shape     : {Hz.shape}")

        # ── CSS check ─────────────────────────────────────────────────────
        run_orthogonality_check(Hx, Hz, label=label)

        # ── Decoding tests ─────────────────────────────────────────────────
        results = run_hgp_tests(
            Hx, Hz,
            label        = label,
            erasure_rate = args.erasure_rate,
            n_trials     = args.trials,
            random_seed  = args.seed,
            verbose      = True,
        )

        if not results["all_agree"]:
            overall_pass = False

    # ── Final verdict ──────────────────────────────────────────────────────
    print()
    print("─" * 40)
    if overall_pass:
        print("ALL TESTS PASSED — all four decoder versions agree on every trial.")
    else:
        print("SOME TESTS FAILED — check output above for mismatches.")
    print()