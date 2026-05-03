# test_reorder.py
#
# Property-based correctness tests for dfs_reorder() and cm_reorder().
# Tests verify that both the OLD (pre-fix) and NEW (post-fix) implementations
# satisfy the same five correctness properties — not that they produce
# identical outputs (which is not required for valid reorderings).
#
# Correctness properties tested per function:
#   P1. Valid permutation    — output shape matches input, cons_ordering is bijection
#   P2. Columns unchanged    — column j of H_reordered equals column j of H
#   P3. Bandwidth reduced    — reordering reduces or preserves bandwidth (heuristic)
#   P4. Row content preserved — reordered rows are a permutation of original rows
#   P5. Decoder correctness  — peeling + GE on H_reordered satisfies H @ sol = s
#
# Regression approach (Option A — property-based):
#   Both old and new implementations are tested against the same five properties.
#   Different but equally valid permutations are accepted — the test verifies
#   correctness of outcome, not identity of specific permutation chosen.
#   This is the right approach for algorithms with multiple valid correct outputs.
#
# Test matrices:
#   Small hand-crafted: Hamming [7,4,3], identity, dense, random LDPC, Step3 cycle
#   HGP code families:  [[625,25]], [[1225,65]], [[1600,64]], [[2025,81]]
#
# Usage
# ─────
#   Run all tests (both functions, all matrices):
#       python test_reorder.py
#
#   Run one function only:
#       python test_reorder.py --fn dfs
#       python test_reorder.py --fn rcm
#
#   Run one property only:
#       python test_reorder.py --prop permutation
#       python test_reorder.py --prop columns
#       python test_reorder.py --prop bandwidth
#       python test_reorder.py --prop rows
#       python test_reorder.py --prop decoder
#
#   Verbose output (per-check detail):
#       python test_reorder.py --verbose
#
#   Include HGP families (requires txt files):
#       python test_reorder.py --data-dir ./codes/
#
# From a notebook or another script:
#   from test_reorder import run_all_tests
#   run_all_tests(verbose=True, data_dir=".")

import numpy as np
import os
import argparse
from scipy.sparse import csr_matrix

from peeling_reorder_benchmark import (
    dfs_reorder,
    cm_reorder,
    no_reorder,
    peeling_decoder,
    load_classical_H,
    build_hgp,
    CODE_FAMILIES,
)
from sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3
# from ge_decoder import erasure_decode_sparse_v3


# ── Bandwidth helper ───────────────────────────────────────────────────────
def compute_bandwidth(H):
    """
    Compute bandwidth of the row-row adjacency matrix of H.
    Bandwidth = max |i - j| over all nonzero entries A[i,j], i != j.
    Uses sparse multiply to avoid dense O(m²n) computation.

    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)

    Returns:
        bandwidth: int
    """
    H_sp   = csr_matrix(H)
    A_sp   = (H_sp @ H_sp.T).astype(bool)
    A_sp.setdiag(0)
    A_sp.eliminate_zeros()
    cx     = A_sp.tocoo()
    if cx.nnz == 0:
        return 0
    return int(np.max(np.abs(cx.row - cx.col)))


# ── Property test functions ────────────────────────────────────────────────
def prop_valid_permutation(H, reorder_fn, label="", verbose=False):
    """
    P1 — Output shape matches input and cons_ordering is a bijection.

    Checks:
        H_reordered.shape == H.shape
        len(cons_ordering) == m
        sorted(cons_ordering) == list(range(m))
    """
    m, n = H.shape
    H_r, perm = reorder_fn(H)

    shape_ok = (H_r.shape == H.shape)
    len_ok   = (len(perm) == m)
    bij_ok   = (sorted(perm) == list(range(m)))
    passed   = shape_ok and len_ok and bij_ok

    if verbose:
        print(f"    P1 valid_permutation [{label}]")
        print(f"       shape   : {H_r.shape} == {H.shape}  → {shape_ok}")
        print(f"       length  : {len(perm)} == {m}        → {len_ok}")
        print(f"       bijection                          → {bij_ok}")

    return passed


def prop_columns_unchanged(H, reorder_fn, label="", verbose=False):
    """
    P2 — Columns of H_reordered are identical to columns of H.
    Ensures erasure_index_set indices remain valid after reordering.

    Checks:
        Column sums are preserved per column (necessary condition)
        Each column of H_reordered is a permutation of same column of H
        H_reordered matches H[perm, :] directly
    """
    H_r, perm = reorder_fn(H)

    col_sums_ok  = np.array_equal(H_r.sum(axis=0), H.sum(axis=0))
    col_content_ok = all(
        sorted(H[:, j].tolist()) == sorted(H_r[:, j].tolist())
        for j in range(H.shape[1])
    )
    manual_ok    = np.array_equal(H[perm, :], H_r)
    passed       = col_sums_ok and col_content_ok and manual_ok

    if verbose:
        print(f"    P2 columns_unchanged [{label}]")
        print(f"       col sums preserved : {col_sums_ok}")
        print(f"       col content valid  : {col_content_ok}")
        print(f"       matches H[perm,:]  : {manual_ok}")

    return passed


def prop_bandwidth_reduced(H, reorder_fn, label="", verbose=False):
    """
    P3 — Reordering reduces or preserves bandwidth (heuristic, not guaranteed).
    Reports the before/after ratio. Non-increase is the target but not enforced
    as a hard failure since both DFS and RCM are heuristics.

    Returns:
        non_increase: bool
        bw_before:    int
        bw_after:     int
        ratio:        float (bw_after / bw_before)
    """
    bw_before   = compute_bandwidth(H)
    H_r, _      = reorder_fn(H)
    bw_after    = compute_bandwidth(H_r)
    non_increase = (bw_after <= bw_before)
    ratio        = bw_after / bw_before if bw_before > 0 else 1.0

    if verbose:
        direction = ("reduced" if bw_after < bw_before
                     else "unchanged" if bw_after == bw_before
                     else "INCREASED")
        print(f"    P3 bandwidth [{label}]")
        print(f"       before : {bw_before}")
        print(f"       after  : {bw_after}  ({direction})")
        print(f"       ratio  : {ratio:.3f}")

    return non_increase, bw_before, bw_after, ratio


def prop_row_content_preserved(H, reorder_fn, label="", verbose=False):
    """
    P4 — Every row in H_reordered exists in H and vice versa.
    The multiset of rows is preserved under permutation.

    Checks:
        Row weight multisets are identical
        Each reordered row exists in the original H
    """
    H_r, _ = reorder_fn(H)

    weights_ok = (
        sorted(H.sum(axis=1).tolist()) ==
        sorted(H_r.sum(axis=1).tolist())
    )
    H_set  = set(tuple(row.tolist()) for row in H)
    rows_ok = all(tuple(row.tolist()) in H_set for row in H_r)
    passed  = weights_ok and rows_ok

    if verbose:
        print(f"    P4 row_content_preserved [{label}]")
        print(f"       row weights preserved : {weights_ok}")
        print(f"       all rows in original  : {rows_ok}")

    return passed


def prop_decoder_correctness(
    H, reorder_fn, erasure_index_set, s=None, label="", verbose=False
):
    """
    P5 — Peeling + GE on H_reordered satisfies the original syndrome
    H @ sol = s over F2.

    Also verifies:
        Consistency verdict matches between original and reordered
        Solution from reordered H satisfies the original (not reordered) H

    Inputs:
        H:                 numpy 2D array, dtype=int
        reorder_fn:        callable
        erasure_index_set: set of int
        s:                 numpy 1D array or None (None = zero syndrome)
        label:             str
        verbose:           bool
    """
    if s is None:
        s = np.zeros(H.shape[0], dtype=int)

    H_r, _ = reorder_fn(H)

    def decode(Hmat):
        sol, residual, res_syn = peeling_decoder(Hmat, s, erasure_index_set)
        if residual:
            s_res = np.array(
                [res_syn.get(i, 0) for i in range(Hmat.shape[0])],
                dtype=int
            )
            ge_sol, ok, _ = erasure_decode_sparse_v3(
                Hmat, s_res, residual
            )
            if ok and ge_sol is not None:
                for j in residual:
                    sol[j] = ge_sol[j]
            return sol, ok
        return sol, True

    sol_orig,  ok_orig  = decode(H)
    sol_reord, ok_reord = decode(H_r)

    # Consistency must agree
    consistency_ok = (ok_orig == ok_reord)

    # Reordered solution must satisfy the ORIGINAL H
    residual_check = (H @ sol_reord) % 2
    syn_ok         = np.array_equal(residual_check, s)

    # Solutions must agree when both consistent
    sol_ok = (
        np.array_equal(sol_orig, sol_reord)
        if (ok_orig and ok_reord) else True
    )

    passed = consistency_ok and syn_ok and sol_ok

    if verbose:
        print(f"    P5 decoder_correctness [{label}]")
        print(f"       consistency match  : {consistency_ok} "
              f"(orig={ok_orig}, reord={ok_reord})")
        print(f"       syndrome satisfied : {syn_ok}")
        print(f"       solution match     : {sol_ok}")
        if not syn_ok:
            bad = np.where(residual_check != s)[0]
            print(f"       residual nonzeros  : {bad[:10].tolist()}")

    return passed


# ── Test matrices ──────────────────────────────────────────────────────────
def make_small_matrices():
    """
    Return list of (label, H) covering structural edge cases.
    """
    rng = np.random.default_rng(42)

    # Hamming [7,4,3]
    H_hamming = np.array([
        [1,0,1,0,1,0,1],
        [0,1,1,0,0,1,1],
        [0,0,0,1,1,1,1]
    ], dtype=int)

    # Identity 5×5 — bandwidth already 0
    H_diag = np.eye(5, dtype=int)

    # Dense 4×8 — worst case fill-in
    H_dense = np.ones((4, 8), dtype=int)

    # Random (3,4)-LDPC n=20
    n, m = 20, 15
    H_rand = np.zeros((m, n), dtype=int)
    for i in range(m):
        cols = rng.choice(n, size=4, replace=False)
        H_rand[i, cols] = 1

    # Step 3 cycle graph from learning journey
    H_step3 = np.array([
        [1,1,0,0],
        [0,1,1,0],
        [1,0,1,1]
    ], dtype=int)

    return [
        ("Hamming [7,4,3]",      H_hamming),
        ("Identity 5x5",         H_diag),
        ("Dense 4x8",            H_dense),
        ("Random LDPC n=20",     H_rand),
        ("Step3 cycle",          H_step3),
    ]


def make_hgp_matrices(data_dir=".", verbose=False):
    """
    Load available HGP families. Returns list of (label, Hx).
    Skips families whose txt files are not found.
    """
    cases = []
    for code_name, meta in CODE_FAMILIES.items():
        filepath = os.path.join(data_dir, meta["file"])
        try:
            H_cl = load_classical_H(filepath)
            Hx, _ = build_hgp(H_cl)
            cases.append((meta["label"], Hx))
            if verbose:
                print(f"    Loaded {meta['label']}  Hx.shape={Hx.shape}")
        except FileNotFoundError:
            if verbose:
                print(f"    SKIP {meta['label']} — file not found")
    return cases


# ── Core runner for one function ───────────────────────────────────────────
def run_property_tests(
    reorder_fn,
    fn_name,
    matrices,
    prop_filter  = None,
    n_dec_trials = 10,
    verbose      = False,
    random_seed  = 42,
):
    """
    Run all five properties against a list of (label, H) matrices.

    Returns:
        passed_all: bool
        summary:    list of dicts with per-matrix results
    """
    rng        = np.random.default_rng(random_seed)
    passed_all = True
    summary    = []

    def should_run(name):
        return prop_filter is None or prop_filter == name

    for label, H in matrices:
        row = {"label": label, "fn": fn_name}

        # P1
        if should_run("permutation"):
            ok = prop_valid_permutation(H, reorder_fn, label=label,
                                        verbose=verbose)
            row["P1"] = ok
            if not ok:
                passed_all = False

        # P2
        if should_run("columns"):
            ok = prop_columns_unchanged(H, reorder_fn, label=label,
                                        verbose=verbose)
            row["P2"] = ok
            if not ok:
                passed_all = False

        # P3 — informational, not a hard failure
        if should_run("bandwidth"):
            ok, bw_b, bw_a, ratio = prop_bandwidth_reduced(
                H, reorder_fn, label=label, verbose=verbose
            )
            row["P3"]       = ok
            row["bw_before"] = bw_b
            row["bw_after"]  = bw_a
            row["bw_ratio"]  = ratio

        # P4
        if should_run("rows"):
            ok = prop_row_content_preserved(H, reorder_fn, label=label,
                                            verbose=verbose)
            row["P4"] = ok
            if not ok:
                passed_all = False

        # P5 — multiple random erasure patterns
        if should_run("decoder"):
            n_vars   = H.shape[1]
            n_passed = 0
            for trial in range(n_dec_trials):
                n_erased    = max(1, int(n_vars * 0.4))
                erased_bits = rng.choice(n_vars, size=n_erased, replace=False)
                erasure_set = set(erased_bits.tolist())
                ok = prop_decoder_correctness(
                    H, reorder_fn, erasure_set,
                    label=f"{label} t{trial+1}", verbose=False,
                )
                if ok:
                    n_passed += 1
            row["P5_passed"] = n_passed
            row["P5_total"]  = n_dec_trials
            if n_passed != n_dec_trials:
                passed_all = False

        summary.append(row)

    return passed_all, summary


def print_summary(summary, fn_name, prop_filter=None):
    """Print one-line result per matrix."""
    print(f"\n  Results for {fn_name}:")
    print(f"  {'Matrix':<28}  P1  P2  P3        P4  P5")
    print(f"  {'-'*28}  --  --  --------  --  ------")

    for row in summary:
        p1  = "✓" if row.get("P1", True) else "✗"
        p2  = "✓" if row.get("P2", True) else "✗"
        p4  = "✓" if row.get("P4", True) else "✗"

        if "bw_before" in row:
            bw_str = f"{row['bw_before']}→{row['bw_after']}"
        else:
            bw_str = "skip"

        if "P5_passed" in row:
            p5_str = f"{row['P5_passed']}/{row['P5_total']}"
        else:
            p5_str = "skip"

        # Core pass/fail: P1, P2, P4, P5 must all pass
        core_ok = (
            row.get("P1", True) and
            row.get("P2", True) and
            row.get("P4", True) and
            row.get("P5_passed", row.get("P5_total", 1)) ==
            row.get("P5_total", 1)
        )
        status = "✓" if core_ok else "✗"

        print(f"  {status} {row['label']:<27}  {p1}   {p2}   "
              f"{bw_str:<8}  {p4}   {p5_str}")


# ── Main runner ────────────────────────────────────────────────────────────
def run_all_tests(
    fn_filter    = None,
    prop_filter  = None,
    data_dir     = ".",
    verbose      = False,
    n_dec_trials = 10,
):
    """
    Run property-based regression tests for dfs_reorder and cm_reorder.

    Both functions are tested against the same five properties.
    Different but equally valid permutations are accepted — correctness
    of outcome is verified, not identity of specific permutation.

    Inputs:
        fn_filter:    "dfs" | "rcm" | None (None = both)
        prop_filter:  property name or None (None = all)
        data_dir:     str, directory containing HGP txt files
        verbose:      bool, print per-check detail
        n_dec_trials: int, erasure patterns per matrix for P5

    Returns:
        all_passed: bool
    """
    functions = {}
    if fn_filter is None or fn_filter == "dfs":
        functions["dfs_reorder"] = dfs_reorder
    if fn_filter is None or fn_filter == "rcm":
        functions["cm_reorder"]  = cm_reorder

    print()
    print("Reorder Correctness Tests (Property-Based Regression)")
    print("══════════════════════════════════════════════════════")
    print(f"  functions tested : {list(functions.keys())}")
    print(f"  property filter  : {prop_filter or 'all'}")
    print(f"  decoder trials   : {n_dec_trials} per matrix")
    print(f"  regression mode  : Option A — property-based")
    print(f"  (different but valid permutations accepted)")

    small_matrices = make_small_matrices()

    print()
    print("── Section 1: Small hand-crafted matrices ──────────────")

    all_passed = True
    for fn_name, reorder_fn in functions.items():
        passed, summary = run_property_tests(
            reorder_fn  = reorder_fn,
            fn_name     = fn_name,
            matrices    = small_matrices,
            prop_filter = prop_filter,
            n_dec_trials = n_dec_trials,
            verbose     = verbose,
        )
        print_summary(summary, fn_name, prop_filter)
        if not passed:
            all_passed = False

    # ── Section 2: HGP families ───────────────────────────────────────────
    print()
    print("── Section 2: HGP code families ────────────────────────")

    hgp_matrices = make_hgp_matrices(data_dir=data_dir, verbose=verbose)

    if not hgp_matrices:
        print(f"  No HGP txt files found in '{os.path.abspath(data_dir)}'")
        print(f"  Skipping Section 2.")
    else:
        for fn_name, reorder_fn in functions.items():
            passed, summary = run_property_tests(
                reorder_fn   = reorder_fn,
                fn_name      = fn_name,
                matrices     = hgp_matrices,
                prop_filter  = prop_filter,
                n_dec_trials = 5,       # fewer trials — large matrices
                verbose      = verbose,
            )
            print_summary(summary, fn_name, prop_filter)
            if not passed:
                all_passed = False

    # ── Section 3: Side-by-side bandwidth comparison ──────────────────────
    if prop_filter is None or prop_filter == "bandwidth":
        print()
        print("── Section 3: Bandwidth comparison (all functions) ─────")
        all_cases = small_matrices + (hgp_matrices if hgp_matrices else [])

        col_w = 28
        header = (f"  {'Matrix':<{col_w}}  "
                  f"{'none':>8}  {'dfs':>8}  {'rcm':>8}  "
                  f"{'dfs/none':>10}  {'rcm/none':>10}")
        print(header)
        print(f"  {'-'*col_w}  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*10}  {'-'*10}")

        for label, H in all_cases:
            bw_none = compute_bandwidth(H)

            H_dfs, _ = dfs_reorder(H)
            bw_dfs   = compute_bandwidth(H_dfs)

            H_rcm, _ = cm_reorder(H)
            bw_rcm   = compute_bandwidth(H_rcm)

            r_dfs = bw_dfs / bw_none if bw_none > 0 else 1.0
            r_rcm = bw_rcm / bw_none if bw_none > 0 else 1.0

            print(f"  {label:<{col_w}}  "
                  f"{bw_none:>8d}  {bw_dfs:>8d}  {bw_rcm:>8d}  "
                  f"{r_dfs:>10.3f}  {r_rcm:>10.3f}")

    # ── Final verdict ──────────────────────────────────────────────────────
    print()
    print("─" * 54)
    print(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED — check output above'}")
    print()

    return all_passed


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Property-based regression tests for dfs_reorder and cm_reorder.\n"
            "Verifies five correctness properties — not permutation identity.\n"
            "Safe to run against both old and new implementations."
        )
    )
    parser.add_argument(
        "--fn", default=None,
        choices=["dfs", "rcm"],
        help="Test one function only (default: both)."
    )
    parser.add_argument(
        "--prop", default=None,
        choices=["permutation", "columns", "bandwidth", "rows", "decoder"],
        help="Test one property only (default: all)."
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing HGP txt files (default: .)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-check detail for each test."
    )
    parser.add_argument(
        "--dec-trials", type=int, default=10,
        help="Erasure patterns per matrix for P5 decoder test (default: 10)."
    )
    args = parser.parse_args()

    run_all_tests(
        fn_filter    = args.fn,
        prop_filter  = args.prop,
        data_dir     = args.data_dir,
        verbose      = args.verbose,
        n_dec_trials = args.dec_trials,
    )