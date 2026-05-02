# test_cm_reorder.py
#
# Correctness tests for the cm_reorder() function (Reverse Cuthill-McKee).
#
# Tests cover:
#   1. Valid permutation    — output shape matches input, cons_ordering is bijection
#   2. Columns unchanged    — column j of H_reordered equals column j of H
#   3. Bandwidth reduced    — RCM reduces or preserves row-row adjacency bandwidth
#   4. Row content preserved — reordered rows are a permutation of original rows
#   5. Decoder correctness  — peeling + GE on H_reordered satisfies original syndrome
#   6. HGP code families    — end-to-end test on real Connolly et al. matrices
#
# Usage
# ─────
#   Run all tests:
#       python test_cm_reorder.py
#
#   Run specific test:
#       python test_cm_reorder.py --test bandwidth
#
#   Verbose output:
#       python test_cm_reorder.py --verbose
#
# From a notebook or another script:
#   from test_cm_reorder import run_all_tests
#   run_all_tests(verbose=True)

import numpy as np
import os
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

# ── Import from benchmark script ───────────────────────────────────────────
from peeling_reorder_benchmark import (
    cm_reorder,
    dfs_reorder,
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

    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)

    Returns:
        bandwidth: int
    """
    A = (H @ H.T).astype(bool).astype(int)
    np.fill_diagonal(A, 0)
    rows, cols = np.where(A > 0)
    if len(rows) == 0:
        return 0
    return int(np.max(np.abs(rows - cols)))


# ── Individual test functions ──────────────────────────────────────────────
def test_valid_permutation(H, label="", verbose=False):
    """
    Claim 1 — output shape matches input and cons_ordering is a bijection.

    Checks:
        - H_reordered.shape == H.shape
        - len(cons_ordering) == m
        - sorted(cons_ordering) == list(range(m))  (no duplicates, no gaps)
    """
    m, n = H.shape
    H_reordered, cons_ordering = cm_reorder(H)

    shape_ok = (H_reordered.shape == H.shape)
    len_ok   = (len(cons_ordering) == m)
    bij_ok   = (sorted(cons_ordering) == list(range(m)))

    passed = shape_ok and len_ok and bij_ok

    if verbose:
        print(f"  [valid_permutation] {label}")
        print(f"    shape match  : {H_reordered.shape} == {H.shape}  → {shape_ok}")
        print(f"    length match : {len(cons_ordering)} == {m}       → {len_ok}")
        print(f"    bijection    : sorted == range(m)               → {bij_ok}")

    return passed


def test_columns_unchanged(H, label="", verbose=False):
    """
    Claim 2 — columns of H_reordered are identical to columns of H.
    This ensures erasure_index_set indices remain valid after reordering.

    Checks:
        - H_reordered[:, j] == H_reordered_sorted_rows[:, j] for all j
        - Equivalently: column sums are preserved per column
        - And: each column of H_reordered is a permutation of the same column of H
    """
    H_reordered, cons_ordering = cm_reorder(H)

    # Column sums must be identical (necessary condition)
    col_sums_ok = np.array_equal(
        H_reordered.sum(axis=0),
        H.sum(axis=0)
    )

    # Column content: sort each column and compare
    # (permuting rows permutes the entries in each column)
    col_content_ok = True
    for j in range(H.shape[1]):
        orig_col    = sorted(H[:, j].tolist())
        reord_col   = sorted(H_reordered[:, j].tolist())
        if orig_col != reord_col:
            col_content_ok = False
            break

    # Direct check: reconstruct reordered by applying cons_ordering
    # and verify it matches cm_reorder output
    H_manual    = H[cons_ordering, :]
    manual_ok   = np.array_equal(H_manual, H_reordered)

    passed = col_sums_ok and col_content_ok and manual_ok

    if verbose:
        print(f"  [columns_unchanged] {label}")
        print(f"    col sums preserved : {col_sums_ok}")
        print(f"    col content valid  : {col_content_ok}")
        print(f"    manual match       : {manual_ok}")

    return passed


def test_bandwidth_reduced(H, label="", verbose=False):
    """
    Claim 3 — RCM reduces or preserves the bandwidth of the row-row
    adjacency matrix. Not guaranteed to always reduce (RCM is a heuristic),
    but should never produce a worse result than the original ordering
    on well-structured LDPC matrices.

    Note: strict reduction is not guaranteed on all inputs. We test for
    non-increase (bw_after <= bw_before) and report the ratio.
    """
    bw_before = compute_bandwidth(H)
    H_reordered, _ = cm_reorder(H)
    bw_after  = compute_bandwidth(H_reordered)

    non_increase = (bw_after <= bw_before)
    ratio        = bw_after / bw_before if bw_before > 0 else 1.0

    if verbose:
        print(f"  [bandwidth_reduced] {label}")
        print(f"    bandwidth before : {bw_before}")
        print(f"    bandwidth after  : {bw_after}")
        print(f"    ratio            : {ratio:.3f}  "
              f"({'reduced' if bw_after < bw_before else 'unchanged' if bw_after == bw_before else 'INCREASED'})")
        print(f"    non-increase     : {non_increase}")

    # Report as info even if bandwidth increased — RCM is a heuristic
    return non_increase, bw_before, bw_after, ratio


def test_row_content_preserved(H, label="", verbose=False):
    """
    Claim 4 — every row in H_reordered exists in H and vice versa.
    The multiset of rows is preserved under permutation.

    Checks:
        - For each row in H_reordered, it exists in H
        - Row weights (sums) are preserved as a multiset
    """
    H_reordered, cons_ordering = cm_reorder(H)

    # Row weights as multisets
    orig_weights   = sorted(H.sum(axis=1).tolist())
    reord_weights  = sorted(H_reordered.sum(axis=1).tolist())
    weights_ok     = (orig_weights == reord_weights)

    # Each reordered row should be findable in original H
    H_set = set(tuple(row.tolist()) for row in H)
    rows_ok = all(
        tuple(row.tolist()) in H_set
        for row in H_reordered
    )

    passed = weights_ok and rows_ok

    if verbose:
        print(f"  [row_content_preserved] {label}")
        print(f"    row weights preserved : {weights_ok}")
        print(f"    all rows in original  : {rows_ok}")

    return passed


def test_decoder_correctness(H, s, erasure_index_set, label="", verbose=False):
    """
    Claim 5 — running peeling + GE on H_reordered gives a solution
    that satisfies the original syndrome H @ sol = s over F2.

    Also verifies that the solution matches the one from the original H.

    Inputs:
        H:                 numpy 2D array, dtype=int, shape (m, n)
        s:                 numpy 1D array, dtype=int, shape (m,)
        erasure_index_set: set of int

    Checks:
        - Solution from H_reordered satisfies H @ sol = s (mod 2)
        - Solution from H_reordered matches solution from original H
        - Consistency verdict matches between both
    """
    H_reordered, _ = cm_reorder(H)

    # Decode on original H
    sol_orig, res_orig, res_syn_orig = peeling_decoder(H, s, erasure_index_set)
    if res_orig:
        s_res = np.array([res_syn_orig.get(i, 0) for i in range(H.shape[0])],
                         dtype=int)
        ge_sol, ok_orig, _ = erasure_decode_sparse_v3(H, s_res, res_orig)
        if ok_orig and ge_sol is not None:
            for j in res_orig:
                sol_orig[j] = ge_sol[j]
    else:
        ok_orig = True

    # Decode on reordered H
    sol_reord, res_reord, res_syn_reord = peeling_decoder(
        H_reordered, s, erasure_index_set
    )
    if res_reord:
        s_res = np.array([res_syn_reord.get(i, 0) for i in range(H.shape[0])],
                         dtype=int)
        ge_sol, ok_reord, _ = erasure_decode_sparse_v3(
            H_reordered, s_res, res_reord
        )
        if ok_reord and ge_sol is not None:
            for j in res_reord:
                sol_reord[j] = ge_sol[j]
    else:
        ok_reord = True

    # Syndrome satisfaction check on reordered solution
    residual     = (H @ sol_reord) % 2
    syn_ok       = np.array_equal(residual, s)

    # Consistency must agree
    consistency_ok = (ok_orig == ok_reord)

    # Solutions must agree (when both consistent)
    if ok_orig and ok_reord:
        sol_ok = np.array_equal(sol_orig, sol_reord)
    else:
        sol_ok = True  # both failed — acceptable

    passed = syn_ok and consistency_ok and sol_ok

    if verbose:
        print(f"  [decoder_correctness] {label}")
        print(f"    syndrome satisfied : {syn_ok}")
        print(f"    consistency match  : {consistency_ok} "
              f"(orig={ok_orig}, reord={ok_reord})")
        print(f"    solution match     : {sol_ok}")
        if not syn_ok:
            bad = np.where(residual != s)[0]
            print(f"    residual nonzeros  : {bad[:10].tolist()}")

    return passed


# ── Small hand-crafted test matrices ──────────────────────────────────────
def make_test_matrices():
    """
    Return a list of (label, H) pairs covering edge cases.
    """
    cases = []

    # Case 1: Hamming [7,4,3] — small, well-known
    H_hamming = np.array([
        [1,0,1,0,1,0,1],
        [0,1,1,0,0,1,1],
        [0,0,0,1,1,1,1]
    ], dtype=int)
    cases.append(("Hamming [7,4,3]", H_hamming))

    # Case 2: Identity-like — each row has one nonzero, bandwidth=0
    H_diag = np.eye(5, dtype=int)
    cases.append(("Identity 5x5", H_diag))

    # Case 3: Dense rows — worst case for fill-in
    H_dense = np.ones((4, 8), dtype=int)
    cases.append(("Dense 4x8", H_dense))

    # Case 4: (3,4)-regular random LDPC
    rng = np.random.default_rng(42)
    n, m = 20, 15
    H_rand = np.zeros((m, n), dtype=int)
    for i in range(m):
        cols = rng.choice(n, size=4, replace=False)
        H_rand[i, cols] = 1
    cases.append(("Random (3,4)-LDPC n=20", H_rand))

    # Case 5: Step 3 example from learning journey — one cycle
    H_step3 = np.array([
        [1,1,0,0],
        [0,1,1,0],
        [1,0,1,1]
    ], dtype=int)
    cases.append(("Step3 cycle graph", H_step3))

    return cases


# ── HGP test cases ─────────────────────────────────────────────────────────
def make_hgp_test_cases(data_dir=".", verbose=False):
    """
    Load available HGP code families and return (label, Hx) pairs.
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
                print(f"  Loaded {meta['label']}  Hx.shape={Hx.shape}")
        except FileNotFoundError:
            if verbose:
                print(f"  SKIP {meta['label']} — file not found")
    return cases


# ── Test runner ────────────────────────────────────────────────────────────
def run_all_tests(data_dir=".", verbose=False, test_filter=None):
    """
    Run all correctness tests for cm_reorder.

    Inputs:
        data_dir:    str, directory containing classical H txt files
        verbose:     bool, print per-test detail
        test_filter: str or None, run only tests matching this name

    Returns:
        all_passed: bool
    """
    results   = {}
    all_passed = True

    def should_run(name):
        return test_filter is None or test_filter in name

    print()
    print("cm_reorder Correctness Tests")
    print("════════════════════════════")

    # ── Section 1: Small matrices ─────────────────────────────────────────
    print()
    print("── Section 1: Small hand-crafted matrices ──")
    test_matrices = make_test_matrices()

    for label, H in test_matrices:
        row_results = {}

        if should_run("permutation"):
            ok = test_valid_permutation(H, label=label, verbose=verbose)
            row_results["valid_permutation"] = ok
            if not ok:
                all_passed = False

        if should_run("columns"):
            ok = test_columns_unchanged(H, label=label, verbose=verbose)
            row_results["columns_unchanged"] = ok
            if not ok:
                all_passed = False

        if should_run("bandwidth"):
            ok, bw_b, bw_a, ratio = test_bandwidth_reduced(
                H, label=label, verbose=verbose
            )
            row_results["bandwidth_reduced"] = ok
            row_results["bw_before"]         = bw_b
            row_results["bw_after"]          = bw_a
            row_results["bw_ratio"]          = ratio
            # Bandwidth increase is informational — not a hard failure
            # RCM is a heuristic and may not always reduce bandwidth

        if should_run("rows"):
            ok = test_row_content_preserved(H, label=label, verbose=verbose)
            row_results["row_content"] = ok
            if not ok:
                all_passed = False

        # Print one-line summary per matrix
        perm_ok   = row_results.get("valid_permutation", True)
        col_ok    = row_results.get("columns_unchanged", True)
        bw_ok     = row_results.get("bandwidth_reduced", True)
        bw_b      = row_results.get("bw_before", 0)
        bw_a      = row_results.get("bw_after",  0)
        row_ok    = row_results.get("row_content", True)
        core_ok   = perm_ok and col_ok and row_ok

        status = "✓ PASS" if core_ok else "✗ FAIL"
        bw_str = (f"bw: {bw_b}→{bw_a}"
                  if "bw_before" in row_results else "")
        print(f"  {status}  {label:<30}  {bw_str}")

        results[label] = row_results

    # ── Section 2: Decoder correctness on small matrices ──────────────────
    if should_run("decoder"):
        print()
        print("── Section 2: Decoder correctness (small matrices) ──")
        rng = np.random.default_rng(42)
        N_DECODER_TRIALS = 10

        for label, H in test_matrices:
            n_vars   = H.shape[1]
            n_passed = 0

            for trial in range(N_DECODER_TRIALS):
                n_erased    = max(1, int(n_vars * 0.4))
                erased_bits = rng.choice(n_vars, size=n_erased, replace=False)
                erasure_set = set(erased_bits.tolist())
                s           = np.zeros(H.shape[0], dtype=int)

                ok = test_decoder_correctness(
                    H, s, erasure_set,
                    label=f"{label} trial {trial+1}",
                    verbose=False,
                )
                if ok:
                    n_passed += 1

            status = "✓ PASS" if n_passed == N_DECODER_TRIALS else "✗ FAIL"
            print(f"  {status}  {label:<30}  "
                  f"{n_passed}/{N_DECODER_TRIALS} trials passed")
            if n_passed != N_DECODER_TRIALS:
                all_passed = False

    # ── Section 3: HGP code families ──────────────────────────────────────
    if should_run("hgp"):
        print()
        print("── Section 3: HGP code families ──")
        hgp_cases = make_hgp_test_cases(data_dir=data_dir, verbose=verbose)

        if not hgp_cases:
            print("  No HGP files found — skipping Section 3")
            print(f"  Place txt files in: {os.path.abspath(data_dir)}")
        else:
            for label, Hx in hgp_cases:
                row_results = {}

                if should_run("permutation"):
                    ok = test_valid_permutation(
                        Hx, label=label, verbose=verbose
                    )
                    row_results["valid_permutation"] = ok
                    if not ok:
                        all_passed = False

                if should_run("columns"):
                    ok = test_columns_unchanged(
                        Hx, label=label, verbose=verbose
                    )
                    row_results["columns_unchanged"] = ok
                    if not ok:
                        all_passed = False

                if should_run("bandwidth"):
                    ok, bw_b, bw_a, ratio = test_bandwidth_reduced(
                        Hx, label=label, verbose=verbose
                    )
                    row_results["bw_before"] = bw_b
                    row_results["bw_after"]  = bw_a
                    row_results["bw_ratio"]  = ratio

                if should_run("rows"):
                    ok = test_row_content_preserved(
                        Hx, label=label, verbose=verbose
                    )
                    row_results["row_content"] = ok
                    if not ok:
                        all_passed = False

                if should_run("decoder"):
                    rng = np.random.default_rng(42)
                    N_DECODER_TRIALS = 5
                    n_vars   = Hx.shape[1]
                    n_passed = 0
                    for trial in range(N_DECODER_TRIALS):
                        n_erased    = int(n_vars * 0.3)
                        erased_bits = rng.choice(
                            n_vars, size=n_erased, replace=False
                        )
                        erasure_set = set(erased_bits.tolist())
                        s           = np.zeros(Hx.shape[0], dtype=int)
                        ok = test_decoder_correctness(
                            Hx, s, erasure_set,
                            label=f"{label} trial {trial+1}",
                            verbose=False,
                        )
                        if ok:
                            n_passed += 1
                    row_results["decoder"] = (n_passed, N_DECODER_TRIALS)
                    if n_passed != N_DECODER_TRIALS:
                        all_passed = False

                perm_ok = row_results.get("valid_permutation", True)
                col_ok  = row_results.get("columns_unchanged", True)
                row_ok  = row_results.get("row_content", True)
                core_ok = perm_ok and col_ok and row_ok
                bw_b    = row_results.get("bw_before", 0)
                bw_a    = row_results.get("bw_after",  0)
                dec     = row_results.get("decoder", None)

                status  = "✓ PASS" if core_ok else "✗ FAIL"
                bw_str  = f"bw: {bw_b}→{bw_a}"
                dec_str = (f"  decoder: {dec[0]}/{dec[1]}"
                           if dec is not None else "")
                print(f"  {status}  {label:<14}  {bw_str}{dec_str}")

    # ── Section 4: Compare all three strategies ────────────────────────────
    if should_run("compare"):
        print()
        print("── Section 4: Strategy comparison (bandwidth) ──")
        print(f"  {'Matrix':<30}  "
              f"{'none':>8}  {'DFS':>8}  {'RCM':>8}  "
              f"{'DFS/none':>10}  {'RCM/none':>10}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  "
              f"{'-'*10}  {'-'*10}")

        all_cases = make_test_matrices()
        hgp_cases = make_hgp_test_cases(data_dir=data_dir, verbose=False)
        all_cases += hgp_cases

        for label, H in all_cases:
            bw_none = compute_bandwidth(H)

            H_dfs, _  = dfs_reorder(H)
            bw_dfs    = compute_bandwidth(H_dfs)

            H_rcm, _  = cm_reorder(H)
            bw_rcm    = compute_bandwidth(H_rcm)

            ratio_dfs = bw_dfs / bw_none if bw_none > 0 else 1.0
            ratio_rcm = bw_rcm / bw_none if bw_none > 0 else 1.0

            print(f"  {label:<30}  "
                  f"{bw_none:>8d}  {bw_dfs:>8d}  {bw_rcm:>8d}  "
                  f"{ratio_dfs:>10.3f}  {ratio_rcm:>10.3f}")

    # ── Final verdict ──────────────────────────────────────────────────────
    print()
    print("─" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — check output above")
    print()

    return all_passed


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Correctness tests for cm_reorder (Reverse Cuthill-McKee)."
    )
    parser.add_argument(
        "--test", default=None,
        choices=["permutation", "columns", "bandwidth",
                 "rows", "decoder", "hgp", "compare"],
        help="Run only tests matching this name (default: all)."
    )
    parser.add_argument(
        "--data-dir", default=".",
        help="Directory containing classical H txt files (default: .)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output for each test."
    )
    args = parser.parse_args()

    run_all_tests(
        data_dir    = args.data_dir,
        verbose     = args.verbose,
        test_filter = args.test,
    )