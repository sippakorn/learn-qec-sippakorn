import numpy as np

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

def peeling_decoder(H, s, erasure_index_set):
    """
    Peeling decoder for classical linear code over the binary erasure channel.

    Iteratively resolves erased variables by finding dangling checks —
    check nodes with exactly one erased variable neighbour. When no
    dangling check exists, peeling is stuck and returns the residual.

    Inputs:
        H:                 numpy 2D array, dtype=int, shape (m, n)
        s:                 numpy 1D array, dtype=int, shape (m,)
        erasure_index_set: set of int, indices of erased bits

    Returns:
        solution:         numpy 1D array, dtype=int, shape (n,)
                          resolved bits set to their values,
                          unresolved bits set to 0
        residual_erasure: set of int
                          erased bits not resolved by peeling
                          empty set means peeling fully succeeded
        residual_syndrome: dict mapping check_index -> syndrome_bit
                           syndrome of checks still connected to
                           residual erasure — needed for GE fallback
    """
    n_vars   = H.shape[1]
    solution = np.zeros(n_vars, dtype=int)

    # ── Step 1 — Build adjacency structures ───────────────────────────────
    # check_to_vars[i] = set of erased variable indices connected to check i
    # var_to_checks[j] = set of check indices connected to erased variable j
    check_to_vars = {}
    var_to_checks = {j: set() for j in erasure_index_set}

    for i in range(H.shape[0]):
        neighbours = set(
            j for j in np.where(H[i] == 1)[0]
            if j in erasure_index_set
        )
        if neighbours:                      # skip checks with no erased neighbours
            check_to_vars[i] = neighbours
            for j in neighbours:
                var_to_checks[j].add(i)

    # ── Step 2 — Working syndrome (mutable copy, only active checks) ──────
    syndrome = {i: int(s[i]) for i in check_to_vars}

    # ── Step 3 — Initialise dangling queue ────────────────────────────────
    # Use a set for O(1) membership test and removal
    dangling = {i for i, nbrs in check_to_vars.items() if len(nbrs) == 1}

    # ── Step 4 — Peeling loop ─────────────────────────────────────────────
    while dangling:

        # Pop one dangling check
        check = dangling.pop()

        # Guard: check may have been invalidated by an earlier peel step
        # (can happen if two dangling checks shared a variable)
        if check not in check_to_vars:
            continue
        if len(check_to_vars[check]) != 1:
            continue

        # Identify and resolve the single erased variable
        var          = next(iter(check_to_vars[check]))
        var_value    = syndrome[check]
        solution[var] = var_value

        # ── Step 5 — Propagate to neighbouring checks ─────────────────────
        for neighbour_check in var_to_checks[var]:
            if neighbour_check == check:
                continue
            if neighbour_check not in check_to_vars:
                continue

            # Update syndrome
            syndrome[neighbour_check] ^= var_value

            # Remove resolved variable from neighbour
            check_to_vars[neighbour_check].discard(var)

            # Check if neighbour became dangling
            if len(check_to_vars[neighbour_check]) == 1:
                dangling.add(neighbour_check)

            # Check if neighbour became empty (all its variables resolved)
            elif len(check_to_vars[neighbour_check]) == 0:
                del check_to_vars[neighbour_check]
                del syndrome[neighbour_check]

        # ── Step 6 — Remove resolved variable and check from graph ────────
        del var_to_checks[var]
        del check_to_vars[check]
        del syndrome[check]

    # ── Step 7 — Collect residual ─────────────────────────────────────────
    residual_erasure = set(var_to_checks.keys())

    return solution, residual_erasure, syndrome