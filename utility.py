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