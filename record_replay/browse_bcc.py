"""Browse saved H_active BCC state files and draw their Tanner graphs.

Run from project root:
    python record_replay/browse_bcc.py               # interactive selection
    python record_replay/browse_bcc.py --state FILE  # load specific file
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

BCC_DIR = os.path.join(os.path.dirname(__file__), "data", "bcc_states")


# ── File listing ──────────────────────────────────────────────────────────────

def list_state_files():
    if not os.path.isdir(BCC_DIR):
        return []
    return sorted(f for f in os.listdir(BCC_DIR) if f.endswith(".npz"))


def load_state(filename: str) -> np.ndarray:
    data = np.load(os.path.join(BCC_DIR, filename))
    return data["H_active"]


# ── Graph construction ────────────────────────────────────────────────────────

def build_tanner_graph(H: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    rows, cols = np.where(H != 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        G.add_edge(("v", i), ("c", j))
    return G


# ── Layout ────────────────────────────────────────────────────────────────────

def bipartite_chain_layout(
    G: nx.Graph,
    gap: float = 1.5,
    node_spacing: float = 1.0,
) -> dict:
    """
    Lay out G as a horizontal chain of bipartite Tanner sub-graphs.

    Variable nodes sit on y=1, check nodes on y=0.  BCCs are ordered
    left-to-right by a DFS on the block-cut tree.  Cut nodes are placed
    at the boundary between adjacent BCCs so they are visually shared.
    """
    bccs = list(nx.biconnected_components(G))
    cut_nodes = set(nx.articulation_points(G))
    n_bcc = len(bccs)

    if n_bcc == 0:
        return {}

    # ── Fallback: no cut nodes ─────────────────────────────────────────
    if not cut_nodes:
        pos = {}
        var_nodes = [n for n in G.nodes if n[0] == "v"]
        chk_nodes = [n for n in G.nodes if n[0] == "c"]
        for i, n in enumerate(var_nodes):
            pos[n] = np.array([i * node_spacing, 1.0])
        for i, n in enumerate(chk_nodes):
            pos[n] = np.array([i * node_spacing, 0.0])
        return pos

    # ── node → BCC membership ─────────────────────────────────────────
    node_to_bccs: dict = {}
    for idx, bcc in enumerate(bccs):
        for n in bcc:
            node_to_bccs.setdefault(n, []).append(idx)

    # ── BCC adjacency via shared cut nodes ────────────────────────────
    bcc_adj: dict = {i: [] for i in range(n_bcc)}
    for node in cut_nodes:
        bcc_idxs = node_to_bccs[node]
        for a in range(len(bcc_idxs)):
            for b in range(a + 1, len(bcc_idxs)):
                bcc_adj[bcc_idxs[a]].append((bcc_idxs[b], node))
                bcc_adj[bcc_idxs[b]].append((bcc_idxs[a], node))

    # ── DFS on BCCs to get left-to-right order ────────────────────────
    visited: set = set()
    order: list = []  # [(bcc_idx, entry_cut_or_None), ...]

    def dfs(idx: int, entry_cut) -> None:
        if idx in visited:
            return
        visited.add(idx)
        order.append((idx, entry_cut))
        for nbr_idx, shared_cut in bcc_adj[idx]:
            if nbr_idx not in visited:
                dfs(nbr_idx, shared_cut)

    for start in range(n_bcc):
        dfs(start, None)

    # ── Assign positions ──────────────────────────────────────────────
    pos: dict = {}
    x_cursor = 0.0

    for bcc_idx, entry_cut in order:
        bcc_nodes = bccs[bcc_idx]

        # Place entry cut node at x_cursor (first time seen)
        if entry_cut is not None:
            if entry_cut not in pos:
                y = 1.0 if entry_cut[0] == "v" else 0.0
                pos[entry_cut] = np.array([x_cursor, y])
            x_cursor = pos[entry_cut][0] + gap

        # Non-cut nodes belonging exclusively to this BCC
        non_cut = [n for n in bcc_nodes if n not in cut_nodes]
        var_nc = [n for n in non_cut if n[0] == "v"]
        chk_nc = [n for n in non_cut if n[0] == "c"]

        n_cols = max(len(var_nc), len(chk_nc), 1)
        width = (n_cols - 1) * node_spacing

        x_start = x_cursor
        x_end = x_start + width

        if var_nc:
            xs = np.linspace(x_start, x_end, len(var_nc))
            for n, x in zip(var_nc, xs):
                pos[n] = np.array([x, 1.0])
        if chk_nc:
            xs = np.linspace(x_start, x_end, len(chk_nc))
            for n, x in zip(chk_nc, xs):
                pos[n] = np.array([x, 0.0])

        x_cursor = x_end + gap

        # Place any exit cut nodes of this BCC not yet positioned
        for n in bcc_nodes:
            if n in cut_nodes and n not in pos:
                y = 1.0 if n[0] == "v" else 0.0
                pos[n] = np.array([x_cursor, y])
                x_cursor += gap

    return pos


# ── Drawing ───────────────────────────────────────────────────────────────────

_NODE_SIZE_NORMAL = 140
_NODE_SIZE_CUT    = 300
_COLOR_VAR = "#3498db"   # blue   — variable nodes
_COLOR_CHK = "#e74c3c"   # red    — check nodes
_COLOR_CUT = "#f39c12"   # orange — cut (articulation) nodes


def _node_label(node) -> str:
    kind, idx = node
    return f"{kind}{idx}"


def draw_merged_tanner_graph(H: np.ndarray, filename: str) -> None:
    G = build_tanner_graph(H)

    bccs       = list(nx.biconnected_components(G))
    bcc_edges  = list(nx.biconnected_component_edges(G))
    cut_nodes  = set(nx.articulation_points(G))
    n_bcc      = len(bccs)

    n_var = sum(1 for n in G if n[0] == "v")
    n_chk = sum(1 for n in G if n[0] == "c")

    print(f"  Nodes      : {G.number_of_nodes()}  (var={n_var}, chk={n_chk})")
    print(f"  Edges      : {G.number_of_edges()}")
    print(f"  BCC count  : {n_bcc}")
    print(f"  Cut nodes  : {len(cut_nodes)}  "
          f"({', '.join(_node_label(n) for n in sorted(cut_nodes))})")

    pos = bipartite_chain_layout(G)

    _, ax = plt.subplots(figsize=(max(14, n_var), 5))
    ax.set_title(
        f"Tanner Graph — {filename}\n"
        f"{H.shape[0]}×{H.shape[1]} matrix  |  "
        f"{n_bcc} BCC(s)  |  {len(cut_nodes)} cut node(s)",
        fontsize=12,
    )
    ax.axis("off")

    # ── Bipartite rails ────────────────────────────────────────────────
    ax.axhline(y=1.0, color="#dddddd", lw=0.8, ls="--", zorder=0)
    ax.axhline(y=0.0, color="#dddddd", lw=0.8, ls="--", zorder=0)
    ax.text(-0.2, 1.05, "variable (v)", va="bottom", ha="right",
            fontsize=9, color="#777777")
    ax.text(-0.2, -0.05, "check (c)", va="top", ha="right",
            fontsize=9, color="#777777")

    # ── BCC region shading ─────────────────────────────────────────────
    bcc_cmap = plt.cm.Pastel1
    bcc_colors = bcc_cmap(np.linspace(0, 0.8, max(n_bcc, 1)))
    for i, bcc_nodes in enumerate(bccs):
        xs = [pos[n][0] for n in bcc_nodes if n in pos]
        if not xs:
            continue
        x_lo = min(xs) - 0.4
        x_hi = max(xs) + 0.4
        ax.axvspan(x_lo, x_hi, ymin=-0.1, ymax=1.1,
                   alpha=0.25, color=bcc_colors[i], zorder=0,
                   label=f"BCC {i + 1}")

    # ── Edges colored by BCC ───────────────────────────────────────────
    edge_cmap = plt.cm.tab10 if n_bcc <= 10 else plt.cm.tab20
    edge_colors = edge_cmap(np.linspace(0, 0.9, max(n_bcc, 1)))
    for i, edge_set in enumerate(bcc_edges):
        nx.draw_networkx_edges(
            G, pos, edgelist=list(edge_set),
            edge_color=[edge_colors[i % len(edge_colors)]],
            width=1.8, alpha=0.75, ax=ax,
        )

    # ── Nodes ──────────────────────────────────────────────────────────
    var_normal = [n for n in G if n[0] == "v" and n not in cut_nodes]
    chk_normal = [n for n in G if n[0] == "c" and n not in cut_nodes]
    var_cut    = [n for n in G if n[0] == "v" and n in cut_nodes]
    chk_cut    = [n for n in G if n[0] == "c" and n in cut_nodes]

    if var_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=var_normal,
                               node_color=_COLOR_VAR, node_shape="o",
                               node_size=_NODE_SIZE_NORMAL, ax=ax)
    if chk_normal:
        nx.draw_networkx_nodes(G, pos, nodelist=chk_normal,
                               node_color=_COLOR_CHK, node_shape="s",
                               node_size=_NODE_SIZE_NORMAL, ax=ax)
    if var_cut:
        nx.draw_networkx_nodes(G, pos, nodelist=var_cut,
                               node_color=_COLOR_CUT, node_shape="o",
                               node_size=_NODE_SIZE_CUT, ax=ax)
    if chk_cut:
        nx.draw_networkx_nodes(G, pos, nodelist=chk_cut,
                               node_color=_COLOR_CUT, node_shape="s",
                               node_size=_NODE_SIZE_CUT, ax=ax)

    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=7,
        labels={n: _node_label(n) for n in G.nodes},
    )

    # ── Legend ─────────────────────────────────────────────────────────
    legend_handles = [
        plt.scatter([], [], c=_COLOR_VAR, marker="o", label="variable node"),
        plt.scatter([], [], c=_COLOR_CHK, marker="s", label="check node"),
        plt.scatter([], [], c=_COLOR_CUT, marker="o", s=80,
                    label="cut node (variable)"),
        plt.scatter([], [], c=_COLOR_CUT, marker="s", s=80,
                    label="cut node (check)"),
    ] + [
        plt.Rectangle((0, 0), 1, 1, fc=bcc_colors[i], alpha=0.4,
                       label=f"BCC {i + 1}")
        for i in range(n_bcc)
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, frameon=True)

    plt.tight_layout()
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def interactive_select(files: list[str]) -> str:
    print("\nSaved BCC state files:")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f}")
    while True:
        try:
            raw = input(f"\nSelect [1-{len(files)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except (ValueError, EOFError):
            pass
        print(f"  Please enter a number between 1 and {len(files)}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse BCC state files and draw merged bipartite Tanner graphs."
    )
    parser.add_argument("--state", metavar="FILE",
                        help="NPZ filename to load directly (basename only)")
    args = parser.parse_args()

    files = list_state_files()

    if not files:
        print(f"No BCC state files found in:\n  {BCC_DIR}")
        print("Run  python record_replay/main_f2.py  first to generate states.")
        sys.exit(1)

    if args.state:
        if args.state not in files:
            print(f"File not found in BCC state dir: {args.state}")
            print("Available files:")
            for f in files:
                print(f"  {f}")
            sys.exit(1)
        filename = args.state
    else:
        filename = interactive_select(files)

    print(f"\nLoading: {filename}")
    H_active = load_state(filename)
    print(f"  Shape : {H_active.shape}")

    draw_merged_tanner_graph(H_active, filename)


if __name__ == "__main__":
    main()
