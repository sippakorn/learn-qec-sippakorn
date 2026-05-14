# How the BCC Plot Works

> Part of the `record_replay` module — see [README.md](README.md) for the
> full overview, quick-start commands, and project structure.

A complete walkthrough from raw HGP code to the final bipartite chain plot.

---

## Why BCC?

After the peeling decoder finishes, any bits it could not resolve form a
**stopping set** — a subgraph where every check node has degree ≥ 2, so no
single bit can be determined without Gaussian Elimination (GE).

A stopping set with a **cut node** (articulation point) is structurally
interesting: the graph splits into two or more **biconnected components**
(BCCs) joined at that single node. This means:

- GE on one BCC can be treated somewhat independently of the other.
- The cut node is the only coupling between the components.

The BCC plot makes this structure visible.

---

## Pipeline: initial state → BCC plot

### Step 1 — Build HGP code

A classical (3,4)-regular LDPC parity-check matrix `H_cl` is loaded from
`codes/` and used to construct the quantum HGP code matrix `Hx`.

### Step 2 — Random erasure

35% of the `N` physical qubits (columns of `Hx`) are randomly erased.
A random syndrome `s` is drawn.

### Step 3 — Peeling decoder

`peeling_decoder(Hx, s, erasure_set)` iteratively resolves erased bits
whose check node has degree 1 — the unique erased bit is forced by the
syndrome. It repeats until no degree-1 check remains.

**Output:** the residual erasure set and residual syndrome — bits the
peeling decoder could not resolve.

If the residual is empty (peeling solved everything), this erasure instance
is discarded and a new one is drawn.

### Step 4 — Extract residual submatrix

The rows (active checks) and columns (unresolved bits) of `Hx` that remain
after peeling are extracted into a smaller dense matrix `H_sub`.

```
H_sub = Hx[active_check_rows, :][:, residual_bit_cols]
```

### Step 5 — DFS reorder

`dfs_reorder(H_sub)` permutes the rows and columns of `H_sub` using a
depth-first traversal of the Tanner graph. This concentrates non-zeros near
the diagonal, reducing fill-in when GE runs later.

The result is `H_active` — the matrix that will be fed to GE.

### Step 6 — Cut-node check (retry loop)

The Tanner graph of `H_active` is built:
- **Rows** of `H_active` → variable nodes  
- **Columns** of `H_active` → check nodes  
- Entry `H[i,j] ≠ 0` → edge between variable `i` and check `j`

`nx.articulation_points(G)` finds nodes whose removal disconnects the
graph. If none exist, this instance is discarded and the loop restarts from
Step 2 with a new random erasure (up to 1 000 attempts).

Once a cut node is found, `H_active` is saved:

```
data/bcc_states/h_active_<session_id>.npz
```

The filename is the session ID so the BCC snapshot is unambiguously linked
to the corresponding GE replay.

---

## Building the plot (`browse_bcc.py`)

### 1. Load and build the graph

`H_active` is loaded from the `.npz` file and the Tanner graph is
reconstructed exactly as in Step 6.

### 2. Find BCCs and cut nodes

```python
bccs      = nx.biconnected_components(G)   # sets of nodes per BCC
cut_nodes = nx.articulation_points(G)      # nodes in 2+ BCCs
```

A **biconnected component** is a maximal subgraph with no articulation
point — removing any single node leaves it connected.  
A **cut node** appears in two or more BCCs and is the only bridge between
them.

### 3. Bipartite chain layout

Instead of a free-form spring layout, the components are arranged as a
**horizontal chain** that mirrors the block-cut tree:

1. **Build block-cut tree** — a tree where BCC nodes and cut nodes
   alternate: `BCC₁ — cut — BCC₂ — cut — BCC₃ — …`
2. **DFS traversal** determines left-to-right order of BCCs.
3. **Assign x positions:**
   - Entry cut node → placed at the current x cursor.
   - Non-cut nodes of each BCC → spread evenly in the next x band.
   - Exit cut nodes → placed at the right edge of the BCC band.
4. **y positions are fixed by node type:**
   - Variable nodes (rows) → y = 1 (top rail)
   - Check nodes (columns) → y = 0 (bottom rail)

### 4. Render

| Node type | Shape | Colour |
| --- | --- | --- |
| Variable (non-cut) | Circle | Blue |
| Check (non-cut) | Square | Red |
| Variable cut node | Circle, larger | Orange |
| Check cut node | Square, larger | Orange |

Edges are drawn between each variable and check node connected in
`H_active`.

---

## What the plot shows

```
  v0   v1   v2        v4   v5   v6
   ○    ○    ○    ◉    ○    ○    ○      ← variable nodes (y=1)
   │╲   │    │╲   │╱  │    │   ╱│
   │  ╲ │    │  ╲ │╱  │    │  ╱ │
   □    □    □    ◈    □    □    □      ← check nodes   (y=0)
  c0   c1   c2        c3   c4   c5

        ◄─── BCC 1 ──►◄─ BCC 2 ──►
                       ↑
                    cut node
```

- **Left cluster** — BCC 1: a biconnected subgraph, fully coupled.
- **Right cluster** — BCC 2: another biconnected subgraph.
- **Orange node** (◉/◈) — the cut node shared by both BCCs. It is the
  only node whose removal would disconnect the stopping set into two
  independent parts.

The cut node tells you exactly where the stopping set is "weakest" — if
you could resolve that one bit or check, GE might be able to peel the
rest separately.
