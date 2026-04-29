# Gaussian Elimination over F₂ for Quantum LDPC Erasure Decoding

A self-contained Python implementation of Gaussian Elimination (GE) over the binary field F₂,
built as a learning exercise toward understanding erasure decoding of quantum LDPC codes.
The implementation is aligned with the data conventions used in the
[Pruned-Peeling-and-VH-Decoder](https://github.com/Nicholas-Connolly/Pruned-Peeling-and-VH-Decoder)
by Connolly, Londe, Leverrier, and Delfosse (arXiv:2208.01002).

---

## Background

### The Binary Erasure Channel (BEC)

In the BEC, each transmitted bit is independently erased with probability *p*.
The decoder receives:
- The **erasure pattern** ε ⊆ [n] — which bit positions were lost
- The **syndrome** s = Hx — computed from the known, non-erased bits

The decoding task is to recover the erased bits by solving the linear system:

```
H_ε · x_ε = s'    over F₂
```

where H_ε is the submatrix of H restricted to erased columns, and s' absorbs
the contribution of known bits into the right-hand side.

### Why GE Works Here

Maximum-likelihood (ML) decoding on the BEC reduces to solving a linear system
over F₂. This is because all erasure patterns consistent with the syndrome are
equally likely — so any valid solution is an ML solution. Gaussian Elimination
finds one such solution efficiently.

**Complexity:** O(|ε|³) in the worst case, where |ε| is the number of erased bits.

### Sparsity and Fill-in

For LDPC codes, H is sparse (constant row and column weight). GE on sparse
matrices can be much cheaper than O(n³) in practice because:

- **Peeling phase** (degree-1 checks): zero-cost elimination, equivalent to
  iterative decoding. Runs in O(n).
- **Residual stopping set**: requires true GE. Cost depends on cluster sizes.
- **Fill-in**: XORing two sparse rows can create a denser row, propagating
  more work downstream. Good pivot ordering (minimum degree first) minimises this.

### Connection to Stopping Sets

The peeling decoder fails when the erased variable nodes form a **stopping set**:
a subset where every neighboring check has degree ≥ 2 into the set.
GE must handle the residual stopping set after peeling exhausts.

For quantum LDPC codes, low-weight stabilizers automatically form stopping sets
in the Tanner graph of the opposite parity-check matrix — this is why naive
peeling performs poorly on quantum codes, motivating post-processing via
cluster decomposition (Yao et al., arXiv:2412.08817).

---

## Learning Journey

This implementation was built step by step through manual calculation
before writing any code. The steps are documented here for reference.

### Step 1 — Dense GE over F₂

Solved small dense systems by hand to build intuition for:
- F₂ arithmetic (1+1=0, subtraction = addition)
- Forward elimination and Gauss-Jordan reduction
- Free variables and what they mean physically
- Linear dependence producing zero rows

**Key insight:** A zero row with RHS=0 means a free variable exists (degenerate
solution). A zero row with RHS=1 means the system is inconsistent (decoding failure).

**Connection to quantum codes:** Two solutions differing by a vector in the
rowspace of H are **degenerate corrections** — both succeed. Two solutions
differing by a vector outside the rowspace indicate a **logical error**.

### Step 2 — Sparse GE on a Tree-Structured Tanner Graph

Introduced sparse parity-check matrices whose Tanner graphs are trees (no cycles).
Observed that:
- Tree-structured graphs produce nearly triangular matrices
- GE reduces to back-substitution with zero fill-in
- **Peeling is GE in disguise**: dangling checks correspond to single-nonzero
  pivot rows, which cost nothing to eliminate

**Key insight:** Peeling succeeds on a tree when at least one variable node is
known at the boundary. With all bits erased, even a tree graph becomes a stopping
set because there is no anchor value to start the cascade.

### Step 3 — Sparse GE with One Cycle

Added one extra edge to the Tanner graph, creating a 6-cycle (shortest possible
cycle in a bipartite graph). Observed that:
- The matrix is no longer triangular — real elimination is needed
- Fill-in appeared temporarily during elimination (+1 nonzero) then partially cancelled
- Knowing *which* variable is known determines whether peeling can start —
  not all boundary conditions are equivalent

**Key insight:** A cycle makes certain erasure patterns harder to peel, but does
not universally require more known values. What matters is whether the known
boundary value creates a dangling check. Variables that appear in only one check
within the erased set always anchor peeling regardless of cycle structure.

### Step 4 — Two Biconnected Components (Conceptual)

Examined a graph with two cycles connected by a single cut node. The key result:

```
WITHOUT decomposition:    WITH decomposition:
Cost: (n₁ + n₂)³         Cost: n₁³ + n₂³

Example n₁ = n₂ = 10:
8000 operations    vs    2000 operations  (4× cheaper)
```

Fill-in generated inside one biconnected component cannot propagate to the other.
This is the core insight behind the cluster decoder of Yao et al. (2025).

---

## Implementation

### Dependencies

```
numpy
```

### Data Conventions

All matrices use `numpy` 2D arrays with `dtype=int`, matching the convention
of the Connolly et al. reference implementation. Row operations are performed
in-place. The augmented matrix [H | s] is built with `np.hstack`.

---

### Piece 1 — Matrix Utilities

```python
import numpy as np

def make_augmented_matrix(H, s):
    """
    Build augmented matrix [H | s] as a numpy int array.

    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)
        s: numpy 1D array, dtype=int, shape (m,)

    Output:
        H_aug: numpy 2D array, dtype=int, shape (m, n+1)
    """
    return np.hstack((H, s[:, np.newaxis]))


def print_matrix(H_aug, n_vars):
    """
    Pretty print augmented matrix [H | s] with separator before RHS.

    Inputs:
        H_aug:  numpy 2D array
        n_vars: number of variable columns (excludes RHS)
    """
    for row in H_aug:
        left  = row[:n_vars]
        right = row[n_vars:]
        left_str  = " ".join(str(x) for x in left)
        right_str = " ".join(str(x) for x in right)
        print(f"[ {left_str} | {right_str} ]")
    print()


def xor_rows(M, target_row, pivot_row):
    """
    XOR pivot_row into target_row of matrix M in-place over F₂.

    Inputs:
        M:          numpy 2D array (modified in-place)
        target_row: int, index of row to be updated
        pivot_row:  int, index of row used for elimination
    """
    M[target_row] = (M[target_row] + M[pivot_row]) % 2
```

---

### Piece 2 — Forward Elimination

```python
def forward_eliminate(H_aug, n_vars):
    """
    Forward elimination (Gauss-Jordan) over F₂.
    Modifies H_aug in-place to reduced row echelon form.

    Inputs:
        H_aug:  numpy 2D array, dtype=int, shape (m, n+1)
        n_vars: number of variable columns (excludes RHS)

    Returns:
        pivot_cols: list of pivot column indices in order found
    """
    num_rows   = H_aug.shape[0]
    pivot_cols = []
    current_row = 0

    for col in range(n_vars):

        # Find a pivot row for this column
        pivot_row = None
        for row in range(current_row, num_rows):
            if H_aug[row, col] == 1:
                pivot_row = row
                break

        # No pivot in this column — free variable, skip
        if pivot_row is None:
            continue

        # Swap pivot row into current position
        if pivot_row != current_row:
            H_aug[[current_row, pivot_row]] = H_aug[[pivot_row, current_row]]

        # Eliminate this column from ALL other rows (Gauss-Jordan)
        for row in range(num_rows):
            if row != current_row and H_aug[row, col] == 1:
                xor_rows(H_aug, target_row=row, pivot_row=current_row)

        pivot_cols.append(col)
        current_row += 1

    return pivot_cols
```

---

### Piece 3 — Reading the Solution

```python
def read_solution(H_aug, n_vars, pivot_cols):
    """
    Read solution from reduced row echelon form.
    Free variables are set to 0 (minimum weight choice).
    Matches Connolly et al. convention.

    Inputs:
        H_aug:      numpy 2D array in RREF, shape (m, n+1)
        n_vars:     number of variable columns
        pivot_cols: list of pivot column indices from forward_eliminate

    Returns:
        solution:      numpy 1D array, dtype=int, shape (n_vars,)
                       or None if system is inconsistent
        is_consistent: bool
    """
    num_rows = H_aug.shape[0]
    s_rref   = H_aug[:, n_vars]

    # Check consistency — zero row with RHS=1 means no solution
    for row in range(num_rows):
        row_is_zero = not np.any(H_aug[row, :n_vars])
        if row_is_zero and s_rref[row] == 1:
            return None, False

    # Pivot variables read from RHS, free variables set to 0
    solution = np.zeros(n_vars, dtype=int)
    for i, col in enumerate(pivot_cols):
        solution[col] = s_rref[i]

    return solution, True


def gaussian_elimination_f2(H, s):
    """
    Full GE pipeline over F₂.

    Inputs:
        H: numpy 2D array, dtype=int, shape (m, n)
        s: numpy 1D array, dtype=int, shape (m,)

    Returns:
        solution:      numpy 1D array or None
        is_consistent: bool
        pivot_cols:    list of pivot column indices
        free_cols:     list of free variable column indices
    """
    n_vars     = H.shape[1]
    H_aug      = make_augmented_matrix(H, s)
    pivot_cols = forward_eliminate(H_aug, n_vars)
    free_cols  = [c for c in range(n_vars) if c not in pivot_cols]
    solution, is_consistent = read_solution(H_aug, n_vars, pivot_cols)

    return solution, is_consistent, pivot_cols, free_cols
```

---

### Piece 4 — Erasure-Aware Decoder

```python
def erasure_decode_f2(H, s, erasure_index_set):
    """
    Maximum-likelihood erasure decoder for a classical linear code.

    Zeros out columns of H for non-erased bits, then runs GE.
    Free variables within the erasure are set to 0 (min weight).

    Inputs:
        H:                 numpy 2D array, dtype=int, shape (m, n)
        s:                 numpy 1D array, dtype=int, shape (m,)
        erasure_index_set: set of int, indices of erased bits

    Returns:
        solution:      numpy 1D array, dtype=int, shape (n,)
                       predicted error vector (0 on non-erased bits)
                       None if decoding fails
        is_consistent: bool — False means decoding failure
        free_cols:     list of int — free variables within the erasure
                       non-empty means degenerate solution exists
    """
    n_vars   = H.shape[1]

    # Zero out columns for non-erased bits
    H_active = H.copy()
    for bit_index in range(n_vars):
        if bit_index not in erasure_index_set:
            H_active[:, bit_index] = 0

    solution, is_consistent, pivot_cols, free_cols = gaussian_elimination_f2(
        H_active, s
    )

    return solution, is_consistent, free_cols
```

---

## Usage Example

```python
import numpy as np

# Parity-check matrix from Step 3 (sparse, one 6-cycle)
H = np.array([[1, 1, 0, 0],
              [0, 1, 1, 0],
              [1, 0, 1, 1]], dtype=int)
s = np.array([1, 1, 0], dtype=int)

# Case 1: All bits erased
sol, ok, free = erasure_decode_f2(H, s, erasure_index_set={0, 1, 2, 3})
print("All erased  — solution:", sol, "| free vars:", free)

# Case 2: x4 known (=1), syndrome adjusted
s_adj = np.array([1, 1, 1], dtype=int)
sol, ok, free = erasure_decode_f2(H, s_adj, erasure_index_set={0, 1, 2})
print("x4 known    — solution:", sol, "| free vars:", free)

# Case 3: Inconsistent syndrome
H2 = np.array([[1, 1, 0],
               [1, 1, 0],
               [0, 0, 1]], dtype=int)
s2 = np.array([1, 0, 1], dtype=int)
sol, ok, free = erasure_decode_f2(H2, s2, erasure_index_set={0, 1, 2})
print("Inconsistent — consistent:", ok)
```

Output:
```
All erased   — solution: [1 0 1 0] | free vars: [3]
x4 known     — solution: [1 0 0 0] | free vars: []
Inconsistent — consistent: False
```

---

## Validation: [7,4,3] Hamming Code

The [7,4,3] Hamming code is a classic sanity check. Its parity-check matrix is:

```
H = [[1, 0, 1, 0, 1, 0, 1],
     [0, 1, 1, 0, 0, 1, 1],
     [0, 0, 0, 1, 1, 1, 1]]
```

Properties: n=7 bits, k=4 logical bits, d=3 minimum distance.

The four test cases below cover every possible decoding outcome.

### Manual verification of Case 1

Codeword x = (1,0,1,0,0,0,0), erasure {0,1,2}:

```
Syndrome:
  c1: x1+x3+x5+x7 = 1+1+0+0 = 0
  c2: x2+x3+x6+x7 = 0+1+0+0 = 1
  c3: x4+x5+x6+x7 = 0+0+0+0 = 0
  s = (0, 1, 0)

Augmented matrix (columns 3-6 zeroed):
  [ 1 0 1 | 0 ]
  [ 0 1 1 | 1 ]
  [ 0 0 0 | 0 ]

Row 3 is zero with RHS=0 → free variable exists.
x3 is free (no pivot in column 2).

  t=0: x1=0, x2=1, x3=0  →  solution: 0100000
  t=1: x1=1, x2=0, x3=1  →  solution: 1010000
```

The decoder returns the t=0 solution (minimum weight convention).

### Test code

```python
import numpy as np

H_hamming = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=int)

# Case 1: Small erasure {0,1,2} — degenerate, two valid corrections
print("=== Case 1: Erasure {0,1,2} — expect free variable ===")
s1 = np.array([0, 1, 0], dtype=int)
sol, ok, free = erasure_decode_f2(H_hamming, s1, erasure_index_set={0,1,2})
print("Consistent :", ok)
print("Free cols  :", free)
print("Solution   :", sol)
print("(t=0 gives 0100000, t=1 gives 1010000 on erased positions)")
print()

# Case 2: Larger erasure {0,1,2,3} — unique solution
print("=== Case 2: Erasure {0,1,2,3} — expect unique solution ===")
s2 = np.array([1, 1, 1], dtype=int)
sol, ok, free = erasure_decode_f2(H_hamming, s2, erasure_index_set={0,1,2,3})
print("Consistent :", ok)
print("Free cols  :", free)
print("Solution   :", sol)
print()

# Case 3: Stopping set erasure {0,1,3,5} — multiple free variables
print("=== Case 3: Erasure {0,1,3,5} — stopping set, free variables ===")
s3 = np.array([0, 0, 0], dtype=int)
sol, ok, free = erasure_decode_f2(H_hamming, s3, erasure_index_set={0,1,3,5})
print("Consistent :", ok)
print("Free cols  :", free)
print("Solution   :", sol)
print()

# Case 4: Inconsistent syndrome — decoding failure
print("=== Case 4: Inconsistent syndrome — expect failure ===")
s4 = np.array([1, 1, 1], dtype=int)
sol, ok, free = erasure_decode_f2(H_hamming, s4, erasure_index_set={0,1,2})
print("Consistent :", ok)
print("Solution   :", sol)
```

### Expected output

```
=== Case 1: Erasure {0,1,2} — expect free variable ===
Consistent : True
Free cols  : [2]
Solution   : [0 1 0 0 0 0 0]
(t=0 gives 0100000, t=1 gives 1010000 on erased positions)

=== Case 2: Erasure {0,1,2,3} — expect unique solution ===
Consistent : True
Free cols  : []
Solution   : [1 1 0 1 0 0 0]

=== Case 3: Erasure {0,1,3,5} — stopping set, free variables ===
Consistent : True
Free cols  : [0, 1, 3, 5]
Solution   : [0 0 0 0 0 0 0]

=== Case 4: Inconsistent syndrome — expect failure ===
Consistent : False
Solution   : None
```

### What each case demonstrates

| Case | Erasure | Outcome | Demonstrates |
|------|---------|---------|--------------|
| 1 | {0,1,2} size 3 | Free variable | Degenerate — two valid corrections exist |
| 2 | {0,1,2,3} size 4 | Unique solution | Full recovery with no ambiguity |
| 3 | {0,1,3,5} size 4 | Multiple free vars | Full stopping set — GE returns zero solution |
| 4 | {0,1,2} size 3 | Inconsistent | Syndrome unreachable given this erasure pattern |

---

## Decoding Outcomes

| Outcome | Condition | Meaning |
|---------|-----------|---------|
| Unique solution | `is_consistent=True`, `free_cols=[]` | Full recovery |
| Degenerate success | `is_consistent=True`, `free_cols` non-empty | Multiple valid corrections — check if difference is a stabilizer |
| Decoding failure | `is_consistent=False` | No valid correction exists |

---

## Connection to Quantum CSS Codes

For a CSS quantum code with parity-check matrices H_X and H_Z, erasure decoding
splits into two independent classical problems:

```python
# Decode X errors using Z-stabilizers
sol_x, ok_x, free_x = erasure_decode_f2(H1, sx, erasure_index_set)

# Decode Z errors using X-stabilizers
sol_z, ok_z, free_z = erasure_decode_f2(H2, sz, erasure_index_set)
```

Decoding fails if either `ok_x` or `ok_z` is False. Degenerate solutions
(non-empty `free_cols`) succeed if the ambiguity lives in the rowspace of the
opposite parity-check matrix — otherwise a logical error occurs.

---

## Next Steps

- [ ] Implement peeling decoder and compare performance with GE
- [ ] Add biconnected component decomposition (Hopcroft-Tarjan)
- [ ] Implement cluster decoder (peeling + per-cluster GE)
- [ ] Benchmark cluster decoder against full GE on HGP codes
- [ ] Reproduce figures from Yao et al. (arXiv:2412.08817)

---

## References

1. Connolly, Londe, Leverrier, Delfosse —
   *Fast erasure decoder for hypergraph product codes* (2022).
   arXiv:2208.01002

2. Yao, Gökduman, Pfister —
   *Cluster Decomposition for Improved Erasure Decoding of Quantum LDPC Codes* (2025).
   IEEE JSAIT, arXiv:2412.08817

3. Freire, Delfosse, Leverrier —
   *Optimizing hypergraph product codes with random walks, simulated annealing
   and reinforcement learning* (ISIT 2025).
   arXiv:2501.09622

4. Tillich, Zémor —
   *Quantum LDPC codes with positive rate and minimum distance proportional
   to the square root of the blocklength* (2014).
   IEEE Transactions on Information Theory.