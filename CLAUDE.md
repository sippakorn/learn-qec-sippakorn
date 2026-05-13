# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running tests and experiments

All commands must be run from the **project root** so Python resolves package imports correctly.

```bash
# Run a specific test
python test/test_sgev3.py
python test/test_reorder.py

# Run an experiment
python experiments/scaling_experiment.py            # benchmark + plot
python experiments/scaling_experiment.py --benchmark  # data only
python experiments/scaling_experiment.py --plot       # plot from saved data

# Run pytest (if installed)
pytest test/
```

There is no build step. There are no `__init__.py` files — Python's implicit namespace packages are used. Do not add `__init__.py` files.

## Project structure

```
core/           — decoder algorithm implementations
experiments/    — benchmark and scaling experiment scripts
test/           — correctness and regression tests
utility/        — plotting helpers (harry_plotter.py)
codes/          — LDPC parity-check matrices (.txt, HGP code families)
rawdata/        — saved benchmark outputs (.msgpack)
```

## Architecture

**Domain:** Erasure decoding of quantum LDPC codes over F₂. The decoder solves `H_ε · x_ε = s'` for erased bits using a two-phase strategy: peeling first, then Gaussian Elimination (GE) on the residual stopping set.

**Core algorithms (`core/`):**

| File | Contents |
|------|----------|
| `gaussian_elimination.py` | Dense GE over F₂: augmented matrix ops, row XOR, forward/back elimination |
| `sparse_gaussian_elimination.py` | Sparse GE v1: dict-of-sets representation |
| `sparse_gaussian_elimination_v2.py` | Sparse GE v2: col-to-rows index for faster pivot elimination |
| `sparse_gaussian_elimination_v3.py` | Sparse GE v3 (`erasure_decode_sparse_v3`): vectorised numpy uint8 batch XOR — main decoder used by experiments and tests |
| `common.py` | `peeling_decoder()` — iterative degree-1 check resolution; `dfs_reorder()` — DFS-based column/row reordering on the Tanner graph |

**Decoder pipeline** (implemented inside experiment scripts, not as a standalone function):
1. `peeling_decoder(H, s, erasure_set)` — resolves erased bits with degree-1 checks; returns residual erasure set and syndrome
2. `dfs_reorder()` or RCM reorder — permutes H columns/rows to reduce GE fill-in on the residual
3. `erasure_decode_sparse_v3()` — runs GE on the reordered residual submatrix

**HGP codes** (`codes/`) follow the `(3,4)-regular` family. Code parameters are encoded in filenames: `n625_k25` means n=625 physical qubits, k=25 logical qubits. Files store classical parity-check matrices that are used to construct the quantum HGP code.

**Experiment outputs** are saved as `.msgpack` files in `rawdata/` and loaded by the `--plot` mode of the same script that produced them.

## Import conventions

Tests and experiments import from `core.*` and `experiments.*` using absolute package paths:

```python
from core.sparse_gaussian_elimination_v3 import erasure_decode_sparse_v3
from core.common import peeling_decoder, dfs_reorder
from experiments.peeling_reorder_benchmark import some_helper
```

Running a script directly (`python test/test_sgev3.py` from the project root) works because the root is on `sys.path`. Never run scripts from inside a subdirectory.
