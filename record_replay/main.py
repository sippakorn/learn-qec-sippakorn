"""Demo: record a full Gaussian elimination session on a random sparse matrix.

Run from the project root:
    python record_replay/main.py
"""

import sys
import os

# Make sibling modules importable when invoked as `python record_replay/main.py`
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import scipy.sparse as sp

from aspects import RecordingSession
from recorder import Recorder
from generator import GaussianEliminationGenerator


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    # ------------------------------------------------------------------ #
    # Build initial matrix                                                 #
    # ------------------------------------------------------------------ #
    n = 200
    density = 0.10
    rng = np.random.default_rng(42)

    # scipy.sparse.random needs a callable that returns n values
    matrix = sp.random(
        n, n,
        density=density,
        format="csr",
        data_rvs=lambda k: rng.standard_normal(k),
        random_state=rng,
    ).astype(np.float64)

    print(f"Initial matrix : {n}×{n}, density={density:.0%}, nnz={matrix.nnz}")

    # ------------------------------------------------------------------ #
    # Start recording session                                              #
    # ------------------------------------------------------------------ #
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    recorder = Recorder(data_dir=data_dir)
    session_id = recorder.start_session(matrix)
    print(f"Session started: {session_id}")

    # ------------------------------------------------------------------ #
    # Run Gaussian elimination inside a RecordingSession.                  #
    # The generator itself is recorder-agnostic; @record_op on its row-op  #
    # methods emits events while the session is active.                    #
    # ------------------------------------------------------------------ #
    gen = GaussianEliminationGenerator(matrix)
    with RecordingSession(recorder, lambda: sp.csr_matrix(gen._mat)) as session:
        gen.run()

    summary = session.summary

    total_bytes = sum(summary["file_sizes"].values())

    print()
    print("=" * 48)
    print("  Session Summary")
    print("=" * 48)
    print(f"  Session ID      : {summary['session_id']}")
    print(f"  Total steps     : {summary['total_events']:,}")
    print(f"  Checkpoints     : {summary['total_checkpoints']}")
    print(f"  Total on disk   : {_human_bytes(total_bytes)}")
    print()
    print("  Files:")
    for fname, size in summary["file_sizes"].items():
        print(f"    {fname:<35s}  {_human_bytes(size):>10s}")
    print("=" * 48)


if __name__ == "__main__":
    main()
