# learn-qec-sippakorn

A learning project on **erasure decoding of quantum LDPC codes** — Gaussian
Elimination over F₂, sparse decoders, peeling, reorderings, and tools to
record, replay, and visualise individual decoding sessions.

Aligned with the data conventions used in the
[Pruned-Peeling-and-VH-Decoder](https://github.com/Nicholas-Connolly/Pruned-Peeling-and-VH-Decoder)
by Connolly, Londe, Leverrier, Delfosse (arXiv:2208.01002).

---

## Subprojects

| Path | What it does |
|------|--------------|
| [record_replay/](record_replay/README.md) | Records and replays matrix-algorithm sessions step by step. Event-sourcing + AOP decorators — drop `@record_op` on any row-op method to capture it. Includes a Dash viewer, REPL annotator, and worked demos. **Start here →** [record_replay/README.md](record_replay/README.md) |
| [webapp/](webapp/README.md) | Next.js front end for browsing recorded sessions. |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Deep dive — the original learning notes: F₂ GE derivation, sparse v1/v2/v3, HGP construction, Hamming-code validation, benchmarks. |

---

## Top-level layout

```
core/           — decoder algorithm implementations (dense GE, sparse v1/v2/v3, common)
experiments/    — benchmark and scaling experiment scripts
test/           — correctness and regression tests
utility/        — plotting helpers
codes/          — LDPC parity-check matrices (HGP code families)
rawdata/        — saved benchmark outputs (.msgpack)
record_replay/  — session recorder, replayer, and Dash viewer
webapp/         — Next.js viewer
docs/           — long-form documentation
```

---

## Running

All commands run from the **project root** so Python's implicit namespace
packages resolve correctly. There are no `__init__.py` files — do not add any.

```bash
# Run a test
python test/test_sgev3.py

# Run an experiment
python experiments/scaling_experiment.py

# Record a Gaussian elimination session, then view it
python record_replay/main.py                # record (dense GE on random matrix)
python record_replay/main_f2.py             # record (F₂ GE on an HGP code)
python record_replay/main_replay.py         # launch the Dash viewer
python record_replay/annotate.py            # tag a session with metadata
```

See each subproject's README for details specific to that piece.
