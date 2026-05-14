# record_replay

Record matrix-algorithm sessions step by step, then scrub, replay, and
annotate them in a browser. Built for F₂ Gaussian Elimination on HGP
codes with biconnected-component (BCC) analysis.

Uses **event sourcing** for storage and **Aspect-Oriented Programming** for
capture — the algorithm never imports the recorder.

---

## Quick start

All commands run from the **project root**.

```bash
# 1. Record an F₂ GE session on an HGP code
#    (retries random erasures until the residual graph has a cut node)
python record_replay/main_f2.py

# 2. Browse the saved BCC state for that session
python record_replay/browse_bcc.py

# 3. Launch the replay viewer
python record_replay/main_replay.py

# 4. (Optional) Annotate the session with metadata
python record_replay/annotate.py
```

The viewer opens at `http://localhost:8050`. Sessions live under
[data/](data/) as `session_<id>/`.

---

## How recording works

### Event sourcing

The recorder writes three file types per session:

| File | Contents |
| --- | --- |
| `initial.msgpack` | Matrix before any operation |
| `commands.msgpack` | One `Event(type, params, step, …)` per row op |
| `checkpoint_<n>.msgpack` | Full matrix snapshot every 50 steps |

Reconstructing step N = load nearest checkpoint ≤ N, replay events forward.
Sequential playback costs one event per step.

### Aspect-Oriented capture

Two primitives in [aspects.py](aspects.py):

| Primitive | Role |
| --- | --- |
| `@record_op("event_type")` | Emits an `Event` on each call while a session is active; no-op otherwise. |
| `RecordingSession(recorder, matrix_fn)` | Arms recording on enter, closes session on exit. |

The algorithm class never references `Recorder` — recording is wired at
the call site only.

```python
from aspects import record_op, RecordingSession
from recorder import Recorder

class MyGE:
    def __init__(self, H): self._mat = H.copy()

    @record_op("swap_rows")
    def _swap_rows(self, row_i, row_j):
        self._mat[[row_i, row_j]] = self._mat[[row_j, row_i]]

    @record_op("xor_rows")
    def _xor_rows(self, target, source):
        self._mat[target] ^= self._mat[source]

    def run(self): ...   # pure algorithm, no recorder imports

recorder = Recorder(data_dir="record_replay/data")
gen = MyGE(H)
session_id = recorder.start_session(sp.csr_matrix(gen._mat))

with RecordingSession(recorder, lambda: sp.csr_matrix(gen._mat)):
    gen.run()
```

To exclude a parameter from recording: `@record_op("scale_row", capture=["row", "scalar"])`.

---

## F₂ session with BCC analysis — `main_f2.py`

This is the main entry point for quantum LDPC experiments. It:

1. Loads an HGP code from `codes/`.
2. Applies a random erasure at 35% rate.
3. Runs the peeling decoder; extracts the residual submatrix.
4. DFS-reorders the residual for GE fill reduction.
5. **Checks for cut nodes (articulation points) in the Tanner graph.**
   Retries steps 2–4 (up to 1 000 attempts) until the residual graph
   has at least one cut node — guaranteeing a non-trivial BCC structure.
6. Starts the recording session and saves the BCC state.

The BCC state file is saved as:

```text
data/bcc_states/h_active_<session_id>.npz
```

The filename matches the session ID so each BCC snapshot is unambiguously
linked to its recording.

---

## BCC state browser — `browse_bcc.py`

> For a full step-by-step explanation of how the BCC plot is produced, see
> [BCC_EXPLAINED.md](BCC_EXPLAINED.md).

Visualises saved `h_active_<id>.npz` files as bipartite Tanner graphs
using a **bipartite chain layout**:

- Variable nodes on top (y = 1), check nodes on bottom (y = 0).
- Biconnected components ordered left→right following the block-cut tree.
- Cut nodes (articulation points) sit at BCC boundaries, highlighted in orange.

```bash
python record_replay/browse_bcc.py               # interactive file selection
python record_replay/browse_bcc.py --state FILE  # load specific .npz
```

---

## Replay viewer — `main_replay.py`

Dash-based scrubber at `http://localhost:8050`.

```bash
python record_replay/main_replay.py
python record_replay/main_replay.py --session <id>
python record_replay/main_replay.py --port 8051
```

**Dual mode** (default) shows three panels simultaneously:

| Panel | Contents |
| --- | --- |
| Top-left | Matrix heatmap — amber overlay on changed cells |
| Middle | BCC Tanner graph — updates at every replay step |
| Bottom | Static BCC Tanner graph — initial H_active for the session |

**Single mode** shows the heatmap or Tanner graph with a toggle button.

Sidebar: event type, params, timestamp, session annotation.

---

## Annotating sessions — `annotate.py`

Attaches human-readable metadata to a session.

```bash
python record_replay/annotate.py
```

Fields: `code_family`, `erasure_rate`, `reorder`, `note`. Written to
`data/session_<id>/metadata.json`. Display only — the recorder and
replayer ignore it.

---

## Project structure

```
record_replay/
├── aspects.py        # @record_op decorator + RecordingSession
├── events.py         # Event dataclass
├── storage.py        # MessagePack serialization for numpy/scipy
├── recorder.py       # Session dir, step counter, checkpoints
├── replayer.py       # Reconstructs any step via checkpoint + replay
├── viewer.py         # Dash dashboard (heatmap + BCC Tanner graphs)
├── browse_bcc.py     # Standalone BCC state visualiser
├── generator.py      # Dense GE over the reals — @record_op example
├── generator_f2.py   # F₂ GE on [H|s] — @record_op example
├── main.py           # Demo: record dense GE on a random matrix
├── main_f2.py        # Demo: record F₂ GE on HGP code with BCC check
├── main_replay.py    # Launch the viewer
├── annotate.py       # Add metadata to a session
└── data/             # Recorded sessions + BCC states
```

### Data layout

```
data/
├── session_<id>/
│   ├── initial.msgpack          # matrix before any operation
│   ├── commands.msgpack         # ordered Events
│   ├── checkpoint_50.msgpack    # snapshot at step 50
│   ├── checkpoint_100.msgpack   # …100
│   └── metadata.json            # optional annotation
└── bcc_states/
    └── h_active_<id>.npz        # residual H_active for session <id>
```

---

## Adding a new event type

1. **Decorate** the method: `@record_op("my_op")`
2. **Teach the replayer** — add a branch in `_apply_event` in
   [replayer.py](replayer.py).

The viewer needs no changes — it renders whatever matrix the replayer
returns and shows `event_type` + `params` in the sidebar.
