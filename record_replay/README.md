# record_replay

Record matrix-algorithm sessions step by step, then scrub, replay, and
annotate them in a browser.

The recorder uses **event sourcing** to store the initial matrix once and
append one event per row operation. The capture layer uses
**Aspect-Oriented Programming** — drop a `@record_op` decorator on any row-op
method and it becomes recordable, with no recorder references inside the
algorithm itself.

---

## Concepts

### Record and replay (event sourcing)

Instead of snapshotting the matrix at every step, the recorder writes:

- `initial.msgpack` — the matrix before any operation
- `commands.msgpack` — every row operation as an `Event(event_type, params, step, …)`
- `checkpoint_<n>.msgpack` — a full matrix snapshot every 50 steps, so
  jumping to step N never replays from zero

Reconstructing any step = load the nearest checkpoint ≤ N, then replay
events forward to N. Sequential playback costs one event per step.

### Aspect-Oriented capture

Recording is orthogonal to the algorithm. Two primitives live in
[aspects.py](aspects.py):

| Primitive | Role |
|-----------|------|
| `@record_op(event_type)` | Method decorator. While a session is active, each call emits an `Event` whose `params` are auto-bound from the method signature. Outside a session, it's a no-op. |
| `RecordingSession(recorder, matrix_provider)` | Context manager. Inside the `with` block, decorated methods record; on exit it closes the session and exposes `.summary`. |

The algorithm class never imports `Recorder`, never builds `Event` objects,
never calls `self._recorder.record(...)`. Recording is wired up at the call
site, not inside the algorithm.

---

## Install

```bash
pip install -r record_replay/requirements.txt
```

Dependencies: `numpy`, `scipy`, `msgpack`, `dash`, `plotly`.

---

## Quick start

All commands run from the **project root**.

```bash
# 1. Record a session (random 200×200, 10% density, dense float GE)
python record_replay/main.py

# 2. Record an F₂ session on an HGP code (peeling + DFS reorder + GE)
python record_replay/main_f2.py

# 3. Launch the viewer on the most recent session
python record_replay/main_replay.py

# 4. (Optional) Annotate the session with metadata
python record_replay/annotate.py
```

The viewer opens at `http://localhost:8050`. Recorded sessions live under
[data/](data/) as `session_<id>/`.

---

## Recording your own matrix algorithm with AOP

If you have a new GE variant, peeling decoder, or any matrix algorithm,
making it recordable takes **two changes**:

1. Make sure each elementary mutation is its own method (extract them if
   the algorithm is monolithic).
2. Decorate those methods with `@record_op("event_name")`.

Then run the algorithm inside a `RecordingSession`. Nothing else changes.

### Minimal example

```python
import numpy as np
import scipy.sparse as sp

from aspects import record_op, RecordingSession
from recorder import Recorder


class MyGE:
    def __init__(self, H):
        self._mat = H.astype(np.float64)

    # Row ops — one decorator each. Params auto-captured from the signature.
    @record_op("swap_rows")
    def _swap_rows(self, row_i: int, row_j: int) -> None:
        self._mat[[row_i, row_j]] = self._mat[[row_j, row_i]]

    @record_op("xor_rows")
    def _xor_rows(self, target: int, source: int) -> None:
        self._mat[target] = (self._mat[target] + self._mat[source]) % 2

    # Algorithm body: no recorder code anywhere.
    def run(self):
        n, m = self._mat.shape
        cur = 0
        for col in range(m):
            piv = next((r for r in range(cur, n) if self._mat[r, col] == 1.0), None)
            if piv is None:
                continue
            if piv != cur:
                self._swap_rows(cur, piv)
            for r in range(n):
                if r != cur and self._mat[r, col] == 1.0:
                    self._xor_rows(target=r, source=cur)
            cur += 1


# Wire up recording at the call site.
H = (np.random.default_rng(0).integers(0, 2, size=(8, 8))).astype(np.float64)

ge = MyGE(H)
recorder = Recorder(data_dir="record_replay/data")
recorder.start_session(sp.csr_matrix(ge._mat))

with RecordingSession(recorder, lambda: sp.csr_matrix(ge._mat)) as session:
    ge.run()

print(session.summary)
```

That's it — every `_swap_rows` / `_xor_rows` call inside `run()` emits an
event while the session is active. Run the same `MyGE` outside the
`with` block and it runs identically with zero recording overhead.

### What the decorator does

For each call to a decorated method:

1. Inspects the method signature (cached once per decoration).
2. Binds the call's arguments to parameter names.
3. Coerces numpy scalars to native Python types so msgpack can serialize them.
4. Emits `recorder.record_event(event_type, params)`.
5. Calls the original method (which performs the mutation).
6. Calls `recorder.maybe_checkpoint(matrix_provider)` — snapshot every 50 events.

If no `RecordingSession` is active, steps 1–6 are skipped and the call
runs as a plain method.

### Customising the captured params

By default, every non-`self` argument is recorded. Override with `capture=[...]`:

```python
@record_op("scale_row", capture=["row", "scalar"])
def _scale_row(self, row: int, scalar: float, *, debug: bool = False):
    ...
```

Here `debug` is excluded from the recorded `params`.

### Real examples in this codebase

| File | What it shows |
|------|---------------|
| [generator.py](generator.py) | Dense GE over the reals — `swap_rows`, `scale_row`, `add_scaled_row` decorated. |
| [generator_f2.py](generator_f2.py) | F₂ GE on `[H \| s]` — `swap_rows`, `xor_rows` decorated. |
| [main.py](main.py), [main_f2.py](main_f2.py) | How to wire `RecordingSession` around the algorithm call. |

---

## Replay viewer — `main_replay.py`

Launch a Dash-based scrubber in the browser.

```bash
python record_replay/main_replay.py                    # most recent session
python record_replay/main_replay.py --session <id>     # specific session id
python record_replay/main_replay.py --port 8051        # custom port
python record_replay/main_replay.py --host 0.0.0.0     # bind all interfaces
```

Then open `http://localhost:8050` (or your chosen port).

### Controls

| Control | Action |
|---------|--------|
| Slider | Scrub to any step |
| Play / Pause | Auto-advance through steps |
| ◀ / ▶ | Step backward / forward one step |
| ⏮ / ⏭ | Jump to start / end |
| Speed | 0.5× 1× 2× 4× playback rate |
| Session dropdown | Switch sessions without restarting |

Amber-tinted cells show what changed since the previous step. The sidebar
shows `event_type`, `params`, and timestamp for the current event.

---

## Annotating sessions — `annotate.py`

Recorded sessions get UUID-style ids — not useful for telling them apart
later. `annotate.py` is a small REPL that attaches metadata to a session
so it shows up readable in the viewer's session dropdown.

```bash
python record_replay/annotate.py
```

Flow:

1. Pick a session from the numbered list (or paste an id).
2. Choose `a`dd/edit or `d`elete.
3. Fill the prompts — press Enter to keep an existing value:
   - `code_family` (e.g. `n625`, `n1225`)
   - `erasure_rate` (0–1)
   - `reorder` (e.g. `dfs`, `none`)
   - `note` (free text)

Metadata is written to `data/session_<id>/metadata.json`. It is purely a
display aid; the recorder and replayer ignore it.

---

## Project structure

```
record_replay/
├── aspects.py        # @record_op decorator + RecordingSession context manager
├── events.py         # Event dataclass
├── storage.py        # MessagePack serialization for numpy/scipy
├── recorder.py       # Recorder — owns session dir, step counter, checkpoints
├── replayer.py       # Replayer — reconstructs any step via checkpoint + replay
├── viewer.py         # Dash dashboard (matrix grid + scrubber)
├── generator.py      # Example: dense GE, recorded via @record_op
├── generator_f2.py   # Example: F₂ GE on [H|s], recorded via @record_op
├── main.py           # Demo: record dense GE on a random matrix
├── main_f2.py        # Demo: record F₂ GE on an HGP code (peeling + DFS reorder)
├── main_replay.py    # Launch the viewer
├── annotate.py       # REPL for adding metadata to a session
└── data/             # Recorded sessions (see layout below)
```

### Session data layout

```
data/session_<id>/
├── initial.msgpack          # matrix before any operation
├── commands.msgpack         # ordered list of Events
├── checkpoint_50.msgpack    # full sparse snapshot at step 50
├── checkpoint_100.msgpack   # …100
├── ...
└── metadata.json            # optional, written by annotate.py
```

Checkpoint interval is `CHECKPOINT_INTERVAL = 50` in [recorder.py](recorder.py).

---

## Adding a new event type

If your algorithm introduces a row op that isn't `swap_rows`, `xor_rows`,
`scale_row`, or `add_scaled_row`, you need to teach the replayer how to
apply it. Two places to touch:

1. **Algorithm side** — decorate the new method:
   ```python
   @record_op("permute_block")
   def _permute_block(self, start: int, end: int, perm: list[int]):
       ...
   ```

2. **Replay side** — add a branch in `_apply_event` in [replayer.py](replayer.py):
   ```python
   elif etype == "permute_block":
       start, end, perm = p["start"], p["end"], p["perm"]
       ...
   ```

The viewer doesn't need changes — it renders whatever matrix the replayer
returns and displays the event's `event_type` + `params` in the sidebar.
