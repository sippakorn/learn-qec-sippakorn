# record_replay

Records and replays Gaussian elimination on sparse matrices step by step.
Uses command sourcing: stores the initial matrix once, then appends one event per row operation.
Checkpoints every 50 steps so replay never starts from the beginning.

## Install

```bash
pip install -r record_replay/requirements.txt
```

## Quick start

Run from the **project root**.

**1. Record a session** (200×200 matrix, 10% density):

```bash
python record_replay/main.py
```

Outputs a session summary and writes data to `record_replay/data/session_<id>/`.

**2. Launch the viewer**:

```bash
python record_replay/main_replay.py                    # most recent session
python record_replay/main_replay.py --session <id>     # specific session
python record_replay/main_replay.py --port 8051        # custom port
```

Open `http://localhost:8050` in a browser.

## Viewer controls

| Control | Action |
|---------|--------|
| Slider | Scrub to any step |
| Play / Pause | Auto-advance through steps |
| ◀ / ▶ | Step backward / forward one step |
| ⏮ / ⏭ | Jump to start / end |
| Speed | 0.5× 1× 2× 4× playback rate |
| Session dropdown | Switch sessions without restarting |

Amber-tinted cells show what changed since the previous step.
The sidebar shows `event_type`, `params`, and timestamp for the current step.

## Project structure

```
record_replay/
├── events.py       # Event dataclass
├── storage.py      # MessagePack serialization for numpy/scipy
├── recorder.py     # Recorder — records a session to disk
├── generator.py    # Gaussian elimination as an event emitter
├── replayer.py     # Replayer — seeks to any step using checkpoints
├── viewer.py       # Dash dashboard
├── main.py         # Record demo
└── main_replay.py  # Launch viewer
```

Session data lives in `record_replay/data/session_<id>/`:

```
initial.msgpack          # matrix before any operations
commands.msgpack         # all events in order
checkpoint_<n>.msgpack   # full matrix snapshot every 50 steps
```
