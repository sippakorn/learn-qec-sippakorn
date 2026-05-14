"""Core recording engine.

Public API
----------
recorder = Recorder(data_dir="...")
session_id = recorder.start_session(initial_matrix)
recorder.record(event)                  # append a pre-built Event
recorder.record_event(type, params)     # convenience: build + append
recorder.checkpoint(matrix, step)       # persist a full-state snapshot
recorder.maybe_checkpoint(mat_provider) # snapshot iff step % INTERVAL == 0
summary = recorder.close()              # flush commands.msgpack, return stats
"""

import time
import uuid
from pathlib import Path
from typing import Callable

import scipy.sparse as sp

from events import Event
from storage import write_msgpack

CHECKPOINT_INTERVAL: int = 50


class Recorder:
    def __init__(self, data_dir: str | Path = "data"):
        self._data_dir = Path(data_dir)
        self.session_id: str | None = None
        self._session_dir: Path | None = None
        self._events: list[dict] = []
        self._event_counter: int = 0
        self._checkpoint_count: int = 0
        self._active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(self, initial_matrix: sp.spmatrix) -> str:
        """Create a new session directory and persist the initial matrix baseline."""
        self.session_id = uuid.uuid4().hex[:8]
        self._session_dir = self._data_dir / f"session_{self.session_id}"
        self._session_dir.mkdir(parents=True, exist_ok=True)

        write_msgpack(self._session_dir / "initial.msgpack", initial_matrix.tocsr())

        self._events = []
        self._event_counter = 0
        self._checkpoint_count = 0
        self._active = True
        return self.session_id

    def record(self, event: Event) -> Event:
        """Append an event to the in-memory command log.

        Assigns a monotonically increasing event_id before storing.
        """
        if not self._active:
            raise RuntimeError("Call start_session() before record()")

        event.event_id = self._event_counter
        self._event_counter += 1
        self._events.append(event.to_dict())
        return event

    def record_event(self, event_type: str, params: dict) -> Event:
        """Build and append an Event from primitive fields.

        Used by the @record_op aspect so algorithm code never has to import
        the Event class. event_id and step are both set to the current
        event counter; they advance in lockstep.
        """
        if not self._active:
            raise RuntimeError("Call start_session() before record_event()")

        event = Event(
            event_id=self._event_counter,
            event_type=event_type,
            params=dict(params),
            timestamp=time.time(),
            step=self._event_counter,
        )
        self._event_counter += 1
        self._events.append(event.to_dict())
        return event

    def checkpoint(self, matrix: sp.spmatrix, step: int) -> None:
        """Persist a full sparse-matrix snapshot at the given step."""
        if not self._active:
            raise RuntimeError("Call start_session() before checkpoint()")

        path = self._session_dir / f"checkpoint_{step}.msgpack"
        write_msgpack(path, matrix.tocsr())
        self._checkpoint_count += 1

    def maybe_checkpoint(self, matrix_provider: Callable) -> None:
        """Snapshot the current matrix when the event counter hits a
        CHECKPOINT_INTERVAL boundary. matrix_provider is a zero-arg callable
        returning the current matrix (sparse or dense — coerced to CSR).
        """
        if not self._active:
            return
        if self._event_counter == 0:
            return
        if self._event_counter % CHECKPOINT_INTERVAL != 0:
            return
        matrix = matrix_provider()
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        self.checkpoint(matrix, self._event_counter)

    def close(self) -> dict:
        """Flush the command log to disk and return a session summary."""
        if not self._active:
            raise RuntimeError("No active session to close")

        write_msgpack(self._session_dir / "commands.msgpack", self._events)
        self._active = False

        file_sizes = {
            p.name: p.stat().st_size
            for p in sorted(self._session_dir.iterdir())
        }
        return {
            "session_id": self.session_id,
            "total_events": self._event_counter,
            "total_checkpoints": self._checkpoint_count,
            "file_sizes": file_sizes,
        }
