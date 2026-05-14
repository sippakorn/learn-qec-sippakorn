"""Aspect-Oriented recording layer for matrix algorithms.

Decorate row-op methods with @record_op(...). Wrap the algorithm call in a
RecordingSession. The algorithm body itself stays recorder-agnostic — no
imports, no `self._recorder` references, no Event construction.

Example
-------
    class MyGE:
        def __init__(self, H):
            self._mat = H.copy()

        @record_op("swap_rows")
        def _swap_rows(self, row_i, row_j):
            self._mat[[row_i, row_j]] = self._mat[[row_j, row_i]]

        @record_op("xor_rows")
        def _xor_rows(self, target, source):
            self._mat[target] ^= self._mat[source]

        def run(self):
            ...   # body unchanged

    recorder = Recorder(data_dir=...)
    recorder.start_session(initial_matrix)
    ge = MyGE(H)
    with RecordingSession(recorder, lambda: sp.csr_matrix(ge._mat)) as session:
        ge.run()
    summary = session.summary
"""
from __future__ import annotations

import inspect
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Optional, Sequence

import numpy as np


_active: ContextVar[Optional[tuple]] = ContextVar(
    "_active_recording", default=None
)


def _to_native(v):
    """Coerce numpy scalars/arrays into plain Python types so msgpack can
    serialise the recorded params without a custom encoder hook."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def record_op(event_type: str, *, capture: Optional[Sequence[str]] = None):
    """Decorate a row-op method so each call emits an Event when a
    RecordingSession is active.

    Outside an active session the decorated method runs normally and emits
    nothing — the same class is usable in unrecorded contexts.

    Args:
        event_type: stored in Event.event_type.
        capture:    optional sequence of parameter names to record.
                    Defaults to every bound argument except `self`.
    """
    def decorator(fn):
        sig = inspect.signature(fn)
        param_names = [p for p in sig.parameters if p != "self"]
        capture_names = list(capture) if capture is not None else param_names

        @wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = _active.get()
            if ctx is None:
                return fn(*args, **kwargs)

            recorder, matrix_provider = ctx

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = {
                name: _to_native(bound.arguments[name])
                for name in capture_names
            }
            recorder.record_event(event_type, params)
            result = fn(*args, **kwargs)
            recorder.maybe_checkpoint(matrix_provider)
            return result

        return wrapper

    return decorator


class RecordingSession:
    """Activate @record_op for the duration of a `with` block.

    The recorder must already be in an active session (i.e. start_session has
    been called) before entering. On clean exit the recorder is closed and the
    resulting summary is exposed via `.summary`.
    """

    def __init__(
        self,
        recorder,
        matrix_provider: Callable,
        *,
        close_on_exit: bool = True,
    ):
        self._recorder = recorder
        self._matrix_provider = matrix_provider
        self._close_on_exit = close_on_exit
        self._token = None
        self.summary: Optional[dict] = None

    def __enter__(self):
        self._token = _active.set((self._recorder, self._matrix_provider))
        return self

    def __exit__(self, exc_type, exc, tb):
        _active.reset(self._token)
        if self._close_on_exit:
            self.summary = self._recorder.close()
        return False
