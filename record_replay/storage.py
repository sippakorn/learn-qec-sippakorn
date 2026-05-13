"""MessagePack serialization for scipy CSR matrices and event logs.

Numpy arrays are encoded as dicts carrying dtype, shape, and raw bytes so that
round-tripping through msgpack is lossless for any numeric dtype.
"""

import numpy as np
import scipy.sparse as sp
import msgpack


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_ndarray(arr: np.ndarray) -> dict:
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def _encode_csr(matrix: sp.csr_matrix) -> dict:
    return {
        "__csr__": True,
        "shape": list(matrix.shape),
        "data": _encode_ndarray(matrix.data),
        "indices": _encode_ndarray(matrix.indices),
        "indptr": _encode_ndarray(matrix.indptr),
    }


def _msgpack_default(obj):
    """Called by msgpack for objects it cannot serialise natively."""
    if isinstance(obj, np.ndarray):
        return _encode_ndarray(obj)
    if sp.issparse(obj):
        return _encode_csr(obj.tocsr())
    raise TypeError(f"Cannot encode type {type(obj)!r}")


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def _decode(obj):
    """Recursively decode msgpack-decoded structures back to numpy/scipy."""
    if not isinstance(obj, dict):
        return obj
    if obj.get("__ndarray__"):
        arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(obj["shape"]) if obj["shape"] else arr
    if obj.get("__csr__"):
        data = _decode(obj["data"])
        indices = _decode(obj["indices"])
        indptr = _decode(obj["indptr"])
        shape = tuple(obj["shape"])
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    return {k: _decode(v) for k, v in obj.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_msgpack(path, data) -> None:
    packed = msgpack.packb(data, default=_msgpack_default, use_bin_type=True)
    with open(path, "wb") as fh:
        fh.write(packed)


def read_msgpack(path):
    with open(path, "rb") as fh:
        raw = msgpack.unpackb(fh.read(), raw=False)
    return _decode(raw)
