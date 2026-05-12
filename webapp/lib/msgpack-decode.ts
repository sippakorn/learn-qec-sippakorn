/**
 * Decodes msgpack buffers produced by the Python record_replay/storage.py.
 *
 * Python encodes numpy arrays as { __ndarray__: true, dtype, shape, data: bytes }
 * and scipy CSR matrices as { __csr__: true, shape, data, indices, indptr }.
 * This mirrors the _decode() function from storage.py.
 */

import { decode } from "@msgpack/msgpack";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CsrMatrix {
  shape: [number, number];
  data: Float64Array;
  indices: Int32Array;
  indptr: Int32Array;
}

// Raw decoded shapes from msgpack
interface RawNdarray {
  __ndarray__: boolean;
  dtype: string;
  shape: number[];
  data: Uint8Array;
}

interface RawCsr {
  __csr__: boolean;
  shape: number[];
  data: unknown;
  indices: unknown;
  indptr: unknown;
}

// ---------------------------------------------------------------------------
// Ndarray reconstruction
// ---------------------------------------------------------------------------

const DTYPE_MAP: Record<string, new (buf: ArrayBuffer) => ArrayLike<number>> = {
  float64: Float64Array,
  float32: Float32Array,
  int64:   BigInt64Array as unknown as new (buf: ArrayBuffer) => ArrayLike<number>,
  int32:   Int32Array,
  int16:   Int16Array,
  int8:    Int8Array,
  uint64:  BigUint64Array as unknown as new (buf: ArrayBuffer) => ArrayLike<number>,
  uint32:  Uint32Array,
  uint16:  Uint16Array,
  uint8:   Uint8Array,
  bool:    Uint8Array,
};

function decodeNdarray(raw: RawNdarray): ArrayLike<number> {
  const ctor = DTYPE_MAP[raw.dtype];
  if (!ctor) throw new Error(`Unsupported dtype: ${raw.dtype}`);
  const buf = raw.data.buffer.slice(
    raw.data.byteOffset,
    raw.data.byteOffset + raw.data.byteLength
  ) as ArrayBuffer;
  return new ctor(buf);
}

// ---------------------------------------------------------------------------
// Recursive decoder (mirrors Python's _decode)
// ---------------------------------------------------------------------------

function decodeValue(obj: unknown): unknown {
  if (obj === null || typeof obj !== "object") return obj;

  if (obj instanceof Uint8Array) return obj;

  const o = obj as Record<string, unknown>;

  if (o["__ndarray__"]) return decodeNdarray(o as unknown as RawNdarray);

  if (o["__csr__"]) {
    const raw = o as unknown as RawCsr;
    const shape = raw.shape as number[];
    return {
      __csr__: true,
      shape:   [shape[0], shape[1]] as [number, number],
      data:    decodeValue(raw.data),
      indices: decodeValue(raw.indices),
      indptr:  decodeValue(raw.indptr),
    };
  }

  if (Array.isArray(obj)) return obj.map(decodeValue);

  return Object.fromEntries(
    Object.entries(o).map(([k, v]) => [k, decodeValue(v)])
  );
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

export function decodeCsr(buffer: Buffer): CsrMatrix {
  const raw = decode(buffer) as unknown;
  const decoded = decodeValue(raw) as {
    __csr__: true;
    shape: [number, number];
    data: ArrayLike<number>;
    indices: ArrayLike<number>;
    indptr: ArrayLike<number>;
  };

  return {
    shape:   decoded.shape,
    data:    Float64Array.from(decoded.data as unknown as Iterable<number>),
    indices: Int32Array.from(decoded.indices as unknown as Iterable<number>),
    indptr:  Int32Array.from(decoded.indptr as unknown as Iterable<number>),
  };
}

export function decodeEvents(buffer: Buffer): EventRecord[] {
  const raw = decode(buffer) as unknown[];
  return (raw as Record<string, unknown>[]).map((e) => ({
    event_id:   e["event_id"]   as number,
    event_type: e["event_type"] as string,
    params:     e["params"]     as Record<string, number>,
    timestamp:  e["timestamp"]  as number,
    step:       e["step"]       as number,
  }));
}

export interface EventRecord {
  event_id:   number;
  event_type: string;
  params:     Record<string, number>;
  timestamp:  number;
  step:       number;
}
