/**
 * TypeScript port of record_replay/replayer.py.
 *
 * Works on a COO (coordinate) representation in-memory.  Blob downloads are
 * done via the azure-blob helpers; dense conversion is deferred to the API
 * route so this layer stays pure data.
 *
 * Step semantics (identical to Python):
 *   step 0   = initial matrix
 *   step N   = state after N events applied
 *   eventAt(N) = the event that caused transition N-1 → N
 */

import { downloadBlob, downloadBlobOptional, listBlobs } from "./azure-blob";
import { decodeCsr, decodeEvents, CsrMatrix, EventRecord } from "./msgpack-decode";

// ---------------------------------------------------------------------------
// Working matrix representation — dense rows of { col → value } maps
// Mirrors the Python lil_matrix used during event replay.
// ---------------------------------------------------------------------------

type LilMatrix = Map<number, number>[];   // rows[r].get(col) → value

function csrToLil(csr: CsrMatrix): LilMatrix {
  const [nRows] = csr.shape;
  const rows: LilMatrix = Array.from({ length: nRows }, () => new Map());
  for (let r = 0; r < nRows; r++) {
    const start = csr.indptr[r];
    const end   = csr.indptr[r + 1];
    for (let k = start; k < end; k++) {
      rows[r].set(csr.indices[k], csr.data[k]);
    }
  }
  return rows;
}

function lilToCsr(rows: LilMatrix, shape: [number, number]): CsrMatrix {
  const dataArr: number[]  = [];
  const idxArr:  number[]  = [];
  const ptrArr:  number[]  = [0];

  for (const row of rows) {
    const sorted = [...row.entries()].sort(([a], [b]) => a - b);
    for (const [col, val] of sorted) {
      idxArr.push(col);
      dataArr.push(val);
    }
    ptrArr.push(dataArr.length);
  }

  return {
    shape,
    data:    new Float64Array(dataArr),
    indices: new Int32Array(idxArr),
    indptr:  new Int32Array(ptrArr),
  };
}

// ---------------------------------------------------------------------------
// Row operation engine (mirrors _apply_event in replayer.py)
// ---------------------------------------------------------------------------

function applyEvent(rows: LilMatrix, ev: EventRecord): void {
  const p = ev.params;

  switch (ev.event_type) {
    case "swap_rows": {
      const tmp = rows[p.row_i];
      rows[p.row_i] = rows[p.row_j];
      rows[p.row_j] = tmp;
      break;
    }
    case "scale_row": {
      const row = rows[p.row];
      for (const [col, val] of row) row.set(col, val * p.scalar);
      break;
    }
    case "add_scaled_row": {
      const tgt = rows[p.target];
      const src = rows[p.source];
      for (const [col, val] of src) {
        const cur = tgt.get(col) ?? 0;
        const next = cur + p.scalar * val;
        if (Math.abs(next) > 1e-14) tgt.set(col, next);
        else tgt.delete(col);
      }
      break;
    }
    case "xor_rows": {
      // F₂ XOR: symmetric difference of column sets — cols in both cancel to 0
      const tgt = rows[p.target];
      for (const [col] of rows[p.source]) {
        if (tgt.has(col)) tgt.delete(col);
        else tgt.set(col, 1);
      }
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// COO export (what the API routes return to the frontend)
// ---------------------------------------------------------------------------

export interface CooMatrix {
  shape: [number, number];
  row:   number[];
  col:   number[];
  data:  number[];
}

function csrToCoo(csr: CsrMatrix): CooMatrix {
  const row: number[] = [];
  const col: number[] = [];
  const data: number[] = [];
  const [nRows] = csr.shape;

  for (let r = 0; r < nRows; r++) {
    const start = csr.indptr[r];
    const end   = csr.indptr[r + 1];
    for (let k = start; k < end; k++) {
      row.push(r);
      col.push(csr.indices[k]);
      data.push(csr.data[k]);
    }
  }
  return { shape: csr.shape, row, col, data };
}

function diffCells(a: CsrMatrix, b: CsrMatrix): [number, number][] {
  const cooA = csrToCoo(a);
  const cooB = csrToCoo(b);

  const mapA = new Map<string, number>();
  for (let i = 0; i < cooA.row.length; i++) {
    mapA.set(`${cooA.row[i]},${cooA.col[i]}`, cooA.data[i]);
  }

  const changed = new Set<string>();
  for (let i = 0; i < cooB.row.length; i++) {
    const key = `${cooB.row[i]},${cooB.col[i]}`;
    if (mapA.get(key) !== cooB.data[i]) changed.add(key);
    mapA.delete(key);
  }
  // Any key still in mapA had a nonzero in A but zero in B → changed
  for (const key of mapA.keys()) changed.add(key);

  return [...changed].map((k) => k.split(",").map(Number) as [number, number]);
}

// ---------------------------------------------------------------------------
// Session state — one instance per session, cached at module level
// ---------------------------------------------------------------------------

interface SessionState {
  events:      EventRecord[];
  initial:     CsrMatrix;
  checkpoints: Map<number, CsrMatrix>;  // step → matrix (lazy-loaded)
  ckptSteps:   number[];                // sorted checkpoint step numbers on disk
  shape:       [number, number];
}

// Module-level cache: survives across requests in `next start` Node.js mode.
const SESSION_CACHE = new Map<string, SessionState>();

async function loadSession(sessionId: string): Promise<SessionState> {
  if (SESSION_CACHE.has(sessionId)) return SESSION_CACHE.get(sessionId)!;

  const prefix = `sessions/${sessionId}/`;
  const blobs  = await listBlobs(prefix);

  const ckptSteps: number[] = [];
  for (const name of blobs) {
    const base = name.slice(prefix.length);
    const m = base.match(/^checkpoint_(\d+)\.msgpack$/);
    if (m) ckptSteps.push(parseInt(m[1], 10));
  }
  ckptSteps.sort((a, b) => a - b);

  const [initialBuf, commandsBuf] = await Promise.all([
    downloadBlob(`${prefix}initial.msgpack`),
    downloadBlob(`${prefix}commands.msgpack`),
  ]);

  const initial = decodeCsr(initialBuf);
  const events  = decodeEvents(commandsBuf);

  const state: SessionState = {
    events,
    initial,
    checkpoints: new Map(),
    ckptSteps,
    shape: initial.shape,
  };
  SESSION_CACHE.set(sessionId, state);
  return state;
}

async function getCheckpoint(state: SessionState, sessionId: string, step: number): Promise<CsrMatrix> {
  if (step === 0) return state.initial;
  if (state.checkpoints.has(step)) return state.checkpoints.get(step)!;

  const buf = await downloadBlob(`sessions/${sessionId}/checkpoint_${step}.msgpack`);
  const mat = decodeCsr(buf);
  state.checkpoints.set(step, mat);
  return mat;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface StepResult {
  step:         number;
  totalSteps:   number;
  matrix:       CooMatrix;
  changedCells: [number, number][];
  event:        EventRecord | null;
}

export async function getStep(sessionId: string, n: number): Promise<StepResult> {
  const state = await loadSession(sessionId);
  const total = state.events.length;
  n = Math.max(0, Math.min(n, total));

  // Find the largest checkpoint step ≤ n
  let bestCkpt = 0;
  for (const s of state.ckptSteps) {
    if (s <= n) bestCkpt = s;
    else break;
  }

  const base    = await getCheckpoint(state, sessionId, bestCkpt);
  const rows    = csrToLil(base);
  for (let i = bestCkpt; i < n; i++) applyEvent(rows, state.events[i]);
  const curr = lilToCsr(rows, state.shape);

  // Diff against previous step
  let prev: CsrMatrix;
  if (n === 0) {
    prev = curr;
  } else {
    const prevRows = csrToLil(curr);
    // undo last event to get prev state
    const prevBase = await getCheckpoint(state, sessionId, bestCkpt);
    const prevRowsFresh = csrToLil(prevBase);
    for (let i = bestCkpt; i < n - 1; i++) applyEvent(prevRowsFresh, state.events[i]);
    prev = lilToCsr(prevRowsFresh, state.shape);
  }

  return {
    step:         n,
    totalSteps:   total,
    matrix:       csrToCoo(curr),
    changedCells: n === 0 ? [] : diffCells(prev, curr),
    event:        n > 0 ? state.events[n - 1] : null,
  };
}

export interface Annotation {
  code_family?:  string;
  erasure_rate?: number;
  reorder?:      string;
  note?:         string;
}

export async function getSessionInfo(sessionId: string) {
  const [state, metaBuf] = await Promise.all([
    loadSession(sessionId),
    downloadBlobOptional(`sessions/${sessionId}/metadata.json`),
  ]);

  const annotation: Annotation | null = metaBuf
    ? (JSON.parse(metaBuf.toString("utf-8")) as Annotation)
    : null;

  return {
    sessionId,
    totalSteps:   state.events.length,
    nCheckpoints: state.ckptSteps.length,
    shape:        state.shape,
    annotation,
  };
}
