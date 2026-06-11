"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import type { CooMatrix } from "@/lib/replayer";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  matrix: CooMatrix;
  step: number;
  totalSteps: number;
  height?: number;
  staticLabel?: string;    // if set, replaces step info in the title
  initMatrix?: CooMatrix;  // step-0 matrix for fixed layout; falls back to matrix
  rowPerm?: number[];      // rowPerm[matrix_row] = original_node_index
}

// ---------------------------------------------------------------------------
// BCC chain layout
// Ports viewer.py _bcc_chain_layout / browse_bcc.py bipartite_chain_layout
//
// Node IDs:  variable i  → i          (0 … nVars-1)
//            check    j  → nVars + j   (nVars … N-1)
// ---------------------------------------------------------------------------

interface LayoutResult {
  varX: Float64Array;
  chkX: Float64Array;
  cutVars: Set<number>; // original var indices
  cutChks: Set<number>; // original chk indices
  xMin: number;
  xMax: number;
  figWidth: number;
}

function computeBccLayout(
  nVars: number,
  nChks: number,
  edgeRows: number[],
  edgeCols: number[],
  gap = 1.5,
  nodeSpacing = 1.0,
): LayoutResult {
  const N = nVars + nChks;
  const nEdges = edgeRows.length;
  const varX = new Float64Array(nVars);
  const chkX = new Float64Array(nChks);

  if (nEdges === 0) {
    for (let i = 0; i < nVars; i++) varX[i] = i * nodeSpacing;
    for (let j = 0; j < nChks; j++) chkX[j] = j * nodeSpacing;
    const xMax = Math.max(nVars, nChks, 1) * nodeSpacing;
    return { varX, chkX, cutVars: new Set(), cutChks: new Set(), xMin: -1, xMax, figWidth: 600 };
  }

  // Build adjacency list
  type EdgeRef = { to: number; eid: number };
  const adj: EdgeRef[][] = Array.from({ length: N }, () => []);
  const edgeU = new Int32Array(nEdges);
  const edgeV = new Int32Array(nEdges);
  for (let eid = 0; eid < nEdges; eid++) {
    const u = edgeRows[eid];
    const v = nVars + edgeCols[eid];
    edgeU[eid] = u;
    edgeV[eid] = v;
    adj[u].push({ to: v, eid });
    adj[v].push({ to: u, eid });
  }

  // ── Tarjan's BCC (iterative DFS) ─────────────────────────────────────────
  const disc = new Int32Array(N).fill(-1);
  const low  = new Int32Array(N);
  const bccs: Set<number>[] = [];
  const eStack: number[] = [];
  let timer = 0;

  interface Frame { u: number; parentEid: number; adjIdx: number; }

  function dfs(start: number): void {
    disc[start] = low[start] = timer++;
    const stack: Frame[] = [{ u: start, parentEid: -1, adjIdx: 0 }];

    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const { u, parentEid } = frame;
      let pushed = false;

      while (frame.adjIdx < adj[u].length) {
        const { to: v, eid } = adj[u][frame.adjIdx++];
        if (eid === parentEid) continue;

        if (disc[v] === -1) {
          disc[v] = low[v] = timer++;
          eStack.push(eid);
          stack.push({ u: v, parentEid: eid, adjIdx: 0 });
          pushed = true;
          break;
        } else if (disc[v] < disc[u]) {
          // back edge
          eStack.push(eid);
          low[u] = Math.min(low[u], disc[v]);
        }
      }

      if (!pushed) {
        stack.pop();
        if (stack.length > 0) {
          const pu = stack[stack.length - 1].u;
          low[pu] = Math.min(low[pu], low[u]);

          // low[u] >= disc[pu] → pu is an articulation point / BCC boundary
          if (low[u] >= disc[pu]) {
            const bcc = new Set<number>();
            while (true) {
              const e = eStack.pop()!;
              bcc.add(edgeU[e]);
              bcc.add(edgeV[e]);
              if (e === parentEid) break;
            }
            bccs.push(bcc);
          }
        }
      }
    }

    // Remaining edges on stack belong to one more BCC (single-BCC component)
    if (eStack.length > 0) {
      const bcc = new Set<number>();
      while (eStack.length > 0) {
        const e = eStack.pop()!;
        bcc.add(edgeU[e]);
        bcc.add(edgeV[e]);
      }
      bccs.push(bcc);
    }
  }

  for (let u = 0; u < N; u++) {
    if (disc[u] === -1) dfs(u);
  }

  // ── Articulation points = nodes in 2+ BCCs ─────────────────────────────
  const nodeInBccCount = new Array<number>(N).fill(0);
  for (const bcc of bccs) for (const nd of bcc) nodeInBccCount[nd]++;
  const cutNodeSet = new Set<number>();
  for (let u = 0; u < N; u++) if (nodeInBccCount[u] > 1) cutNodeSet.add(u);

  const cutVars = new Set<number>();
  const cutChks = new Set<number>();
  for (const u of cutNodeSet) {
    if (u < nVars) cutVars.add(u);
    else cutChks.add(u - nVars);
  }

  // ── Fallback: no cut nodes ────────────────────────────────────────────
  if (cutNodeSet.size === 0) {
    for (let i = 0; i < nVars; i++) varX[i] = i * nodeSpacing;
    for (let j = 0; j < nChks; j++) chkX[j] = j * nodeSpacing;
    const xMax = Math.max(nVars, nChks, 1) * nodeSpacing;
    return { varX, chkX, cutVars, cutChks, xMin: -1, xMax, figWidth: Math.max(600, xMax * 18) };
  }

  // ── BCC adjacency via cut nodes ───────────────────────────────────────
  const nBcc = bccs.length;
  const nodeToBccs = new Map<number, number[]>();
  for (let i = 0; i < nBcc; i++)
    for (const nd of bccs[i]) {
      if (!nodeToBccs.has(nd)) nodeToBccs.set(nd, []);
      nodeToBccs.get(nd)!.push(i);
    }

  type BccEdge = { nbr: number; cutNode: number };
  const bccAdj: BccEdge[][] = Array.from({ length: nBcc }, () => []);
  for (const cn of cutNodeSet) {
    const idxs = nodeToBccs.get(cn)!;
    for (let a = 0; a < idxs.length; a++)
      for (let b = a + 1; b < idxs.length; b++) {
        bccAdj[idxs[a]].push({ nbr: idxs[b], cutNode: cn });
        bccAdj[idxs[b]].push({ nbr: idxs[a], cutNode: cn });
      }
  }

  // ── DFS on BCC adjacency to get left-to-right order ──────────────────
  // Pre-compute min variable index per BCC to drive ordering.
  const bccMinVar = bccs.map(bcc => {
    let min = Infinity;
    for (const nd of bcc) if (nd < nVars && nd < min) min = nd;
    return min;
  });

  const visitedBcc = new Set<number>();
  const order: Array<{ bccIdx: number; entryCut: number | null }> = [];

  function dfsBcc(idx: number, entryCut: number | null): void {
    if (visitedBcc.has(idx)) return;
    visitedBcc.add(idx);
    order.push({ bccIdx: idx, entryCut });
    const sorted = bccAdj[idx].slice().sort((a, b) => bccMinVar[a.nbr] - bccMinVar[b.nbr]);
    for (const { nbr, cutNode } of sorted)
      if (!visitedBcc.has(nbr)) dfsBcc(nbr, cutNode);
  }
  const bccStart = Array.from({ length: nBcc }, (_, i) => i).sort((a, b) => bccMinVar[a] - bccMinVar[b]);
  for (const i of bccStart) dfsBcc(i, null);

  // ── Assign x positions ────────────────────────────────────────────────
  const placed = new Map<number, number>(); // nodeId → x
  let xCursor = 0;

  function linspace(start: number, end: number, n: number): number[] {
    if (n === 1) return [start];
    return Array.from({ length: n }, (_, k) => start + k * (end - start) / (n - 1));
  }

  for (const { bccIdx, entryCut } of order) {
    const bcc = bccs[bccIdx];

    if (entryCut !== null) {
      if (!placed.has(entryCut)) placed.set(entryCut, xCursor);
      xCursor = placed.get(entryCut)! + gap;
    }

    const varNc: number[] = [];
    const chkNc: number[] = [];
    for (const nd of bcc) {
      if (cutNodeSet.has(nd)) continue;
      if (nd < nVars) varNc.push(nd);
      else chkNc.push(nd - nVars);
    }
    varNc.sort((a, b) => a - b);
    chkNc.sort((a, b) => a - b);

    const nCols = Math.max(varNc.length, chkNc.length, 1);
    const width = (nCols - 1) * nodeSpacing;
    const xStart = xCursor, xEnd = xStart + width;

    linspace(xStart, xEnd, varNc.length).forEach((x, k) => placed.set(varNc[k], x));
    linspace(xStart, xEnd, chkNc.length).forEach((x, k) => placed.set(nVars + chkNc[k], x));

    xCursor = xEnd + gap;

    // Exit cut nodes not yet placed
    for (const nd of [...bcc].sort((a, b) => a - b))
      if (cutNodeSet.has(nd) && !placed.has(nd)) {
        placed.set(nd, xCursor);
        xCursor += gap;
      }
  }

  // Write back to typed arrays
  for (let i = 0; i < nVars; i++) varX[i] = placed.get(i) ?? 0;
  for (let j = 0; j < nChks; j++) chkX[j] = placed.get(nVars + j) ?? 0;

  const allX = [...placed.values()];
  const xMin = Math.min(...allX) - 1;
  const xMax = Math.max(...allX) + 1;
  return { varX, chkX, cutVars, cutChks, xMin, xMax, figWidth: Math.max(600, Math.round((xMax - xMin) * 20)) };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function BccTannerGraph({
  matrix, step, totalSteps, height = 380, staticLabel, initMatrix, rowPerm,
}: Props) {
  const { shape, row, col, data } = matrix;
  const [nRows, nCols] = shape;
  const nVars = nRows;
  const nChks = nCols - 1; // strip augmented syndrome column

  // ── Degrees and edges from the current matrix ───────────────────────────
  const rowDeg = new Array<number>(nVars).fill(0);
  const colDeg = new Array<number>(nChks).fill(0);
  const edgeRows: number[] = [];
  const edgeCols: number[] = [];

  for (let i = 0; i < row.length; i++) {
    const r = row[i], c = col[i];
    if (c >= nChks || data[i] === 0) continue;
    rowDeg[r]++;
    colDeg[c]++;
    edgeRows.push(r);
    edgeCols.push(c);
  }

  // ── Fixed layout from initMatrix (or current matrix as fallback) ────────
  // layoutSrc is stable between steps when initMatrix is provided (SWR caches
  // the step-0 response and returns the same object reference each render).
  const layoutSrc = initMatrix ?? matrix;
  const { varX, chkX, cutVars, cutChks, xMin, xMax, figWidth } = useMemo(() => {
    const sNVars = layoutSrc.shape[0];
    const sNChks = layoutSrc.shape[1] - 1;
    const sEdgeRows: number[] = [];
    const sEdgeCols: number[] = [];
    for (let i = 0; i < layoutSrc.row.length; i++) {
      const r = layoutSrc.row[i], c = layoutSrc.col[i];
      if (c < sNChks && layoutSrc.data[i] !== 0) {
        sEdgeRows.push(r);
        sEdgeCols.push(c);
      }
    }
    return computeBccLayout(sNVars, sNChks, sEdgeRows, sEdgeCols);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layoutSrc]);

  // ── origToCurr[original_node_j] = current_matrix_row ────────────────────
  // Needed to look up live degree for a node identified by original index j.
  const origToCurr = useMemo(() => {
    if (!rowPerm) return null;
    const arr = new Int32Array(nVars);
    for (let r = 0; r < nVars; r++) arr[rowPerm[r]] = r;
    return arr;
  }, [rowPerm, nVars]);

  // Helper: degree of original node j in the current matrix.
  const varDeg = (j: number) => origToCurr ? rowDeg[origToCurr[j]] : rowDeg[j];

  // ── Edge trace ───────────────────────────────────────────────────────────
  // For each edge (current matrix row r, col c), the original node is
  // rowPerm[r].  Its x position in the fixed layout is varX[rowPerm[r]].
  const edgeX: (number | null)[] = [];
  const edgeY: (number | null)[] = [];
  for (let i = 0; i < edgeRows.length; i++) {
    const r = edgeRows[i];
    const origR = rowPerm ? rowPerm[r] : r;
    edgeX.push(varX[origR], chkX[edgeCols[i]], null);
    edgeY.push(1, 0, null);
  }

  // ── Node partitions — iterate over original node indices (layout keys) ──
  const normalVarIdx = Array.from({ length: nVars }, (_, j) => j).filter(j => !cutVars.has(j));
  const cutVarIdx    = [...cutVars];
  const normalChkIdx = Array.from({ length: nChks }, (_, j) => j).filter(j => !cutChks.has(j));
  const cutChkIdx    = [...cutChks];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const traces: any[] = [];

  if (edgeX.length > 0) {
    traces.push({
      type: "scatter", x: edgeX, y: edgeY, mode: "lines",
      line: { color: "rgba(160,160,210,0.20)", width: 0.8 },
      hoverinfo: "skip", showlegend: false,
    });
  }

  // Normal variable nodes (blue circles, y=1)
  if (normalVarIdx.length > 0) {
    traces.push({
      type: "scatter",
      x: normalVarIdx.map(j => varX[j]),
      y: normalVarIdx.map(() => 1),
      mode: "markers",
      marker: { size: normalVarIdx.map(j => 5 + varDeg(j)), color: "#4c8bf5", line: { width: 0.5, color: "#2a5fc4" } },
      customdata: normalVarIdx.map(j => [j, varDeg(j)]),
      hovertemplate: "v%{customdata[0]}  deg %{customdata[1]}<extra></extra>",
      name: "variable",
    });
  }

  // Cut variable nodes (orange circles, y=1, larger)
  if (cutVarIdx.length > 0) {
    traces.push({
      type: "scatter",
      x: cutVarIdx.map(j => varX[j]),
      y: cutVarIdx.map(() => 1),
      mode: "markers",
      marker: { size: cutVarIdx.map(j => 9 + varDeg(j)), color: "#f39c12", line: { width: 1.5, color: "#fff" } },
      customdata: cutVarIdx.map(j => [j, varDeg(j)]),
      hovertemplate: "v%{customdata[0]}  deg %{customdata[1]}  (cut)<extra></extra>",
      name: "variable (cut)",
    });
  }

  // Normal check nodes (red squares, y=0) — no column permutation
  if (normalChkIdx.length > 0) {
    traces.push({
      type: "scatter",
      x: normalChkIdx.map(j => chkX[j]),
      y: normalChkIdx.map(() => 0),
      mode: "markers",
      marker: { size: normalChkIdx.map(j => 5 + colDeg[j]), color: "#e74c3c", symbol: "square", line: { width: 0.5, color: "#c0392b" } },
      customdata: normalChkIdx.map(j => [j, colDeg[j]]),
      hovertemplate: "c%{customdata[0]}  deg %{customdata[1]}<extra></extra>",
      name: "check",
    });
  }

  // Cut check nodes (orange squares, y=0, larger)
  if (cutChkIdx.length > 0) {
    traces.push({
      type: "scatter",
      x: cutChkIdx.map(j => chkX[j]),
      y: cutChkIdx.map(() => 0),
      mode: "markers",
      marker: { size: cutChkIdx.map(j => 9 + colDeg[j]), color: "#f39c12", symbol: "square", line: { width: 1.5, color: "#fff" } },
      customdata: cutChkIdx.map(j => [j, colDeg[j]]),
      hovertemplate: "c%{customdata[0]}  deg %{customdata[1]}  (cut)<extra></extra>",
      name: "check (cut)",
    });
  }

  const nCuts = cutVarIdx.length + cutChkIdx.length;
  const titleText = staticLabel
    ? `${staticLabel}  ·  ${nVars} var  ${nChks} chk  ·  ${nCuts} cut node(s)`
    : `BCC Tanner Graph — Step ${step} / ${totalSteps}  ·  ${nVars} var  ${nChks} chk  ·  ${edgeRows.length} edges`;

  return (
    <div style={{ overflowX: "auto", width: "100%", textAlign: "center" }}>
      <div style={{ display: "inline-block" }}>
        <Plot
          data={traces}
          layout={{
            title: { text: titleText, x: 0.5, font: { size: 12, color: "#ddd" } },
            annotations: [
              {
                x: 0.5, y: 1.18, xref: "paper", yref: "paper",
                text: "▲ Variable nodes (rows)", showarrow: false,
                font: { color: "#4c8bf5", size: 10 }, xanchor: "center",
              },
              {
                x: 0.5, y: -0.13, xref: "paper", yref: "paper",
                text: "▼ Check nodes (cols)", showarrow: false,
                font: { color: "#e74c3c", size: 10 }, xanchor: "center",
              },
            ],
            margin: { l: 20, r: 20, t: 60, b: 40 },
            height,
            width: figWidth,
            autosize: false,
            paper_bgcolor: "#16213e",
            plot_bgcolor: "#16213e",
            font: { color: "#ccc" },
            showlegend: true,
            legend: { orientation: "h", x: 0.5, xanchor: "center", y: 1.14 },
            xaxis: { range: [xMin, xMax], showticklabels: false, showgrid: false, zeroline: false },
            yaxis: { range: [-0.3, 1.3], showticklabels: false, showgrid: false, zeroline: false },
          }}
          config={{ displayModeBar: false }}
          style={{ width: figWidth, height: height + 10 }}
        />
      </div>
    </div>
  );
}
