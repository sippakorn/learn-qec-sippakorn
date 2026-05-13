"use client";

import dynamic from "next/dynamic";
import type { CooMatrix } from "@/lib/replayer";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  matrix: CooMatrix;
  step: number;
  totalSteps: number;
}

// ---------------------------------------------------------------------------
// Layout: Union-Find to detect components, then spread nodes horizontally
// Variable nodes → y=1 (top), Check nodes → y=0 (bottom)
// ---------------------------------------------------------------------------

function computeLayout(nVars: number, nChks: number, edgeRows: number[], edgeCols: number[]) {
  const total = nVars + nChks;
  const parent = Array.from({ length: total }, (_, i) => i);

  function find(x: number): number {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  function union(a: number, b: number) {
    a = find(a); b = find(b);
    if (a !== b) parent[a] = b;
  }

  for (let i = 0; i < edgeRows.length; i++) union(edgeRows[i], nVars + edgeCols[i]);

  const compVars = new Map<number, number[]>();
  const compChks = new Map<number, number[]>();
  for (let i = 0; i < nVars; i++) {
    const r = find(i);
    if (!compVars.has(r)) compVars.set(r, []);
    compVars.get(r)!.push(i);
  }
  for (let j = 0; j < nChks; j++) {
    const r = find(nVars + j);
    if (!compChks.has(r)) compChks.set(r, []);
    compChks.get(r)!.push(j);
  }

  const allRoots = new Set([...compVars.keys(), ...compChks.keys()]);
  const components = [...allRoots].sort((a, b) => {
    const sa = (compVars.get(a)?.length ?? 0) + (compChks.get(a)?.length ?? 0);
    const sb = (compVars.get(b)?.length ?? 0) + (compChks.get(b)?.length ?? 0);
    return sb - sa;
  });

  const GAP = 3;
  const varX = new Float64Array(nVars);
  const chkX = new Float64Array(nChks);
  let xOffset = 0;

  for (const root of components) {
    const vars = (compVars.get(root) ?? []).slice().sort((a, b) => a - b);
    const chks = (compChks.get(root) ?? []).slice().sort((a, b) => a - b);
    const nV = vars.length, nC = chks.length;
    const band = Math.max(nV, nC, 1);

    vars.forEach((i, k) => {
      varX[i] = xOffset + (nV > 1 ? (k * (band - 1)) / (nV - 1) : (band - 1) / 2);
    });
    chks.forEach((j, k) => {
      chkX[j] = xOffset + (nC > 1 ? (k * (band - 1)) / (nC - 1) : (band - 1) / 2);
    });

    xOffset += band + GAP;
  }

  const totalSpan = xOffset - GAP;
  const figWidth  = Math.max(800, Math.round(totalSpan * 18));
  return { varX, chkX, totalSpan, figWidth };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TannerGraph({ matrix, step, totalSteps }: Props) {
  const { shape, row, col, data } = matrix;
  const [nRows, nCols] = shape;
  const nVars = nRows;
  const nChks = nCols - 1; // strip augmented syndrome column

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

  const { varX, chkX, totalSpan, figWidth } = computeLayout(nVars, nChks, edgeRows, edgeCols);

  // Edge trace
  const edgeX: (number | null)[] = [];
  const edgeY: (number | null)[] = [];
  for (let i = 0; i < edgeRows.length; i++) {
    edgeX.push(varX[edgeRows[i]], chkX[edgeCols[i]], null);
    edgeY.push(1, 0, null);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const traces: any[] = [];

  if (edgeX.length > 0) {
    traces.push({
      type: "scatter", x: edgeX, y: edgeY,
      mode: "lines",
      line: { color: "rgba(160,160,210,0.22)", width: 0.8 },
      hoverinfo: "skip", showlegend: false,
    });
  }

  // Variable nodes — top (y=1)
  traces.push({
    type: "scatter",
    x: Array.from(varX),
    y: Array<number>(nVars).fill(1),
    mode: "markers",
    marker: { size: rowDeg.map((d) => 5 + d), color: "#4c8bf5", line: { width: 0.5, color: "#2a5fc4" } },
    customdata: Array.from({ length: nVars }, (_, i) => [i, rowDeg[i]]),
    hovertemplate: "v%{customdata[0]}  deg %{customdata[1]}<extra></extra>",
    name: "variable nodes",
  });

  // Check nodes — bottom (y=0)
  traces.push({
    type: "scatter",
    x: Array.from(chkX),
    y: Array<number>(nChks).fill(0),
    mode: "markers",
    marker: { size: colDeg.map((d) => 5 + d), color: "#e74c3c", line: { width: 0.5, color: "#c0392b" } },
    customdata: Array.from({ length: nChks }, (_, j) => [j, colDeg[j]]),
    hovertemplate: "c%{customdata[0]}  deg %{customdata[1]}<extra></extra>",
    name: "check nodes",
  });

  return (
    <div style={{ overflowX: "auto", width: "100%" }}>
      <Plot
        data={traces}
        layout={{
          title: {
            text: `Tanner Graph — Step ${step} / ${totalSteps}  ·  ${nVars} var  ${nChks} chk  ${edgeRows.length} edges`,
            x: 0.5, font: { size: 13, color: "#ddd" },
          },
          annotations: [
            {
              x: 0.5, y: 1.13, xref: "paper", yref: "paper",
              text: "▲ Variable nodes (rows)", showarrow: false,
              font: { color: "#4c8bf5", size: 11 }, xanchor: "center",
            },
            {
              x: 0.5, y: -0.09, xref: "paper", yref: "paper",
              text: "▼ Check nodes (cols)", showarrow: false,
              font: { color: "#e74c3c", size: 11 }, xanchor: "center",
            },
          ],
          margin: { l: 20, r: 20, t: 70, b: 40 },
          height: 500,
          width: figWidth,
          autosize: false,
          paper_bgcolor: "#16213e",
          plot_bgcolor: "#16213e",
          font: { color: "#ccc" },
          showlegend: true,
          legend: { orientation: "h", x: 0.5, xanchor: "center", y: 1.08 },
          xaxis: { range: [-1, totalSpan + 1], showticklabels: false, showgrid: false, zeroline: false },
          yaxis: { range: [-0.3, 1.3], showticklabels: false, showgrid: false, zeroline: false },
        }}
        config={{ displayModeBar: false }}
        style={{ width: figWidth, height: 510 }}
      />
    </div>
  );
}
