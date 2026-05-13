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

export default function TannerGraph({ matrix, step, totalSteps }: Props) {
  const { shape, row, col, data } = matrix;
  const [nRows, nCols] = shape;
  const nVars = nRows;
  const nChks = nCols - 1; // strip last column (augmented syndrome)

  // Degree arrays
  const rowDeg = new Array<number>(nVars).fill(0);
  const colDeg = new Array<number>(nChks).fill(0);

  // Collect edges (row i → variable node, col j → check node)
  const edgeRowIdx: number[] = [];
  const edgeColIdx: number[] = [];
  for (let i = 0; i < row.length; i++) {
    const r = row[i];
    const c = col[i];
    if (c >= nChks || data[i] === 0) continue; // skip syndrome col and zeros
    rowDeg[r]++;
    colDeg[c]++;
    edgeRowIdx.push(r);
    edgeColIdx.push(c);
  }

  // Normalised y positions: both sides span [0, 1] regardless of count difference
  const varYPos = Array.from({ length: nVars }, (_, i) =>
    nVars > 1 ? i / (nVars - 1) : 0.5
  );
  const chkYPos = Array.from({ length: nChks }, (_, j) =>
    nChks > 1 ? j / (nChks - 1) : 0.5
  );

  // Edge trace (lines with null separators)
  const edgeX: (number | null)[] = [];
  const edgeY: (number | null)[] = [];
  for (let i = 0; i < edgeRowIdx.length; i++) {
    edgeX.push(0, 1, null);
    edgeY.push(varYPos[edgeRowIdx[i]], chkYPos[edgeColIdx[i]], null);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const traces: any[] = [];

  if (edgeX.length > 0) {
    traces.push({
      type: "scatter",
      x: edgeX,
      y: edgeY,
      mode: "lines",
      line: { color: "rgba(160,160,210,0.22)", width: 0.8 },
      hoverinfo: "skip",
      showlegend: false,
    });
  }

  // Variable nodes — left column x=0
  traces.push({
    type: "scatter",
    x: Array<number>(nVars).fill(0),
    y: varYPos,
    mode: "markers",
    marker: {
      size: rowDeg.map((d) => 5 + d),
      color: "#4c8bf5",
      line: { width: 0.5, color: "#2a5fc4" },
    },
    customdata: Array.from({ length: nVars }, (_, i) => [i, rowDeg[i]]),
    hovertemplate: "v%{customdata[0]}  deg %{customdata[1]}<extra></extra>",
    name: "variable nodes",
  });

  // Check nodes — right column x=1
  traces.push({
    type: "scatter",
    x: Array<number>(nChks).fill(1),
    y: chkYPos,
    mode: "markers",
    marker: {
      size: colDeg.map((d) => 5 + d),
      color: "#e74c3c",
      line: { width: 0.5, color: "#c0392b" },
    },
    customdata: Array.from({ length: nChks }, (_, j) => [j, colDeg[j]]),
    hovertemplate: "c%{customdata[0]}  deg %{customdata[1]}<extra></extra>",
    name: "check nodes",
  });

  return (
    <Plot
      data={traces}
      layout={{
        title: {
          text: `Tanner Graph — Step ${step} / ${totalSteps}  ·  ${nVars} var  ${nChks} chk  ${edgeRowIdx.length} edges`,
          x: 0.5,
          font: { size: 13, color: "#ddd" },
        },
        annotations: [
          {
            x: 0, y: 1.08, xref: "paper", yref: "paper",
            text: "Variable nodes (rows)",
            showarrow: false,
            font: { color: "#4c8bf5", size: 11 },
            xanchor: "center",
          },
          {
            x: 1, y: 1.08, xref: "paper", yref: "paper",
            text: "Check nodes (cols)",
            showarrow: false,
            font: { color: "#e74c3c", size: 11 },
            xanchor: "center",
          },
        ],
        margin: { l: 20, r: 20, t: 70, b: 20 },
        height: 500,
        paper_bgcolor: "#16213e",
        plot_bgcolor: "#16213e",
        font: { color: "#ccc" },
        showlegend: true,
        legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.04 },
        xaxis: {
          range: [-0.15, 1.15],
          showticklabels: false,
          showgrid: false,
          zeroline: false,
        },
        yaxis: {
          showticklabels: false,
          showgrid: false,
          zeroline: false,
          autorange: "reversed",
        },
      }}
      config={{ displayModeBar: false }}
      style={{ width: "100%", height: "510px" }}
    />
  );
}
