"use client";

import dynamic from "next/dynamic";
import type { CooMatrix } from "@/lib/replayer";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  matrix: CooMatrix;
  changedCells: [number, number][];
  step: number;
  totalSteps: number;
  rowPerm?: number[];   // rowPerm[matrix_row] = original_node_index
}

const LABEL_MARGIN = 52;  // px reserved for y-axis tick labels

export default function MatrixHeatmap({ matrix, changedCells, step, totalSteps, rowPerm }: Props) {
  const { shape, row, col, data } = matrix;
  const [nRows, nCols] = shape;

  // Square cells: scale so the larger dimension fills MAX_SIDE plot-area pixels
  const MAX_SIDE = 300;
  const cellPx   = MAX_SIDE / Math.max(nRows, nCols);
  const plotW    = Math.max(40, Math.round(nCols * cellPx * 1.5));  // 1.5× wider
  const plotH    = Math.max(40, Math.round(nRows * cellPx));
  const lMargin  = rowPerm ? LABEL_MARGIN : 10;
  const figWidth  = plotW + lMargin + 10;  // l + r margins
  const figHeight = plotH + 50;            // margin t=40, b=10

  // Y-axis tick labels: "v{j}" for identity rows, "v{j} ←" for swapped rows.
  const tickText = rowPerm
    ? Array.from({ length: nRows }, (_, i) =>
        rowPerm[i] === i ? `v${i}` : `v${rowPerm[i]} ←`
      )
    : null;

  // Build dense grid for Plotly (null = zero)
  const z: (number | null)[][] = Array.from({ length: nRows }, () =>
    Array(nCols).fill(null)
  );
  for (let i = 0; i < row.length; i++) z[row[i]][col[i]] = data[i];

  // Diff overlay — amber tint on changed cells
  const diffZ: (number | null)[][] = Array.from({ length: nRows }, () =>
    Array(nCols).fill(null)
  );
  for (const [r, c] of changedCells) diffZ[r][c] = 1;

  return (
    <div style={{ overflowX: "auto", textAlign: "center" }}>
      <div style={{ display: "inline-block" }}>
      <Plot
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        data={[
          {
            type: "heatmap",
            z,
            colorscale: "RdBu",
            zmid: 0,
            showscale: true,
            colorbar: { title: { text: "Value" }, thickness: 12, len: 0.8 },
            name: "Matrix",
            hovertemplate: "row %{y}  col %{x}<br>value: %{z:.5g}<extra></extra>",
          },
          {
            type: "heatmap",
            z: diffZ,
            colorscale: [
              [0, "rgba(0,0,0,0)"],
              [1, "rgba(255,185,0,0.45)"],
            ],
            showscale: false,
            name: "Changed",
            hoverinfo: "skip",
            zmin: 0,
            zmax: 1,
          },
        ] as any[]}
        layout={{
          title: {
            text: `Step ${step} / ${totalSteps}`,
            x: 0.5,
            font: { size: 14, color: "#ddd" },
          },
          margin: { l: lMargin, r: 10, t: 40, b: 10 },
          height: figHeight,
          width: figWidth,
          autosize: false,
          paper_bgcolor: "#16213e",
          plot_bgcolor: "#16213e",
          font: { color: "#ccc" },
          xaxis: { showticklabels: false, showgrid: false, zeroline: false },
          yaxis: tickText
            ? {
                tickmode: "array",
                tickvals: Array.from({ length: nRows }, (_, i) => i),
                ticktext: tickText,
                tickfont: { size: 9, color: "#aaa" },
                showticklabels: true,
                showgrid: false,
                zeroline: false,
                autorange: "reversed",
              }
            : { showticklabels: false, showgrid: false, zeroline: false, autorange: "reversed" },
        }}
        config={{ displayModeBar: false }}
        style={{ width: figWidth, height: figHeight + 10 }}
      />
      </div>
    </div>
  );
}
