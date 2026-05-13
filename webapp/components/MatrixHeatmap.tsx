"use client";

import dynamic from "next/dynamic";
import type { CooMatrix } from "@/lib/replayer";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  matrix: CooMatrix;
  changedCells: [number, number][];
  step: number;
  totalSteps: number;
}

export default function MatrixHeatmap({ matrix, changedCells, step, totalSteps }: Props) {
  const { shape, row, col, data } = matrix;
  const [nRows, nCols] = shape;

  // Build dense grid for Plotly (null = zero, keeps zeros transparent-ish)
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
        margin: { l: 10, r: 10, t: 40, b: 10 },
        height: 500,
        paper_bgcolor: "#16213e",
        plot_bgcolor: "#16213e",
        font: { color: "#ccc" },
        xaxis: { showticklabels: false, showgrid: false, zeroline: false },
        yaxis: { showticklabels: false, showgrid: false, zeroline: false, autorange: "reversed" },
      }}
      config={{ displayModeBar: false }}
      style={{ width: "100%", height: "510px" }}
    />
  );
}
