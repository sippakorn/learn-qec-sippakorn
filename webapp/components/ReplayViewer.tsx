"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import MatrixHeatmap from "./MatrixHeatmap";
import TannerGraph from "./TannerGraph";
import BccTannerGraph from "./BccTannerGraph";
import EventInfoPanel from "./EventInfoPanel";
import PlaybackControls from "./PlaybackControls";
import SessionAnnotation from "./SessionAnnotation";
import type { SessionMeta } from "@/lib/azure-table";
import type { Annotation, StepResult } from "@/lib/replayer";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

interface Props {
  sessions: SessionMeta[];
}

export default function ReplayViewer({ sessions }: Props) {
  const [sessionId,   setSessionId]   = useState(sessions[0]?.sessionId ?? "");
  const [step,        setStep]        = useState(0);
  const [playing,     setPlaying]     = useState(false);
  const [speed,       setSpeed]       = useState(1000);
  const [viewMode,    setViewMode]    = useState<"matrix" | "graph">("matrix");
  const [previewMode, setPreviewMode] = useState<"single" | "dual">("dual");
  const intervalRef                   = useRef<ReturnType<typeof setInterval> | null>(null);

  // Session metadata (total_steps, shape, annotation)
  const { data: info } = useSWR<{
    totalSteps:   number;
    nCheckpoints: number;
    shape:        [number, number];
    annotation:   Annotation | null;
  }>(
    sessionId ? `/api/sessions/${sessionId}/info` : null,
    fetcher
  );

  // Step data
  const { data: stepData, isLoading } = useSWR<StepResult>(
    sessionId ? `/api/sessions/${sessionId}/step/${step}` : null,
    fetcher,
    { keepPreviousData: true }
  );

  // Step-0 data for static BCC panel (H_active before GE starts)
  const { data: bccInitData } = useSWR<StepResult>(
    sessionId ? `/api/sessions/${sessionId}/step/0` : null,
    fetcher,
  );

  const totalSteps = info?.totalSteps ?? 1;

  // Reset step when session changes
  useEffect(() => { setStep(0); setPlaying(false); }, [sessionId]);

  // Always show heatmap in left slot when switching to Dual
  useEffect(() => { if (previewMode === "dual") setViewMode("matrix"); }, [previewMode]);

  // Auto-advance playback
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (!playing) return;

    intervalRef.current = setInterval(() => {
      setStep((s) => {
        if (s >= totalSteps) { setPlaying(false); return s; }
        return s + 1;
      });
    }, speed);

    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, totalSteps]);

  const handleStep = useCallback((n: number) => { setStep(n); setPlaying(false); }, []);

  return (
    <div className="min-h-screen bg-[#0f0e17] text-gray-300 font-mono p-4 space-y-3">

      {/* Top bar */}
      <div className="flex items-end gap-5 flex-wrap">
        <div>
          <div className="text-xs text-gray-500 mb-1">Session</div>
          <select
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            className="bg-[#16213e] text-gray-300 border border-[#334] rounded px-2 py-1 text-sm w-52"
          >
            {sessions.map((s) => (
              <option key={s.sessionId} value={s.sessionId}>{s.sessionId}</option>
            ))}
          </select>
        </div>

        <div>
          <div className="text-xs text-gray-500 mb-1">Preview</div>
          <select
            value={previewMode}
            onChange={(e) => setPreviewMode(e.target.value as "single" | "dual")}
            className="bg-[#16213e] text-gray-300 border border-[#334] rounded px-2 py-1 text-sm"
          >
            <option value="single">Single</option>
            <option value="dual">Dual</option>
          </select>
        </div>

        {info && (
          <div className="text-xs text-gray-500 pb-1">
            {info.totalSteps.toLocaleString()} steps · {info.nCheckpoints} checkpoints ·{" "}
            {info.shape[0]}×{info.shape[1]} matrix
          </div>
        )}

        {isLoading && (
          <div className="text-xs text-blue-400 pb-1 animate-pulse">loading…</div>
        )}
      </div>

      {/* Main panel — always 2 columns (heatmap + sidebar) */}
      <div className="grid gap-3" style={{ gridTemplateColumns: "1fr 260px" }}>

        {/* Left: heatmap always; in Single mode can toggle to Tanner */}
        <div className="bg-[#16213e] rounded-md p-3 overflow-x-auto">
          {stepData ? (
            previewMode === "dual" || viewMode === "matrix" ? (
              <MatrixHeatmap
                matrix={stepData.matrix}
                changedCells={stepData.changedCells}
                step={stepData.step}
                totalSteps={stepData.totalSteps}
                rowPerm={stepData.rowPerm}
              />
            ) : (
              <TannerGraph
                matrix={stepData.matrix}
                step={stepData.step}
                totalSteps={stepData.totalSteps}
              />
            )
          ) : (
            <div className="h-40 flex items-center justify-center text-gray-600">
              {sessionId ? "Loading…" : "Select a session"}
            </div>
          )}
        </div>

        {/* Event info panel */}
        <div className="bg-[#16213e] rounded-md p-3 overflow-y-auto max-h-[540px]">
          <div className="font-bold text-gray-400 mb-3 text-sm">Event info</div>
          <EventInfoPanel event={stepData?.event ?? null} />
          <hr className="border-[#2a2a4a] my-4" />
          <div className="text-xs text-gray-500 mb-1">Diff legend</div>
          <div className="text-xs text-gray-600">
            <span className="text-yellow-400">■</span> Changed since previous step
          </div>
          <hr className="border-[#2a2a4a] my-4" />
          <div className="font-bold text-gray-400 mb-2 text-sm">Session annotation</div>
          <SessionAnnotation annotation={info?.annotation ?? null} />
        </div>
      </div>

      {/* Playback controls */}
      <div className="bg-[#16213e] rounded-md p-3">
        <PlaybackControls
          step={step}
          totalSteps={totalSteps}
          playing={playing}
          speed={speed}
          viewMode={viewMode}
          previewMode={previewMode}
          onStep={handleStep}
          onTogglePlay={() => setPlaying((p) => !p)}
          onSpeedChange={setSpeed}
          onToggleView={() => setViewMode((m) => m === "matrix" ? "graph" : "matrix")}
        />
      </div>

      {/* BCC Tanner graph — Dual mode only, updates with each replay step */}
      {previewMode === "dual" && (
        <div className="bg-[#16213e] rounded-md p-3 overflow-x-auto">
          {stepData ? (
            <BccTannerGraph
              matrix={stepData.matrix}
              step={stepData.step}
              totalSteps={stepData.totalSteps}
              height={380}
              initMatrix={bccInitData?.matrix}
              rowPerm={stepData.rowPerm}
            />
          ) : (
            <div className="h-40 flex items-center justify-center text-gray-600">Loading…</div>
          )}
        </div>
      )}

      {/* Static BCC panel — Dual mode only, fixed at initial H_active (step 0) */}
      {previewMode === "dual" && (
        <div className="bg-[#16213e] rounded-md p-3 overflow-x-auto">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-500">BCC state · initial H_active</span>
            <a
              href={`/api/sessions/${sessionId}/bcc`}
              download={`h_active_${sessionId}.npz`}
              className="text-xs px-3 py-1 rounded border border-[#334] bg-[#1e2d50] text-gray-300 hover:bg-[#2a3f6e] hover:text-white transition-colors"
            >
              ⬇ Download .npz
            </a>
          </div>
          {bccInitData ? (
            <BccTannerGraph
              matrix={bccInitData.matrix}
              step={0}
              totalSteps={totalSteps}
              height={280}
              staticLabel="BCC State (initial H_active)"
            />
          ) : (
            <div className="h-32 flex items-center justify-center text-gray-600">Loading BCC state…</div>
          )}
        </div>
      )}
    </div>
  );
}
