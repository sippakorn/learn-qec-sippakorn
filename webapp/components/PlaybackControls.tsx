"use client";

interface Props {
  step: number;
  totalSteps: number;
  playing: boolean;
  speed: number;
  onStep: (n: number) => void;
  onTogglePlay: () => void;
  onSpeedChange: (ms: number) => void;
}

const SPEED_OPTIONS = [
  { label: "0.5×", value: 2000 },
  { label: "1×",   value: 1000 },
  { label: "2×",   value: 500  },
  { label: "4×",   value: 250  },
];

const BTN = "px-3 py-1 bg-[#1e2d50] text-gray-300 border border-[#334] rounded cursor-pointer hover:bg-[#2a3f6e] transition-colors";

export default function PlaybackControls({
  step, totalSteps, playing, speed,
  onStep, onTogglePlay, onSpeedChange,
}: Props) {
  return (
    <div className="space-y-3">
      {/* Slider */}
      <input
        type="range"
        min={0}
        max={totalSteps}
        value={step}
        step={1}
        onChange={(e) => onStep(parseInt(e.target.value, 10))}
        className="w-full accent-blue-400"
      />
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>0</span>
        <span>{step.toLocaleString()} / {totalSteps.toLocaleString()}</span>
        <span>{totalSteps.toLocaleString()}</span>
      </div>

      {/* Buttons + speed */}
      <div className="flex items-center justify-center gap-2 flex-wrap">
        <button className={BTN} onClick={() => onStep(0)} title="Go to start">⏮</button>
        <button className={BTN} onClick={() => onStep(Math.max(0, step - 1))} title="Previous">◀</button>
        <button
          className={`${BTN} font-bold min-w-[90px]`}
          onClick={onTogglePlay}
        >
          {playing ? "⏸  Pause" : "▶  Play"}
        </button>
        <button className={BTN} onClick={() => onStep(Math.min(step + 1, totalSteps))} title="Next">▶</button>
        <button className={BTN} onClick={() => onStep(totalSteps)} title="Go to end">⏭</button>

        <div className="flex items-center gap-1 ml-4">
          <span className="text-gray-500 text-xs">Speed</span>
          <select
            value={speed}
            onChange={(e) => onSpeedChange(parseInt(e.target.value, 10))}
            className="bg-[#1e2d50] text-gray-300 border border-[#334] rounded px-1 py-0.5 text-xs"
          >
            {SPEED_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
