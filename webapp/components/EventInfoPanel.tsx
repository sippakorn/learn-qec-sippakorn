"use client";

import type { EventRecord } from "@/lib/msgpack-decode";

const BADGE_COLOR: Record<string, string> = {
  swap_rows:      "#4c8bf5",
  scale_row:      "#f5a623",
  add_scaled_row: "#7ed321",
};

interface Props {
  event: EventRecord | null;
}

export default function EventInfoPanel({ event }: Props) {
  if (!event) {
    return <span className="text-gray-500 text-sm">Initial state — no prior operation</span>;
  }

  const ts = new Date(event.timestamp * 1000).toISOString().replace("T", "  ").slice(0, 22);
  const badgeBg = BADGE_COLOR[event.event_type] ?? "#666";

  return (
    <div className="space-y-1 text-xs font-mono">
      <div className="mb-2">
        <span
          className="px-2 py-0.5 rounded text-white text-xs font-bold"
          style={{ background: badgeBg }}
        >
          {event.event_type}
        </span>
      </div>
      <Row label="event_id" value={String(event.event_id)} />
      <Row label="step"     value={String(event.step)} />
      <Row label="time"     value={ts} />
      <hr className="border-[#2a2a4a] my-2" />
      <div className="text-gray-500 mb-1">params</div>
      {Object.entries(event.params).map(([k, v]) => (
        <Row key={k} label={k} value={typeof v === "number" ? v.toFixed(6).replace(/\.?0+$/, "") : String(v)} />
      ))}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <span className="text-gray-500 min-w-[72px]">{label}:</span>
      <span className="text-yellow-300">{value}</span>
    </div>
  );
}
