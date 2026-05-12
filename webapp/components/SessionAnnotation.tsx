"use client";

import type { Annotation } from "@/lib/replayer";

const LABELS: { key: keyof Annotation; label: string }[] = [
  { key: "code_family",  label: "Code family"  },
  { key: "erasure_rate", label: "Erasure rate"  },
  { key: "reorder",      label: "Reorder"       },
  { key: "note",         label: "Note"          },
];

interface Props {
  annotation: Annotation | null;
}

export default function SessionAnnotation({ annotation }: Props) {
  if (!annotation) {
    return (
      <span className="text-gray-600 text-xs">
        No annotation — run annotate.py to add one.
      </span>
    );
  }

  return (
    <div className="space-y-1 text-xs font-mono">
      {LABELS.map(({ key, label }) => {
        const val = annotation[key];
        if (val === undefined || val === null || val === "") return null;
        const display = key === "erasure_rate"
          ? (val as number).toFixed(2)
          : String(val);
        return (
          <div key={key} className="flex gap-2">
            <span className="text-gray-500 min-w-[80px]">{label}:</span>
            <span className="text-yellow-300 break-all">{display}</span>
          </div>
        );
      })}
    </div>
  );
}
