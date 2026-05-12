"""Interactive REPL to annotate replay sessions with metadata.

Metadata is stored locally as:
    record_replay/data/session_{id}/metadata.json

Run from project root:
    python record_replay/annotate.py
"""

import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

FIELDS = [
    ("code_family",  "Code family (e.g. n625, n1225)",   str,   None),
    ("erasure_rate", "Erasure rate (0–1)",                float, lambda v: 0.0 <= v <= 1.0),
    ("reorder",      "Reorder method (e.g. dfs, none)",  str,   None),
    ("note",         "Note (free text)",                  str,   None),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_sessions() -> list[str]:
    if not DATA_DIR.exists():
        return []
    dirs = sorted(
        (d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("session_")),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [d.name.removeprefix("session_") for d in dirs]


def metadata_path(session_id: str) -> Path:
    return DATA_DIR / f"session_{session_id}" / "metadata.json"


def load_metadata(session_id: str) -> dict:
    path = metadata_path(session_id)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_metadata(session_id: str, data: dict) -> None:
    metadata_path(session_id).write_text(json.dumps(data, indent=2))


def delete_metadata(session_id: str) -> None:
    path = metadata_path(session_id)
    if path.exists():
        path.unlink()


def fmt_meta(meta: dict) -> str:
    if not meta:
        return "[no annotation]"
    parts = []
    if meta.get("code_family"):
        parts.append(meta["code_family"])
    if meta.get("erasure_rate") is not None:
        parts.append(f"rate={meta['erasure_rate']}")
    if meta.get("reorder"):
        parts.append(f"reorder={meta['reorder']}")
    return ", ".join(parts) if parts else "[no annotation]"


def prompt(label: str, hint: str, current, cast, validate) -> tuple[bool, object]:
    """Prompt for one field. Returns (changed, value). Enter alone = keep current."""
    cur_str = str(current) if current is not None else ""
    bracket = f" [{cur_str}]" if cur_str else ""
    raw = input(f"  {hint}{bracket}: ").strip()

    if raw == "":
        return False, current  # skip — keep existing

    try:
        value = cast(raw)
    except (ValueError, TypeError):
        print(f"    ! Invalid value for {label}, keeping previous.")
        return False, current

    if validate and not validate(value):
        print(f"    ! Value out of range for {label}, keeping previous.")
        return False, current

    return True, value


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main() -> None:
    sessions = list_sessions()
    if not sessions:
        print(f"No sessions found in {DATA_DIR}.")
        sys.exit(0)

    # ── Session picker ─────────────────────────────────────────────────
    print("\nAvailable sessions:")
    for i, sid in enumerate(sessions, 1):
        meta = load_metadata(sid)
        print(f"  {i:>3}. {sid}  {fmt_meta(meta)}")
    print()

    while True:
        raw = input("Enter session ID or number (q to quit): ").strip()
        if raw.lower() == "q":
            sys.exit(0)
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(sessions):
                session_id = sessions[idx]
                break
            print("  ! Number out of range.")
        elif raw in sessions:
            session_id = raw
            break
        else:
            print("  ! Session not found.")

    meta = load_metadata(session_id)
    print(f"\nSession: {session_id}  {fmt_meta(meta)}")

    # ── Mode picker ────────────────────────────────────────────────────
    while True:
        mode = input("\n[a]dd/edit  [d]elete  [q]uit: ").strip().lower()
        if mode in ("a", "e", ""):
            mode = "edit"
            break
        elif mode == "d":
            mode = "delete"
            break
        elif mode == "q":
            sys.exit(0)

    if mode == "delete":
        confirm = input(f"  Delete annotation for {session_id}? [y/N]: ").strip().lower()
        if confirm == "y":
            delete_metadata(session_id)
            print("  Deleted.")
        else:
            print("  Cancelled.")
        sys.exit(0)

    # ── Add / edit ─────────────────────────────────────────────────────
    print("\nPress Enter to keep existing value, or type a new one.\n")
    updated = dict(meta)

    for key, hint, cast, validate in FIELDS:
        _, value = prompt(key, hint, updated.get(key), cast, validate)
        if value is not None:
            updated[key] = value
        elif key in updated and value is None:
            pass  # keep existing None-valued key as-is

    # Strip keys that are empty string
    updated = {k: v for k, v in updated.items() if v is not None and v != ""}

    save_metadata(session_id, updated)
    path = metadata_path(session_id)
    print(f"\nSaved → {path}")
    print(json.dumps(updated, indent=2))


if __name__ == "__main__":
    main()
