"""Interactive REPL to delete replay sessions from local storage and Azure.

Run from project root:
    python utility/housekeeping.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT          = Path(__file__).parent.parent
_DATA_DIR      = _ROOT / "record_replay" / "data"
_CONFIG_PATH   = _ROOT / "azure_config.toml"

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict | None:
    """Return Azure config dict, or None if unavailable."""
    if tomllib is None or not _CONFIG_PATH.exists():
        return None
    with open(_CONFIG_PATH, "rb") as fh:
        data = tomllib.load(fh)
    azure = data.get("azure", {})
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or azure.get("connection_string", "")
    if not conn_str or conn_str.startswith("DefaultEndpoints") is False:
        # Very basic check — just ensure the key exists and looks non-empty
        pass
    return {
        "connection_string": conn_str,
        "blob_container":    os.environ.get("AZURE_BLOB_CONTAINER") or azure.get("blob_container", "replay-data"),
        "table_name":        os.environ.get("AZURE_TABLE_NAME")     or azure.get("table_name", "sessions"),
    } if conn_str else None


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _list_local_sessions() -> list[str]:
    if not _DATA_DIR.exists():
        return []
    dirs = sorted(
        (d for d in _DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("session_")),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [d.name.removeprefix("session_") for d in dirs]


def _session_summary(session_id: str) -> dict:
    """Collect file list, sizes, and annotation for a session."""
    session_dir = _DATA_DIR / f"session_{session_id}"
    files = sorted(session_dir.iterdir())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    mtime = session_dir.stat().st_mtime
    created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    annotation: dict = {}
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        try:
            annotation = json.loads(meta_path.read_text())
        except Exception:
            pass

    return {
        "session_dir":  session_dir,
        "files":        [f for f in files if f.is_file()],
        "total_bytes":  total_bytes,
        "created_at":   created_at,
        "annotation":   annotation,
    }


def _print_summary(session_id: str, summary: dict) -> None:
    ann = summary["annotation"]
    print(f"\n  Session : {session_id}")
    print(f"  Created : {summary['created_at']}")
    if ann:
        for k in ("code_family", "erasure_rate", "reorder", "note"):
            if k in ann:
                print(f"  {k:<13}: {ann[k]}")
    print(f"\n  Files to delete ({len(summary['files'])}):")
    for f in summary["files"]:
        print(f"    {f.name:<40s}  {_human_bytes(f.stat().st_size):>10s}")
    print(f"\n  Total   : {_human_bytes(summary['total_bytes'])}")


# ---------------------------------------------------------------------------
# Azure helpers
# ---------------------------------------------------------------------------

def _list_azure_blobs(cfg: dict, session_id: str) -> list[str]:
    from azure.storage.blob import BlobServiceClient
    client = BlobServiceClient.from_connection_string(
        cfg["connection_string"]
    ).get_container_client(cfg["blob_container"])
    prefix = f"sessions/{session_id}/"
    return [b.name for b in client.list_blobs(name_starts_with=prefix)]


def _delete_azure_blobs(cfg: dict, blob_names: list[str]) -> None:
    from azure.storage.blob import BlobServiceClient
    container = BlobServiceClient.from_connection_string(
        cfg["connection_string"]
    ).get_container_client(cfg["blob_container"])
    for name in blob_names:
        container.delete_blob(name)
        print(f"    deleted  {name}")


def _delete_azure_table_row(cfg: dict, session_id: str) -> None:
    from azure.data.tables import TableClient
    client = TableClient.from_connection_string(
        cfg["connection_string"], table_name=cfg["table_name"]
    )
    try:
        client.delete_entity(partition_key="session", row_key=session_id)
        print(f"    deleted  table row  session/{session_id}")
    except Exception as exc:
        print(f"    (table row not found or already deleted: {exc})")


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _load_config()

    # ── List sessions ───────────────────────────────────────────────────
    sessions = _list_local_sessions()
    if not sessions:
        print(f"No sessions found in {_DATA_DIR}.")
        sys.exit(0)

    print(f"\nLocal sessions in {_DATA_DIR}:\n")
    for i, sid in enumerate(sessions, 1):
        sd = _DATA_DIR / f"session_{sid}"
        total = sum(f.stat().st_size for f in sd.iterdir() if f.is_file())
        n_files = sum(1 for f in sd.iterdir() if f.is_file())
        meta_path = sd / "metadata.json"
        tag = ""
        if meta_path.exists():
            try:
                ann = json.loads(meta_path.read_text())
                parts = [ann[k] for k in ("code_family", "erasure_rate") if k in ann]
                tag = f"  [{', '.join(str(p) for p in parts)}]" if parts else ""
            except Exception:
                pass
        print(f"  {i:>3}. {sid}  {_human_bytes(total):>9s}  {n_files} files{tag}")

    # ── Pick session ────────────────────────────────────────────────────
    print()
    while True:
        raw = input("Enter session ID or number to delete (q to quit): ").strip()
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

    # ── Show deletion summary ───────────────────────────────────────────
    summary = _session_summary(session_id)
    print("\n" + "─" * 52)
    print("  DELETION SUMMARY — local")
    print("─" * 52)
    _print_summary(session_id, summary)
    print("─" * 52)

    # ── Confirm local delete ────────────────────────────────────────────
    answer = input("\nDelete local session? [y/N]: ").strip().lower()
    if answer != "y":
        print("Cancelled.")
        sys.exit(0)

    shutil.rmtree(summary["session_dir"])
    print(f"\nDeleted local: {summary['session_dir']}")

    # ── Offer Azure delete ──────────────────────────────────────────────
    if cfg is None:
        print("\n(No Azure config found — skipping cloud deletion.)")
        sys.exit(0)

    answer = input("\nAlso delete from Azure Blob Storage? [y/N]: ").strip().lower()
    if answer != "y":
        print("Azure data kept.")
        sys.exit(0)

    # Discover blobs first so we can show the user what will be removed
    print("\n  Listing Azure blobs …")
    try:
        blobs = _list_azure_blobs(cfg, session_id)
    except Exception as exc:
        print(f"  Error listing blobs: {exc}")
        sys.exit(1)

    if not blobs:
        print(f"  No blobs found for session/{session_id} — nothing to delete.")
        sys.exit(0)

    print(f"\n  Blobs to delete ({len(blobs)}):")
    for b in blobs:
        print(f"    {b}")

    # ── Re-type session ID to confirm Azure delete ──────────────────────
    print()
    confirm = input(f"Retype session ID to confirm Azure deletion: ").strip()
    if confirm != session_id:
        print("  ID mismatch — Azure data kept.")
        sys.exit(0)

    print("\n  Deleting from Azure …")
    _delete_azure_blobs(cfg, blobs)
    _delete_azure_table_row(cfg, session_id)
    print(f"\nDone. Session {session_id} removed from local and Azure.")


if __name__ == "__main__":
    main()
