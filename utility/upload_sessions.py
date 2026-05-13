"""Upload local record-replay sessions to Azure Blob + Table Storage.

Usage (from project root):
    python utility/upload_sessions.py                       # upload all sessions
    python utility/upload_sessions.py --session <id>        # upload one session
    python utility/upload_sessions.py --dry-run             # preview only
    python utility/upload_sessions.py --config my.toml      # use custom config file

Config file (default: azure_config.toml in project root):
    Copy azure_config.toml.example → azure_config.toml and fill in your values.

    [azure]
    connection_string = "DefaultEndpointsProtocol=https;AccountName=..."
    blob_container    = "replay-data"
    table_name        = "sessions"

Install deps (once):
    pip install azure-storage-blob azure-data-tables msgpack tomli
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib                # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib   # pip install tomli
    except ImportError:
        tomllib = None            # handled at config-load time

# ---------------------------------------------------------------------------
# Path setup — lets us reuse record_replay/storage.py for msgpack decoding
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_RECORD_REPLAY = _ROOT / "record_replay"
sys.path.insert(0, str(_RECORD_REPLAY))

from storage import read_msgpack  # noqa: E402  (path setup must come first)

# ---------------------------------------------------------------------------
# Azure clients — imported lazily so --dry-run works without them installed
# ---------------------------------------------------------------------------

def _blob_client(conn_str: str, container: str):
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(conn_str).get_container_client(container)


def _table_client(conn_str: str, table: str):
    from azure.data.tables import TableClient
    return TableClient.from_connection_string(conn_str, table_name=table)


# ---------------------------------------------------------------------------
# Metadata extraction — reads local msgpack files; no Azure needed
# ---------------------------------------------------------------------------

def extract_metadata(session_dir: Path) -> dict:
    """Return a dict with shape, total_steps, n_checkpoints, created_at."""
    initial = read_msgpack(session_dir / "initial.msgpack")
    events  = read_msgpack(session_dir / "commands.msgpack")

    shape_rows, shape_cols = initial.shape
    total_steps = len(events)

    n_checkpoints = sum(
        1 for p in session_dir.iterdir()
        if re.fullmatch(r"checkpoint_\d+\.msgpack", p.name)
    )

    # Use the mtime of initial.msgpack as the session creation timestamp
    mtime = (session_dir / "initial.msgpack").stat().st_mtime
    created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    return {
        "shape_rows":    shape_rows,
        "shape_cols":    shape_cols,
        "total_steps":   total_steps,
        "n_checkpoints": n_checkpoints,
        "created_at":    created_at,
    }


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def upload_session_blobs(
    session_id: str,
    session_dir: Path,
    container_client,
    *,
    dry_run: bool,
    skip_existing: bool,
) -> int:
    """Upload all msgpack files and metadata.json for one session. Returns number of files uploaded."""
    files = sorted(session_dir.glob("*.msgpack"))
    meta_file = session_dir / "metadata.json"
    if meta_file.exists():
        files = [meta_file] + files
    uploaded = 0

    for local_path in files:
        blob_name = f"sessions/{session_id}/{local_path.name}"

        if skip_existing and not dry_run:
            blob = container_client.get_blob_client(blob_name)
            if blob.exists():
                print(f"    skip  {blob_name}  (already exists)")
                continue

        size_kb = local_path.stat().st_size / 1024
        print(f"    {'[dry] ' if dry_run else ''}upload  {blob_name}  ({size_kb:.1f} KB)")

        if not dry_run:
            with open(local_path, "rb") as fh:
                container_client.upload_blob(
                    name=blob_name,
                    data=fh,
                    overwrite=True,
                )
        uploaded += 1

    return uploaded


def upsert_session_metadata(
    session_id: str,
    metadata: dict,
    table_client,
    *,
    dry_run: bool,
    table_name: str,
) -> None:
    """Write (or overwrite) one row in the sessions table."""
    entity = {
        "PartitionKey": "session",
        "RowKey":       session_id,
        **metadata,
    }

    print(f"  {'[dry] ' if dry_run else ''}upsert  Table:{table_name}  RowKey={session_id}")
    if not dry_run:
        table_client.upsert_entity(entity=entity)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def list_local_sessions(data_dir: Path) -> list[str]:
    dirs = sorted(
        (d for d in data_dir.iterdir()
         if d.is_dir() and d.name.startswith("session_")),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [d.name.removeprefix("session_") for d in dirs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load [azure] section from a TOML config file."""
    if tomllib is None:
        print("ERROR: TOML support not available.")
        print("  pip install tomli")
        sys.exit(1)

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print(f"  Copy azure_config.toml.example → {config_path} and fill in your values.")
        sys.exit(1)

    with open(config_path, "rb") as fh:
        data = tomllib.load(fh)

    azure = data.get("azure", {})

    # Env vars override config file values (useful for CI / scripting)
    return {
        "connection_string": os.environ.get("REDACTED")
                             or azure.get("connection_string", ""),
        "blob_container":    os.environ.get("AZURE_BLOB_CONTAINER")
                             or azure.get("blob_container", "replay-data"),
        "table_name":        os.environ.get("AZURE_TABLE_NAME")
                             or azure.get("table_name", "sessions"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload replay sessions to Azure")
    p.add_argument("--session", metavar="ID",
                   help="Upload only this session (default: all)")
    p.add_argument("--data-dir", default=str(_RECORD_REPLAY / "data"),
                   help="Local data directory (default: record_replay/data)")
    p.add_argument("--config", default=str(_ROOT / "azure_config.toml"),
                   metavar="PATH", help="TOML config file (default: azure_config.toml)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be uploaded without doing it")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip blobs that already exist (default: True)")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    # ── Config from TOML file (env vars override individual keys) ────────
    cfg = load_config(Path(args.config))
    conn_str       = cfg["connection_string"]
    blob_container = cfg["blob_container"]
    table_name     = cfg["table_name"]

    if not conn_str and not args.dry_run:
        print("ERROR: connection_string is empty.")
        print(f"  Fill in azure_config.toml or set REDACTED.")
        sys.exit(1)

    # ── Session list ──────────────────────────────────────────────────────
    all_sessions = list_local_sessions(data_dir)
    if not all_sessions:
        print(f"No sessions found in {data_dir}")
        sys.exit(0)

    target_sessions = [args.session] if args.session else all_sessions
    for sid in target_sessions:
        if sid not in all_sessions:
            print(f"Session {sid!r} not found in {data_dir}")
            sys.exit(1)

    # ── Azure clients ─────────────────────────────────────────────────────
    container_client = _blob_client(conn_str, blob_container) if not args.dry_run else None
    table_client     = _table_client(conn_str, table_name)    if not args.dry_run else None

    # ── Upload loop ───────────────────────────────────────────────────────
    total_uploaded = 0
    for session_id in target_sessions:
        session_dir = data_dir / f"session_{session_id}"
        print(f"\nSession: {session_id}")

        print("  Extracting metadata ...")
        meta = extract_metadata(session_dir)
        print(f"    shape=({meta['shape_rows']}×{meta['shape_cols']})  "
              f"steps={meta['total_steps']}  checkpoints={meta['n_checkpoints']}")

        print("  Uploading blobs ...")
        n = upload_session_blobs(
            session_id, session_dir, container_client,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
        total_uploaded += n

        upsert_session_metadata(
            session_id, meta, table_client,
            dry_run=args.dry_run,
            table_name=table_name,
        )

    print(f"\nDone. {total_uploaded} file(s) uploaded across {len(target_sessions)} session(s).")


if __name__ == "__main__":
    main()
