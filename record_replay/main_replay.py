"""Launch the matrix replay viewer.

Usage (from project root):
    python record_replay/main_replay.py                  # most recent session
    python record_replay/main_replay.py --session <id>   # specific session
    python record_replay/main_replay.py --port 8051
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from viewer import build_app, list_sessions

DATA_DIR = Path(__file__).parent / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matrix replay viewer")
    p.add_argument(
        "--session", metavar="SESSION_ID",
        help="Session id to pre-load (default: most recent by mtime)",
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8050)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Run `python record_replay/main.py` first to record a session.")
        sys.exit(1)

    sessions = list_sessions(DATA_DIR)
    if not sessions:
        print(f"No sessions found in {DATA_DIR}")
        print("Run `python record_replay/main.py` first to record a session.")
        sys.exit(1)

    session_id = args.session
    if session_id and session_id not in sessions:
        print(f"Session {session_id!r} not found.")
        print(f"Available sessions: {sessions}")
        sys.exit(1)

    if not session_id:
        session_id = sessions[0]
        print(f"Using most recent session: {session_id}")

    print(f"Sessions available: {len(sessions)}")
    app = build_app(DATA_DIR, initial_session=session_id)
    print(f"Launching viewer at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
