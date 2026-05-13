"""Plotly Dash dashboard for interactive replay of recorded GE sessions.

Entry point: build_app(data_dir, initial_session) → Dash
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

METADATA_LABELS = {
    "code_family":  "Code family",
    "erasure_rate": "Erasure rate",
    "reorder":      "Reorder",
    "note":         "Note",
}

import numpy as np
import plotly.graph_objects as go
from dash import (
    Dash, Input, Output, State,
    ctx, dcc, html, no_update,
)

from replayer import Replayer

# ---------------------------------------------------------------------------
# Module-level session cache (survives across Dash callbacks)
# ---------------------------------------------------------------------------

_DATA_DIR: Path = Path(__file__).parent / "data"
_REPLAYERS: dict[str, Replayer] = {}

SPEED_OPTIONS = [
    {"label": "0.5×", "value": 2000},
    {"label": "1×",   "value": 1000},
    {"label": "2×",   "value": 500},
    {"label": "4×",   "value": 250},
]

EVENT_BADGE_COLOR = {
    "swap_rows":      "#4c8bf5",
    "scale_row":      "#f5a623",
    "add_scaled_row": "#7ed321",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_sessions(data_dir: Path) -> list[str]:
    """Return session ids sorted newest-first by folder mtime."""
    dirs = sorted(
        (d for d in data_dir.iterdir()
         if d.is_dir() and d.name.startswith("session_")),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [d.name.removeprefix("session_") for d in dirs]


def load_metadata(session_id: str) -> dict:
    path = _DATA_DIR / f"session_{session_id}" / "metadata.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def get_replayer(session_id: str) -> Replayer:
    """Return a (possibly cached) Replayer for the given session id."""
    if session_id not in _REPLAYERS:
        r = Replayer(_DATA_DIR)
        r.load_session(session_id)
        _REPLAYERS[session_id] = r
    return _REPLAYERS[session_id]


def _slider_marks(total: int) -> dict:
    marks: dict[int, str] = {0: "0"}
    for frac in (0.25, 0.5, 0.75):
        v = int(total * frac)
        if 0 < v < total:
            marks[v] = str(v)
    marks[total] = str(total)
    return marks


# ---------------------------------------------------------------------------
# Figure builder — called only when step or session changes
# ---------------------------------------------------------------------------

def build_figure(step: int, session_id: str) -> go.Figure:
    rep = get_replayer(session_id)

    # Fetch prev first: cache at (step-1) lets get_step(step) cost 1 event
    prev_step = max(0, step - 1)
    mat_prev = rep.get_step(prev_step)
    mat_curr = rep.get_step(step)

    # Dense conversion is deferred to here (the render boundary)
    dense = mat_curr.toarray()
    dense_prev = mat_prev.toarray()
    diff_mask = (dense != dense_prev).astype(np.float32)

    fig = go.Figure()

    # Primary layer: matrix values with diverging colorscale
    fig.add_trace(go.Heatmap(
        z=dense,
        colorscale="RdBu",
        zmid=0,
        showscale=True,
        colorbar=dict(title="Value", thickness=12, len=0.8),
        name="Matrix",
        hovertemplate="row %{y}  col %{x}<br>value: %{z:.5g}<extra></extra>",
    ))

    # Diff overlay: changed cells tinted amber (transparent where unchanged)
    fig.add_trace(go.Heatmap(
        z=diff_mask,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(255,185,0,0.45)"]],
        showscale=False,
        name="Changed",
        hoverinfo="skip",
        zmin=0,
        zmax=1,
    ))

    fig.update_layout(
        title=dict(
            text=f"Step {step} / {rep.total_steps()}",
            x=0.5, font=dict(size=14, color="#ddd"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        font=dict(color="#ccc"),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False,
                     zeroline=False, autorange="reversed")

    return fig


def build_tanner_figure(step: int, session_id: str) -> go.Figure:
    """Bipartite Tanner graph of the current matrix state.

    Rows = variable nodes (left, blue).
    Cols = check nodes (right, red), last column (syndrome) excluded.
    a(i, j) = 1  →  edge between v_i and c_j.
    Node size scales with degree.
    """
    rep = get_replayer(session_id)
    mat = rep.get_step(step)
    H = mat.toarray()[:, :-1]          # strip augmented syndrome column
    num_rows, num_cols = H.shape

    row_deg = H.sum(axis=1)             # variable-node degrees
    col_deg = H.sum(axis=0)             # check-node degrees

    # Normalised y positions so both sides span [0, 1]
    var_y = [i / max(1, num_rows - 1) for i in range(num_rows)]
    chk_y = [j / max(1, num_cols - 1) for j in range(num_cols)]

    ri, ci = np.where(H != 0)
    edge_x: list = []
    edge_y: list = []
    for r, c in zip(ri, ci):
        edge_x += [0, 1, None]
        edge_y += [var_y[r], chk_y[c], None]

    fig = go.Figure()

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="rgba(160,160,210,0.22)", width=0.8),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=[0] * num_rows, y=var_y,
        mode="markers",
        marker=dict(
            size=[5 + int(d) for d in row_deg],
            color="#4c8bf5",
            line=dict(width=0.5, color="#2a5fc4"),
        ),
        customdata=[[i, int(row_deg[i])] for i in range(num_rows)],
        hovertemplate="v%{customdata[0]}  deg %{customdata[1]}<extra></extra>",
        name="variable nodes",
    ))

    fig.add_trace(go.Scatter(
        x=[1] * num_cols, y=chk_y,
        mode="markers",
        marker=dict(
            size=[5 + int(d) for d in col_deg],
            color="#e74c3c",
            line=dict(width=0.5, color="#c0392b"),
        ),
        customdata=[[j, int(col_deg[j])] for j in range(num_cols)],
        hovertemplate="c%{customdata[0]}  deg %{customdata[1]}<extra></extra>",
        name="check nodes",
    ))

    fig.update_layout(
        title=dict(
            text=(f"Tanner Graph — Step {step} / {rep.total_steps()}"
                  f"  ·  {num_rows} var  {num_cols} chk  {len(ri)} edges"),
            x=0.5, font=dict(size=13, color="#ddd"),
        ),
        annotations=[
            dict(x=0, y=1.04, xref="paper", yref="paper",
                 text="Variable nodes (rows)", showarrow=False,
                 font=dict(color="#4c8bf5", size=11), xanchor="center"),
            dict(x=1, y=1.04, xref="paper", yref="paper",
                 text="Check nodes (cols)", showarrow=False,
                 font=dict(color="#e74c3c", size=11), xanchor="center"),
        ],
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        font=dict(color="#ccc"),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.04),
        xaxis=dict(range=[-0.15, 1.15],
                   showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False,
                   zeroline=False, autorange="reversed"),
    )
    return fig


def build_event_info(step: int, session_id: str) -> list:
    rep = get_replayer(session_id)
    ev = rep.event_at(step)

    if ev is None:
        return [html.Span("Initial state — no prior operation",
                           style={"color": "#888"})]

    ts = datetime.datetime.fromtimestamp(ev["timestamp"]).strftime(
        "%Y-%m-%d  %H:%M:%S.%f")[:-3]
    badge_bg = EVENT_BADGE_COLOR.get(ev["event_type"], "#666")

    rows = [
        # Type badge + ids
        html.Div([
            html.Span(ev["event_type"],
                      style={"background": badge_bg, "color": "#fff",
                             "borderRadius": "4px", "padding": "2px 8px",
                             "fontWeight": "bold", "fontSize": "0.85rem"}),
        ], style={"marginBottom": "8px"}),

        _info_row("event_id", str(ev["event_id"])),
        _info_row("step",     str(ev["step"])),
        _info_row("time",     ts),
        html.Hr(style={"borderColor": "#2a2a4a", "margin": "8px 0"}),
        html.Div("params", style={"color": "#888", "fontSize": "0.72rem",
                                   "marginBottom": "4px"}),
    ]

    for k, v in ev["params"].items():
        rows.append(_info_row(k, f"{v:.6g}" if isinstance(v, float) else str(v)))

    return rows


def _info_row(key: str, val: str) -> html.Div:
    return html.Div([
        html.Span(key + ":", style={"color": "#777", "marginRight": "6px",
                                     "minWidth": "72px", "display": "inline-block"}),
        html.Span(val, style={"color": "#f0c040"}),
    ], style={"fontSize": "0.8rem", "marginBottom": "3px"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(data_dir: Path, initial_session: Optional[str] = None) -> Dash:
    global _DATA_DIR
    _DATA_DIR = data_dir

    sessions = list_sessions(data_dir)
    default_session = initial_session or (sessions[0] if sessions else None)
    session_opts = [{"label": s, "value": s} for s in sessions]

    app = Dash(__name__, title="Matrix Replay Viewer")
    app.config.suppress_callback_exceptions = True

    # ── style constants ───────────────────────────────────────────────────
    BG_PAGE  = "#0f0e17"
    BG_CARD  = "#16213e"
    CLR_TEXT = "#ddd"
    CARD = {"background": BG_CARD, "borderRadius": "6px",
            "padding": "12px 16px", "marginBottom": "10px"}
    LABEL = {"color": "#888", "fontSize": "0.72rem", "marginBottom": "3px"}
    BTN = {"padding": "6px 14px", "cursor": "pointer",
           "background": "#1e2d50", "color": "#ddd",
           "border": "1px solid #334", "borderRadius": "4px"}

    # ── layout ───────────────────────────────────────────────────────────
    app.layout = html.Div(
        style={"background": BG_PAGE, "minHeight": "100vh",
               "padding": "14px 18px", "fontFamily": "monospace",
               "color": CLR_TEXT},
        children=[

            # ── top bar: session + speed selectors ──────────────────────
            html.Div(style={"display": "flex", "gap": "20px",
                            "alignItems": "flex-end", "marginBottom": "10px"},
                     children=[
                html.Div([
                    html.Div("Session", style=LABEL),
                    dcc.Dropdown(id="session-dropdown", options=session_opts,
                                 value=default_session, clearable=False,
                                 style={"width": "220px", "color": "#111",
                                        "minWidth": "160px"}),
                ]),
                html.Div([
                    html.Div("Speed", style=LABEL),
                    dcc.Dropdown(id="speed-dropdown", options=SPEED_OPTIONS,
                                 value=1000, clearable=False,
                                 style={"width": "90px", "color": "#111"}),
                ]),
                html.Div(id="session-stats",
                         style={"color": "#777", "fontSize": "0.78rem",
                                "paddingBottom": "6px"}),
            ]),

            # ── main panel: heatmap + event sidebar ─────────────────────
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "1fr 260px",
                            "gap": "10px"},
                     children=[

                # Heatmap
                html.Div(style=CARD, children=[
                    dcc.Graph(id="matrix-graph",
                              config={"displayModeBar": False},
                              style={"height": "510px"}),
                ]),

                # Event info panel
                html.Div(style={**CARD, "fontSize": "0.82rem",
                                "overflowY": "auto", "maxHeight": "540px"},
                         children=[
                    html.Div("Event info",
                             style={"fontWeight": "bold", "marginBottom": "10px",
                                    "color": "#aaa"}),
                    html.Div(id="event-info"),
                    html.Hr(style={"borderColor": "#2a2a4a", "margin": "14px 0 8px"}),
                    html.Div("Diff legend", style={**LABEL, "marginBottom": "5px"}),
                    html.Div([
                        html.Span("■ ", style={"color": "rgba(255,185,0,0.8)"}),
                        "Changed since previous step",
                    ], style={"fontSize": "0.75rem", "color": "#777"}),
                    html.Hr(style={"borderColor": "#2a2a4a", "margin": "14px 0 8px"}),
                    html.Div("Session annotation",
                             style={"fontWeight": "bold", "marginBottom": "8px",
                                    "color": "#aaa", "fontSize": "0.82rem"}),
                    html.Div(id="session-annotation"),
                ]),
            ]),

            # ── scrubber ────────────────────────────────────────────────
            html.Div(style=CARD, children=[
                dcc.Slider(
                    id="step-slider",
                    min=0, max=1, step=1, value=0,
                    marks={0: "0", 1: "1"},
                    updatemode="drag",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]),

            # ── playback controls ────────────────────────────────────────
            html.Div(style={**CARD, "textAlign": "center"}, children=[
                html.Button("⏮", id="goto-start-btn", n_clicks=0,
                            style=BTN, title="Go to start"),
                html.Button("◀", id="prev-btn", n_clicks=0,
                            style={**BTN, "marginLeft": "6px"}, title="Previous step"),
                html.Button("▶  Play", id="play-pause-btn", n_clicks=0,
                            style={**BTN, "marginLeft": "6px",
                                   "fontWeight": "bold", "minWidth": "90px"}),
                html.Button("▶", id="next-btn", n_clicks=0,
                            style={**BTN, "marginLeft": "6px"}, title="Next step"),
                html.Button("⏭", id="goto-end-btn", n_clicks=0,
                            style={**BTN, "marginLeft": "6px"}, title="Go to end"),
                html.Button("Graph", id="graph-toggle-btn", n_clicks=0,
                            style={**BTN, "marginLeft": "20px",
                                   "background": "#1e3a3a", "borderColor": "#2ecc71",
                                   "color": "#2ecc71"},
                            title="Toggle Tanner graph view"),
            ]),

            # ── hidden state stores ──────────────────────────────────────
            dcc.Store(id="is-playing-store", data=False),
            dcc.Store(id="view-mode-store", data="matrix"),
            dcc.Interval(id="interval", interval=1000, disabled=True),
        ],
    )

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_annotation(session_id: str):
        meta = load_metadata(session_id)
        if not meta:
            return html.Span("No annotation — run annotate.py to add one.",
                             style={"color": "#555", "fontSize": "0.75rem"})
        rows = []
        for key, label in METADATA_LABELS.items():
            if key not in meta:
                continue
            val = meta[key]
            if key == "erasure_rate":
                val = f"{val:.2f}"
            rows.append(html.Div([
                html.Span(label + ":", style={"color": "#777", "marginRight": "6px",
                                              "minWidth": "80px", "display": "inline-block"}),
                html.Span(str(val), style={"color": "#f0c040"}),
            ], style={"fontSize": "0.78rem", "marginBottom": "3px"}))
        return rows

    # ── callbacks ─────────────────────────────────────────────────────────

    @app.callback(
        Output("step-slider", "max"),
        Output("step-slider", "marks"),
        Output("step-slider", "value"),
        Output("session-stats", "children"),
        Output("session-annotation", "children"),
        Input("session-dropdown", "value"),
    )
    def on_session_change(session_id: str):
        if not session_id:
            return 1, {0: "0", 1: "1"}, 0, "", ""
        rep = get_replayer(session_id)
        total = rep.total_steps()
        nckpt = len(rep._ckpt_steps)
        stats = f"{total:,} steps · {nckpt} checkpoints · checkpoint every 50 steps"
        annotation = _build_annotation(session_id)
        return total, _slider_marks(total), 0, stats, annotation

    @app.callback(
        Output("matrix-graph", "figure"),
        Output("event-info", "children"),
        Input("step-slider", "value"),
        Input("session-dropdown", "value"),
        Input("view-mode-store", "data"),
    )
    def update_view(step, session_id: str, view_mode: str):
        if not session_id:
            return go.Figure(), html.Span("No session loaded.", style={"color": "#888"})
        step = int(step or 0)
        view_mode = view_mode or "matrix"
        fig = (build_tanner_figure(step, session_id)
               if view_mode == "graph"
               else build_figure(step, session_id))
        return fig, build_event_info(step, session_id)

    @app.callback(
        Output("is-playing-store", "data"),
        Output("play-pause-btn", "children"),
        Output("interval", "disabled"),
        Input("play-pause-btn", "n_clicks"),
        State("is-playing-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_play(_, is_playing: bool):
        playing = not is_playing
        label = "⏸  Pause" if playing else "▶  Play"
        return playing, label, not playing

    @app.callback(
        Output("interval", "interval"),
        Input("speed-dropdown", "value"),
    )
    def set_speed(ms: int):
        return ms or 1000

    @app.callback(
        Output("step-slider", "value", allow_duplicate=True),
        Input("interval", "n_intervals"),
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
        Input("goto-start-btn", "n_clicks"),
        Input("goto-end-btn", "n_clicks"),
        State("step-slider", "value"),
        State("step-slider", "max"),
        State("is-playing-store", "data"),
        prevent_initial_call=True,
    )
    def navigate(_, _p, _n, _gs, _ge, current, maximum, playing):
        trigger = ctx.triggered_id
        current = int(current or 0)
        maximum = int(maximum or 1)

        if trigger == "interval":
            if not playing or current >= maximum:
                return no_update
            return current + 1
        if trigger == "prev-btn":
            return max(0, current - 1)
        if trigger == "next-btn":
            return min(current + 1, maximum)
        if trigger == "goto-start-btn":
            return 0
        if trigger == "goto-end-btn":
            return maximum
        return no_update

    @app.callback(
        Output("view-mode-store", "data"),
        Output("graph-toggle-btn", "children"),
        Output("graph-toggle-btn", "style"),
        Input("graph-toggle-btn", "n_clicks"),
        State("view-mode-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_graph_view(_, mode: str):
        new_mode = "graph" if mode == "matrix" else "matrix"
        if new_mode == "graph":
            label = "Matrix"
            style = {**BTN, "marginLeft": "20px",
                     "background": "#3a1e3a", "borderColor": "#9b59b6",
                     "color": "#9b59b6"}
        else:
            label = "Graph"
            style = {**BTN, "marginLeft": "20px",
                     "background": "#1e3a3a", "borderColor": "#2ecc71",
                     "color": "#2ecc71"}
        return new_mode, label, style

    return app
