"""
app.py
======
AI-Powered Network Traffic Analyzer — LIVE DASHBOARD

Loads trained models, starts live packet capture, and serves the existing
Dash dashboard on http://127.0.0.1:8051

Run:
    python app.py

Prerequisites:
  1. python train_models.py     (once, to save models/)
  2. Npcap installed            (https://npcap.com/)
  3. Run as Administrator       (required for raw packet capture on Windows)

Configuration:
  Set CAPTURE_IFACE below if you want to hard-code an interface name.
  Leave as None to be prompted at startup.
"""

import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  ← edit here if needed
# ─────────────────────────────────────────────────────────────────────────────
CAPTURE_IFACE : str  = None   # e.g. "Wi-Fi" or "Ethernet" or None for prompt
DASHBOARD_PORT: int        = 8051
WINDOW_SIZE   : int        = 50     # number of recent flows shown in graphs / cards
# ─────────────────────────────────────────────────────────────────────────────

# ── Verify model artifacts exist ──────────────────────────────────────────────
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
for _f in ("rf_model.pkl", "ae_model.keras", "scaler.pkl", "artifacts.json"):
    if not os.path.exists(os.path.join(_MODELS_DIR, _f)):
        print(
            f"\n[ERROR] Missing artifact: models/{_f}\n"
            "Please run  python train_models.py  first.\n"
        )
        sys.exit(1)

# ── Load inference engine ─────────────────────────────────────────────────────
from inference import engine as _inference_engine
_inference_engine.load()

# ── Interface selection ───────────────────────────────────────────────────────
from live_capture import (
    select_interface_interactive,
    start_capture,
    stop_capture,
    get_flows_snapshot,
    capture_status,
)

if CAPTURE_IFACE is None:
    CAPTURE_IFACE = select_interface_interactive()

# ── Start capture ─────────────────────────────────────────────────────────────
start_capture(iface=CAPTURE_IFACE)
logger.info(
    f"Live capture running on '{CAPTURE_IFACE or 'default'}'. "
    "Dashboard starting …"
)

# ── Dash imports ──────────────────────────────────────────────────────────────
import dash
from dash import Dash, html, dcc, dash_table, Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "AI-Powered Network Traffic Analyzer"


# ── Layout (preserved exactly from the original notebook) ────────────────────
app.layout = html.Div([

    html.H2(
        "AI-Powered Network Traffic Analyzer — Dashboard"
    ),

    # Live capture status bar (minimal addition required for live data)
    html.Div(
        id="capture-status-bar",
        style={
            "padding"      : "6px 12px",
            "marginBottom" : "6px",
            "background"   : "#f0f8ff",
            "border"       : "1px solid #b0d0f0",
            "borderRadius" : "4px",
            "fontSize"     : "13px",
            "color"        : "#336",
        }
    ),

    html.Div([

        html.Div([

            html.Label(
                "Stream position (index)"
            ),

            dcc.Slider(
                id="pos-slider",
                min=0,
                max=0,
                step=1,
                value=0,
            )

        ], style={
            "width"        : "65%",
            "display"      : "inline-block",
            "padding"      : "10px",
        }),

        html.Div([

            html.Button(
                "Play Stream",
                id="play-btn",
                n_clicks=0,
            ),

            html.Button(
                "Pause",
                id="pause-btn",
                n_clicks=0,
            ),

            html.Button(
                "Jump to End",
                id="end-btn",
                n_clicks=0,
            )

        ], style={
            "display"      : "inline-block",
            "verticalAlign": "top",
            "padding"      : "10px",
        })

    ]),

    html.Div(
        id="summary-cards",
        style={
            "display": "flex",
            "gap"    : "20px",
        }
    ),

    dcc.Graph(
        id="time-series-agg"
    ),

    dcc.Graph(
        id="scatter-score"
    ),

    html.H4(
        "Recent events"
    ),

    dash_table.DataTable(

        id="recent-table",

        columns=[
            {"name": c, "id": c}
            for c in [
                "timestamp",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "protocol",
                "pkt_count",
                "byte_count",
                "duration",
                "true_label",
                "rf_prob",
                "ae_mse",
                "score_combined",
            ]
        ],

        data=[],

        page_size=10,

        style_table={
            "overflowX": "auto"
        },

        style_data_conditional=[
            {
                "if"           : {"filter_query": "{rf_prob} >= 0.5"},
                "backgroundColor": "#fff0f0",
                "color"        : "#900",
            }
        ],
    ),

    html.H4(
        "Confusion Matrix (RF)"
    ),

    dcc.Graph(
        id="conf-matrix"
    ),

    dcc.Interval(
        id="interval",
        interval=1500,    # ms — same as original notebook
        n_intervals=0,
    ),

    # Hidden store for playing state
    dcc.Store(id="playing-store", data=False),

], style={
    "padding": "10px"
})


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build a DataFrame from the current live_flows snapshot up to 'pos'
# ─────────────────────────────────────────────────────────────────────────────

def _get_window(pos: int) -> pd.DataFrame:
    flows = get_flows_snapshot()
    if not flows:
        return pd.DataFrame()

    # Slice up to pos (inclusive), then take the trailing WINDOW_SIZE rows
    end   = min(pos + 1, len(flows))
    start = max(0, end - WINDOW_SIZE)
    subset = flows[start:end]

    if not subset:
        return pd.DataFrame()

    df = pd.DataFrame(subset)

    # Ensure timestamp is datetime for groupby
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 1 — Slider max / auto-advance
# (Preserved exactly from notebook; slider max updated dynamically)
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("pos-slider", "value"),
    Output("pos-slider", "max"),
    Output("playing-store", "data"),
    Input("interval",      "n_intervals"),
    Input("play-btn",      "n_clicks"),
    Input("pause-btn",     "n_clicks"),
    Input("end-btn",       "n_clicks"),
    Input("pos-slider",    "value"),
    Input("playing-store", "data"),
)
def auto_advance(n_intervals, play, pause, end, slider_val, is_playing):

    ctx = dash.callback_context

    n_flows = len(get_flows_snapshot())
    new_max = max(0, n_flows - 1)

    if not ctx.triggered:
        return slider_val, new_max, is_playing

    tid = ctx.triggered[0]["prop_id"].split(".")[0]

    if tid == "play-btn":
        return min(new_max, slider_val + 1), new_max, True

    if tid == "pause-btn":
        return slider_val, new_max, False

    if tid == "end-btn":
        return new_max, new_max, is_playing

    if tid == "interval":
        if is_playing:
            return min(new_max, slider_val + 1), new_max, is_playing
        return slider_val, new_max, is_playing

    return slider_val, new_max, is_playing


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 2 — Dashboard update
# (Preserved exactly from notebook; ground-truth handled gracefully)
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(

    Output("recent-table",    "data"),
    Output("time-series-agg", "figure"),
    Output("scatter-score",   "figure"),
    Output("summary-cards",   "children"),
    Output("conf-matrix",     "figure"),
    Output("capture-status-bar", "children"),

    Input("pos-slider", "value"),
    Input("interval",   "n_intervals"),
)
def update_ui(pos, _n_intervals):

    if pos is None:
        raise PreventUpdate

    window = _get_window(pos)

    # ── Capture status bar ────────────────────────────────────────────────────
    s = capture_status
    status_text = (
        f"🔴 LIVE  |  Interface: {s['iface'] or 'default'}  |  "
        f"Packets: {s['packets']:,}  |  "
        f"IPv4: {s['ipv4_packets']:,}  |  "
        f"Active flows: {s['active_flows']:,}  |  "
        f"Finalized: {s['flows_finalized']:,}  |  "
        f"→ Inference: {s['flows_to_inference']:,}  |  "
        f"Dashboard flows: {s['flows']:,}  |  "
        f"Errors: {s['errors']}"
        + (f"  ← {s['last_error']}" if s["errors"] and s["last_error"] else "")
    )

    # ── Summary cards ─────────────────────────────────────────────────────────
    total    = len(window)
    rf_alerts = int((window["rf_pred"] == 1).sum()) if total else 0
    ae_alerts = int((window["ae_pred"] == 1).sum()) if total else 0

    # true_label is always "N/A — Live Traffic" for live flows
    # The card still exists but shows N/A to preserve layout
    cards = [

        html.Div([
            html.H3("Window size"),
            html.P(str(total)),
        ], style={
            "padding"     : "10px",
            "border"      : "1px solid #ddd",
            "borderRadius": "6px",
        }),

        html.Div([
            html.H3("True Attacks"),
            html.P("N/A — Live Traffic"),
        ], style={
            "padding"         : "10px",
            "border"          : "1px solid #f88",
            "borderRadius"    : "6px",
            "backgroundColor" : "#ffeeee",
        }),

        html.Div([
            html.H3("RF Alerts"),
            html.P(str(rf_alerts)),
        ], style={
            "padding"     : "10px",
            "border"      : "1px solid #f8d88c",
            "borderRadius": "6px",
        }),

        html.Div([
            html.H3("AE Alerts"),
            html.P(str(ae_alerts)),
        ], style={
            "padding"     : "10px",
            "border"      : "1px solid #8cf8d8",
            "borderRadius": "6px",
        }),
    ]

    # ── Time-series traffic count graph ───────────────────────────────────────
    if window.empty:
        fig_ts = go.Figure()
        fig_ts.update_layout(title="Traffic count (recent window)")
    else:
        agg = (
            window
            .groupby(window["timestamp"].dt.floor("s"))
            .agg(
                total   =("timestamp", "count"),
                rf_alert=("rf_pred",   lambda s: (s == 1).sum()),
            )
            .reset_index()
        )

        fig_ts = go.Figure()

        if not agg.empty:
            fig_ts.add_trace(
                go.Bar(
                    x=agg["timestamp"],
                    y=agg["total"],
                    name="Total",
                )
            )
            fig_ts.add_trace(
                go.Bar(
                    x=agg["timestamp"],
                    y=agg["rf_alert"],
                    name="RF Alerts",
                    marker_color="red",
                )
            )
            fig_ts.update_layout(
                barmode="overlay",
                title="Traffic count (recent window)",
            )

    # ── Score vs Packet Rate scatter ──────────────────────────────────────────
    if window.empty:
        fig_sc = go.Figure()
        fig_sc.update_layout(title="Score vs Packet rate (recent)")
    else:
        # Colour by RF prediction since true_label is N/A in live mode
        window_plot          = window.copy()
        window_plot["label"] = window_plot["rf_pred"].map(
            {0: "Normal (RF)", 1: "Attack (RF)"}
        ).fillna("Unknown")

        fig_sc = px.scatter(
            window_plot,
            x="pkt_rate",
            y="score_combined",
            color="label",
            color_discrete_map={
                "Normal (RF)": "#1f77b4",
                "Attack (RF)": "#d62728",
            },
            hover_data=["src_ip", "dst_ip", "rf_prob", "ae_mse"],
        )
        fig_sc.update_layout(
            title="Score vs Packet rate (recent)"
        )

    # ── Confusion matrix ──────────────────────────────────────────────────────
    # No ground truth in live mode. Show an informative placeholder that
    # preserves the graph component and its position in the layout.
    cm_fig = go.Figure()
    cm_fig.add_annotation(
        text=(
            "N/A — Live Traffic<br>"
            "<span style='font-size:13px;color:#666'>"
            "Confusion matrix requires ground-truth labels.<br>"
            "Ground truth is not available for live captured traffic."
            "</span>"
        ),
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16),
        align="center",
    )
    cm_fig.update_layout(
        title="Confusion Matrix (RF) — N/A: Live Traffic (no ground truth)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    # ── Recent events table ───────────────────────────────────────────────────
    recent_cols = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "protocol", "pkt_count", "byte_count", "duration",
        "true_label", "rf_prob", "ae_mse", "score_combined",
    ]

    if window.empty:
        recent_data = []
    else:
        present_cols = [c for c in recent_cols if c in window.columns]
        recent_data = (
            window
            .sort_values("timestamp", ascending=False)
            .head(20)[present_cols]
            .to_dict("records")
        )

    return (
        recent_data,
        fig_ts,
        fig_sc,
        cards,
        cm_fig,
        status_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import atexit
    atexit.register(stop_capture)

    print(f"\nDashboard → http://127.0.0.1:{DASHBOARD_PORT}\n")

    app.run(
        debug=False,
        port=DASHBOARD_PORT,
        use_reloader=False,     # MUST be False — reloader kills the capture thread
    )
