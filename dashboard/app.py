"""
dashboard/app.py — IoT Anomaly Detection Showcase Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import torch
import sys, os, time, threading
from datetime import datetime
from collections import deque
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))
sys.path.insert(0, os.path.join(BASE_DIR, "simulation"))
sys.path.insert(0, os.path.join(BASE_DIR, "llm"))

from cnn_lstm          import CNNLSTM
from traffic_simulator import generate_normal, generate_ddos, generate_botnet, FEATURES
from interpreter       import stream_attack_analysis

MODEL_PATH  = os.path.join(BASE_DIR, "models", "saved", "best_model.pth")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 10
LABELS      = {0: "Normal", 1: "DDoS", 2: "Botnet"}
COLORS      = {"Normal": "#2ecc71", "DDoS": "#e74c3c", "Botnet": "#9b59b6"}
BG_COLORS   = {"Normal": "rgba(46,204,113,0.1)", "DDoS": "rgba(231,76,60,0.15)", "Botnet": "rgba(155,89,182,0.15)"}
ICONS       = {"Normal": "🟢", "DDoS": "🔴", "Botnet": "🟣"}

SCENARIO = [
    ("normal",    22, "Normal Baseline"),
    ("syn_flood", 20, "DDoS — SYN Flood"),
    ("normal",     8, "Brief Recovery"),
    ("udp_flood", 16, "DDoS — UDP Flood"),
    ("normal",    10, "Traffic Normalizing"),
    ("botnet",    22, "Mirai Botnet Infection"),
    ("normal",    15, "Full Recovery"),
]

st.set_page_config(page_title="IoT Anomaly Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

* { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.main-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px;
}
.status-card {
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 2px solid;
    transition: all 0.3s ease;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.flow-log {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px;
    height: 200px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.llm-card {
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 12px;
    border-left: 4px solid;
    background: #161b22;
}
.alert-banner {
    background: linear-gradient(90deg, #2d0f0f, #1a0a0a);
    border: 1px solid #e74c3c;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 10px;
    animation: glow 1.5s ease-in-out infinite alternate;
}
.botnet-banner {
    background: linear-gradient(90deg, #1a0d2e, #0f0820);
    border: 1px solid #9b59b6;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 10px;
}
@keyframes glow {
    from { box-shadow: 0 0 5px rgba(231,76,60,0.3); }
    to   { box-shadow: 0 0 20px rgba(231,76,60,0.6); }
}
.phase-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
div[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CNNLSTM(n_features=ckpt.get("n_features", 38),
                    n_classes=ckpt.get("n_classes", 3)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, window):
    x     = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    cls   = int(probs.argmax())
    return cls, float(probs[cls]), probs


def get_flow(t):
    if t == "normal":    return generate_normal(1)[0]
    if t == "syn_flood": return generate_ddos(1, attack_type="syn_flood")[0]
    if t == "udp_flood": return generate_ddos(1, attack_type="udp_flood")[0]
    if t == "botnet":    return generate_botnet(1)[0]
    return generate_normal(1)[0]


def init():
    d = {
        "running": False, "done": False,
        "history": deque(maxlen=80),
        "log_lines": deque(maxlen=30),
        "stats": {"Normal": 0, "DDoS": 0, "Botnet": 0},
        "alerts": [],
        "llm_cards": [],
        "buffer": [], "flow_count": 0,
        "scenario_idx": 0, "phase_start": 0.0,
        "current_phase": "—", "current_label": "Normal",
        "current_conf": 0.0, "current_probs": [1.0, 0.0, 0.0],
        "last_llm_label": None, "last_llm_flow": -999,
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()
model = load_model()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ IoT Anomaly Detection")
    st.caption("CNN-LSTM Deep Learning + Ollama LLaMA 3.2:3b")
    st.divider()

    st.markdown("**Simulation Scenario**")
    for i, (t, dur, desc) in enumerate(SCENARIO):
        icon = "🟢" if t == "normal" else ("🔴" if "flood" in t else "🟣")
        active = (i == st.session_state.scenario_idx and st.session_state.running)
        style  = "font-weight:bold;color:white" if active else "color:#8b949e"
        st.markdown(f"<p style='{style};font-size:13px;margin:3px 0'>{icon} {desc} ({dur}s)</p>",
                    unsafe_allow_html=True)

    st.divider()
    flow_speed = st.slider("Flow speed (s)", 0.1, 0.8, 0.3, 0.1)
    enable_llm = st.toggle("🤖 LLM Analysis", value=True)
    st.divider()

    c1, c2 = st.columns(2)
    start_btn = c1.button("▶ Start", use_container_width=True, type="primary",
                           disabled=st.session_state.running)
    stop_btn  = c2.button("⏹ Stop",  use_container_width=True,
                           disabled=not st.session_state.running)

    if start_btn:
        ss = st.session_state
        for k in ["history","log_lines","alerts","llm_cards","buffer"]:
            ss[k] = (deque(maxlen=80) if "history" in k else
                     deque(maxlen=30) if "log" in k else [])
        ss.stats        = {"Normal": 0, "DDoS": 0, "Botnet": 0}
        ss.flow_count   = 0
        ss.scenario_idx = 0
        ss.phase_start  = time.time()
        ss.current_phase= SCENARIO[0][2]
        ss.current_label= "Normal"
        ss.current_conf = 0.0
        ss.last_llm_label = None
        ss.last_llm_flow  = -999
        ss.running = True
        ss.done    = False
        st.rerun()

    if stop_btn:
        st.session_state.running = False
        st.rerun()

    # Phase progress bar
    if st.session_state.running:
        idx = st.session_state.scenario_idx
        if idx < len(SCENARIO):
            _, dur, desc = SCENARIO[idx]
            elapsed = time.time() - st.session_state.phase_start
            st.markdown(f"<p style='color:#8b949e;font-size:12px;margin:4px 0'>Phase {idx+1}/{len(SCENARIO)}: {desc}</p>",
                        unsafe_allow_html=True)
            st.progress(min(elapsed / dur, 1.0))
            st.caption(f"{max(0, dur - elapsed):.0f}s remaining")

    st.divider()
    st.caption(f"Device: **{DEVICE.upper()}**")
    st.caption(f"Model accuracy: **99.92%**")
    st.caption(f"Features: **38** | Window: **10 flows**")

# ─── HEADER ───────────────────────────────────────────────────────────────────
label = st.session_state.current_label
color = COLORS.get(label, "#58a6ff")
icon  = ICONS.get(label, "⚪")
conf  = st.session_state.current_conf

st.markdown(f"""
<div class="main-header">
    <div style="font-size:48px">{icon}</div>
    <div style="flex:1">
        <div style="color:#8b949e;font-size:13px;margin-bottom:2px">CURRENT NETWORK STATUS</div>
        <div style="color:{color};font-size:28px;font-weight:700">{label.upper()}</div>
        <div style="color:#8b949e;font-size:13px">{st.session_state.current_phase}</div>
    </div>
    <div style="text-align:right">
        <div style="color:#8b949e;font-size:12px">CONFIDENCE</div>
        <div style="color:white;font-size:36px;font-weight:700">{conf*100:.1f}%</div>
        <div style="color:#8b949e;font-size:12px">Flow #{st.session_state.flow_count}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── METRICS ──────────────────────────────────────────────────────────────────
stats = st.session_state.stats
total = max(sum(stats.values()), 1)
c1, c2, c3, c4, c5 = st.columns(5)

def mcard(col, icon, label, val, pct, color):
    col.markdown(f"""
<div class="metric-card">
    <div style="font-size:22px">{icon}</div>
    <div style="color:#8b949e;font-size:11px;margin:2px 0">{label}</div>
    <div style="color:white;font-size:22px;font-weight:700">{val}</div>
    <div style="color:{color};font-size:12px">{pct:.0f}%</div>
</div>""", unsafe_allow_html=True)

mcard(c1, "📊", "Total Flows", total-1, 100, "#58a6ff")
mcard(c2, "🟢", "Normal",  stats["Normal"],  stats["Normal"]/total*100,  "#2ecc71")
mcard(c3, "🔴", "DDoS",    stats["DDoS"],    stats["DDoS"]/total*100,    "#e74c3c")
mcard(c4, "🟣", "Botnet",  stats["Botnet"],  stats["Botnet"]/total*100,  "#9b59b6")
mcard(c5, "🚨", "Alerts",  len(st.session_state.alerts),
      len(st.session_state.alerts)/max(total,1)*100, "#f39c12")

st.markdown("<br>", unsafe_allow_html=True)

# ─── CHART (rendered from session state — always executes before st.rerun) ────
chart_col, log_col = st.columns([3, 2])

with chart_col:
    st.markdown("#### 📈 Live Traffic Classification")
    history = list(st.session_state.history)
    if history:
        flows  = [h["flow"]  for h in history]
        labels = [h["label"] for h in history]
        confs  = [h["conf"]*100 for h in history]
        clrs   = [COLORS.get(l, "#888") for l in labels]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=flows, y=confs, fill="tozeroy",
            fillcolor="rgba(52,152,219,0.05)",
            line=dict(color="#1a3a5c", width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=flows, y=confs, mode="lines+markers",
            line=dict(color="#58a6ff", width=2),
            marker=dict(color=clrs, size=8, line=dict(width=1.5, color="#0d1117")),
            customdata=labels,
            hovertemplate="<b>Flow #%{x}</b><br>%{customdata}<br>Confidence: %{y:.1f}%<extra></extra>",
            showlegend=False,
        ))
        in_attack, atk_start, atk_label = False, None, None
        for i, (f, l) in enumerate(zip(flows, labels)):
            if l != "Normal" and not in_attack:
                in_attack, atk_start, atk_label = True, f, l
            elif l == "Normal" and in_attack:
                fig.add_vrect(x0=atk_start, x1=flows[i-1],
                              fillcolor=BG_COLORS.get(atk_label, "rgba(200,0,0,0.1)"),
                              layer="below", line_width=0)
                in_attack = False
        if in_attack and atk_start is not None:
            fig.add_vrect(x0=atk_start, x1=flows[-1],
                          fillcolor=BG_COLORS.get(atk_label, "rgba(200,0,0,0.1)"),
                          layer="below", line_width=0)
        fig.update_layout(
            height=260,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", size=11),
            xaxis=dict(title="Flow #", gridcolor="#21262d", color="#8b949e", showgrid=True),
            yaxis=dict(title="Confidence (%)", range=[0, 108], gridcolor="#21262d",
                       color="#8b949e", showgrid=True),
            margin=dict(l=50, r=10, t=10, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<p style='color:#8b949e;padding:30px;text-align:center'>"
                    "Press ▶ Start to begin simulation</p>", unsafe_allow_html=True)

# ─── LIVE FLOW LOG ─────────────────────────────────────────────────────────────
with log_col:
    st.markdown("#### 🖥️ Live Flow Log")
    log_lines = list(st.session_state.log_lines)
    if log_lines:
        lines_html = "<br>".join(reversed(log_lines))
        st.markdown(f'<div class="flow-log">{lines_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="flow-log"><span style="color:#8b949e">'
                    'Waiting for flows...</span></div>', unsafe_allow_html=True)

st.markdown("---")

# ─── ACTIVE ALERT BANNER ──────────────────────────────────────────────────────
_cur_lbl = st.session_state.current_label
if _cur_lbl != "Normal" and st.session_state.running:
    _color  = COLORS[_cur_lbl]
    _icon   = ICONS[_cur_lbl]
    _conf   = st.session_state.current_conf
    _probs  = st.session_state.current_probs
    _banner = "alert-banner" if _cur_lbl == "DDoS" else "botnet-banner"
    st.markdown(f"""
<div class="{_banner}">
    <span style="font-size:18px">{_icon}</span>
    <span style="color:{_color};font-weight:700;font-size:16px;margin-left:10px">
        🚨 ACTIVE ATTACK: {_cur_lbl.upper()}
    </span>
    <span style="color:#c9d1d9;margin-left:16px">Confidence: {_conf*100:.1f}%</span>
    <span style="color:#8b949e;margin-left:16px;font-size:13px">
        N:{_probs[0]*100:.0f}%&nbsp;&nbsp;D:{_probs[1]*100:.0f}%&nbsp;&nbsp;B:{_probs[2]*100:.0f}%
    </span>
    <span style="color:#8b949e;float:right;font-size:13px">
        Flow #{st.session_state.flow_count} | {st.session_state.current_phase}
    </span>
</div>""", unsafe_allow_html=True)

# ─── LLM ANALYSIS CARDS ───────────────────────────────────────────────────────
st.markdown("#### 🤖 LLM Security Analysis")

llm_cards = st.session_state.llm_cards
if not llm_cards:
    st.markdown("""
<div style="background:#161b22;border:1px dashed #30363d;border-radius:10px;
            padding:30px;text-align:center;color:#8b949e">
    🤖 LLM analysis will stream here in real-time when an attack is detected
</div>""", unsafe_allow_html=True)
else:
    for card in reversed(llm_cards):
        _lbl       = card["label"]
        _color     = COLORS.get(_lbl, "#58a6ff")
        _icon      = ICONS.get(_lbl, "⚠️")
        _is_latest = (card == llm_cards[-1])
        _pending   = card.get("response") is None
        _streaming = card.get("streaming", False)
        _text      = card.get("response") or ""

        if _pending:
            _hdr_status = "⏳ connecting..."
        elif _streaming:
            _hdr_status = f"⚡ streaming live"
        else:
            _hdr_status = f"{card['confidence']:.0f}% confidence"

        with st.expander(
            f"{_icon} {_lbl} Attack  •  {_hdr_status}  •  Flow #{card['flow']}  •  {card['time']}",
            expanded=_is_latest,
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                if _pending:
                    st.markdown(f"""
<div style="background:#0d1117;border-radius:8px;padding:22px 14px;
            text-align:center;border-left:3px solid {_color}">
    <div style="color:#58a6ff;font-size:14px;margin-bottom:6px">⏳ Connecting to LLaMA 3.2:3b...</div>
    <div style="color:#8b949e;font-size:12px">Simulation keeps running — response streams here</div>
</div>""", unsafe_allow_html=True)
                else:
                    _cursor    = "<span style='color:#58a6ff;font-weight:700;animation:blink 1s step-end infinite'>▌</span>" if _streaming else ""
                    _safe_text = _text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    st.markdown(f"""
<div style="background:#0d1117;border-radius:8px;padding:14px;
            font-family:'JetBrains Mono',monospace;font-size:12.5px;
            color:#c9d1d9;white-space:pre-wrap;line-height:1.8;
            border-left:3px solid {_color}">{_safe_text}{_cursor}</div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
<div style="background:#0d1117;border-radius:8px;padding:14px">
    <div style="color:#8b949e;font-size:11px;margin-bottom:8px">MODEL PROBABILITIES</div>
    <div style="margin:6px 0">
        <div style="color:#2ecc71;font-size:12px">🟢 Normal</div>
        <div style="background:#21262d;border-radius:4px;height:8px;margin:3px 0">
            <div style="background:#2ecc71;width:{card.get('p_normal',0):.0f}%;height:8px;border-radius:4px"></div>
        </div>
        <div style="color:#8b949e;font-size:11px">{card.get('p_normal',0):.1f}%</div>
    </div>
    <div style="margin:6px 0">
        <div style="color:#e74c3c;font-size:12px">🔴 DDoS</div>
        <div style="background:#21262d;border-radius:4px;height:8px;margin:3px 0">
            <div style="background:#e74c3c;width:{card.get('p_ddos',0):.0f}%;height:8px;border-radius:4px"></div>
        </div>
        <div style="color:#8b949e;font-size:11px">{card.get('p_ddos',0):.1f}%</div>
    </div>
    <div style="margin:6px 0">
        <div style="color:#9b59b6;font-size:12px">🟣 Botnet</div>
        <div style="background:#21262d;border-radius:4px;height:8px;margin:3px 0">
            <div style="background:#9b59b6;width:{card.get('p_botnet',0):.0f}%;height:8px;border-radius:4px"></div>
        </div>
        <div style="color:#8b949e;font-size:11px">{card.get('p_botnet',0):.1f}%</div>
    </div>
    <hr style="border-color:#21262d;margin:10px 0">
    <div style="color:#8b949e;font-size:11px">{card['phase']}</div>
</div>""", unsafe_allow_html=True)

# ─── ALERT TABLE ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📋 Attack Event Log")

alerts = st.session_state.alerts
if not alerts:
    st.markdown("""
<div style="background:#161b22;border:1px dashed #30363d;border-radius:10px;
            padding:20px;text-align:center;color:#8b949e">
    No attacks detected yet
</div>""", unsafe_allow_html=True)
else:
    rows = ""
    for a in reversed(alerts[-20:]):
        _clr = COLORS.get(a["label"], "#888")
        _ico = ICONS.get(a["label"], "")
        rows += f"""<tr style="border-bottom:1px solid #21262d">
            <td style="padding:8px 14px;color:#8b949e;font-family:monospace">{a['time']}</td>
            <td style="padding:8px 14px;color:{_clr};font-weight:700">{_ico} {a['label']}</td>
            <td style="padding:8px 14px;color:white;font-weight:600">{a['confidence']:.1f}%</td>
            <td style="padding:8px 14px;color:#8b949e">#{a['flow']}</td>
            <td style="padding:8px 14px;color:#8b949e">{a['phase']}</td>
        </tr>"""
    st.markdown(f"""
<div style="border-radius:10px;overflow:hidden;border:1px solid #30363d">
<table style="width:100%;border-collapse:collapse;background:#161b22">
    <thead><tr style="background:#21262d">
        <th style="padding:10px 14px;text-align:left;color:#8b949e;font-weight:600">Time</th>
        <th style="padding:10px 14px;text-align:left;color:#8b949e;font-weight:600">Attack Type</th>
        <th style="padding:10px 14px;text-align:left;color:#8b949e;font-weight:600">Confidence</th>
        <th style="padding:10px 14px;text-align:left;color:#8b949e;font-weight:600">Flow</th>
        <th style="padding:10px 14px;text-align:left;color:#8b949e;font-weight:600">Phase</th>
    </tr></thead>
    <tbody>{rows}</tbody>
</table></div>""", unsafe_allow_html=True)

if st.session_state.done:
    st.success("✅ Simulation complete! All phases finished.")

# ─── SIMULATION STEP ── MUST BE LAST (st.rerun halts all code after it) ───────
if st.session_state.running:
    ss  = st.session_state
    idx = ss.scenario_idx

    if idx >= len(SCENARIO):
        ss.running = False
        ss.done    = True
    else:
        t_type, duration, phase_desc = SCENARIO[idx]
        ss.current_phase = phase_desc

        if time.time() - ss.phase_start > duration:
            ss.scenario_idx += 1
            ss.phase_start   = time.time()
            if ss.scenario_idx >= len(SCENARIO):
                ss.running = False
                ss.done    = True
            else:
                t_type, duration, phase_desc = SCENARIO[ss.scenario_idx]
                ss.current_phase = phase_desc

    if ss.running:
        flow = get_flow(t_type)
        ss.buffer.append(flow)
        if len(ss.buffer) > WINDOW_SIZE:
            ss.buffer.pop(0)

        if len(ss.buffer) == WINDOW_SIZE:
            ss.flow_count += 1
            fc = ss.flow_count
            cls, conf, probs = predict(model, np.array(ss.buffer))
            lbl = LABELS[cls]

            ss.current_label = lbl
            ss.current_conf  = conf
            ss.current_probs = probs.tolist()
            ss.stats[lbl]   += 1
            ss.history.append({"flow": fc, "label": lbl, "conf": conf})

            ts  = datetime.now().strftime("%H:%M:%S")
            clr = {"Normal": "#2ecc71", "DDoS": "#e74c3c", "Botnet": "#9b59b6"}[lbl]
            ss.log_lines.append(
                f'<span style="color:#8b949e">[{ts}]</span> '
                f'<span style="color:{clr};font-weight:bold">#{fc:>4} {lbl:8}</span> '
                f'<span style="color:white">{conf*100:.1f}%</span> '
                f'<span style="color:#8b949e">{phase_desc}</span>'
            )

            if cls != 0 and conf >= 0.85:
                ss.alerts.append({
                    "time": ts, "label": lbl,
                    "confidence": round(conf * 100, 1),
                    "flow": fc, "phase": phase_desc,
                    "p_normal": round(float(probs[0]) * 100, 1),
                    "p_ddos":   round(float(probs[1]) * 100, 1),
                    "p_botnet": round(float(probs[2]) * 100, 1),
                })

                last_active = bool(ss.llm_cards) and (
                    ss.llm_cards[-1].get("response") is None or
                    ss.llm_cards[-1].get("streaming", False)
                )
                trigger = (
                    enable_llm and
                    not last_active and
                    (ss.last_llm_label != lbl or fc - ss.last_llm_flow >= 20)
                )
                if trigger:
                    ss.last_llm_label = lbl
                    ss.last_llm_flow  = fc
                    new_card = {
                        "label"     : lbl,
                        "confidence": round(conf * 100, 1),
                        "flow"      : fc,
                        "time"      : ts,
                        "phase"     : phase_desc,
                        "response"  : None,
                        "streaming" : False,
                        "p_normal"  : round(float(probs[0]) * 100, 1),
                        "p_ddos"    : round(float(probs[1]) * 100, 1),
                        "p_botnet"  : round(float(probs[2]) * 100, 1),
                    }
                    ss.llm_cards.append(new_card)
                    _a, _c = lbl, conf * 100
                    _p0, _p1, _p2 = float(probs[0]), float(probs[1]), float(probs[2])
                    def _run_llm(card=new_card, a=_a, c=_c, p0=_p0, p1=_p1, p2=_p2):
                        card["response"]  = ""
                        card["streaming"] = True
                        for token in stream_attack_analysis(a, c, p0, p1, p2):
                            card["response"] += token
                        card["streaming"] = False
                    threading.Thread(target=_run_llm, daemon=True).start()

        time.sleep(flow_speed)
        st.rerun()   # ← always last — halts execution after this point