"""
detection/realtime_pipeline.py
================================
Real-time anomaly detection pipeline for IoT network traffic.
Simulates a multi-phase attack scenario and classifies each flow
using the trained CNN-LSTM model.

Run: python detection/realtime_pipeline.py
Press Ctrl+C to stop.
"""

import os, sys, time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))
sys.path.insert(0, os.path.join(BASE_DIR, "simulation"))

from cnn_lstm          import CNNLSTM
from traffic_simulator import generate_normal, generate_ddos, generate_botnet, FEATURES

MODEL_PATH  = os.path.join(BASE_DIR, "models", "saved", "best_model.pth")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 10
LABELS      = {0: "Normal", 1: "DDoS", 2: "Botnet"}
ICONS       = {0: "\u001b[32m\u25cf\u001b[0m",    # green circle
               1: "\u001b[31m\u25cf\u001b[0m",    # red circle
               2: "\u001b[35m\u25cf\u001b[0m"}    # purple circle
ALERT_THRESHOLD = 0.85

SCENARIO = [
    ("normal",    8,  "Normal Baseline"),
    ("syn_flood", 6,  "DDoS — SYN Flood"),
    ("normal",    5,  "Recovery"),
    ("udp_flood", 6,  "DDoS — UDP Flood"),
    ("normal",    4,  "Normalizing"),
    ("botnet",    8,  "Mirai Botnet"),
    ("normal",    6,  "Full Recovery"),
]

FLOW_INTERVAL = 0.3


def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CNNLSTM(
        n_features=ckpt.get("n_features", 38),
        n_classes=ckpt.get("n_classes", 3),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, window: np.ndarray):
    x     = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    cls   = int(probs.argmax())
    return cls, float(probs[cls]), probs


def generate_flow(traffic_type: str) -> np.ndarray:
    if traffic_type == "normal":
        return generate_normal(1)[0]
    if traffic_type == "syn_flood":
        return generate_ddos(1, attack_type="syn_flood")[0]
    if traffic_type == "udp_flood":
        return generate_ddos(1, attack_type="udp_flood")[0]
    if traffic_type == "botnet":
        return generate_botnet(1)[0]
    return generate_normal(1)[0]


def run():
    RED    = "\u001b[31m"
    GREEN  = "\u001b[32m"
    PURPLE = "\u001b[35m"
    YELLOW = "\u001b[33m"
    BOLD   = "\u001b[1m"
    DIM    = "\u001b[2m"
    RESET  = "\u001b[0m"

    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  \U0001f6e1\ufe0f  Real-Time IoT Anomaly Detection Pipeline{RESET}")
    print(f"{DIM}     Device: {DEVICE.upper()}  |  Window: {WINDOW_SIZE} flows  |  Threshold: {ALERT_THRESHOLD*100:.0f}%{RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}")
    print(f"\n{DIM}  Scenario:{RESET}")
    for t, dur, desc in SCENARIO:
        icon = GREEN if t == "normal" else (RED if "flood" in t else PURPLE)
        print(f"  {icon}\u25cf{RESET} {desc:30s} {DIM}({dur}s){RESET}")
    print(f"\n{DIM}  Starting in 2 seconds... (Ctrl+C to stop){RESET}\n")
    time.sleep(2)

    model    = load_model()
    buffer   = []
    flow_num = 0
    prev_cls = None
    alerts   = 0

    try:
        for traffic_type, duration, phase_desc in SCENARIO:
            print(f"\n{BOLD}{'─' * 65}")
            print(f"  \u25b6  {phase_desc.upper()}")
            print(f"{'─' * 65}{RESET}")

            phase_start = time.time()
            while time.time() - phase_start < duration:
                flow_num += 1
                flow = generate_flow(traffic_type)
                buffer.append(flow)
                if len(buffer) > WINDOW_SIZE:
                    buffer.pop(0)

                if len(buffer) < WINDOW_SIZE:
                    print(f"  {DIM}\u23f3 Filling window... ({len(buffer)}/{WINDOW_SIZE}){RESET}")
                    time.sleep(FLOW_INTERVAL)
                    continue

                cls, conf, probs = predict(model, np.array(buffer))
                label = LABELS[cls]
                ts    = time.strftime("%H:%M:%S")

                if cls != 0 and conf >= ALERT_THRESHOLD:
                    alerts += 1
                    color = RED if cls == 1 else PURPLE
                    print(f"\n  {color}{BOLD}\u2588\u2588 ALERT #{alerts} [{ts}] Flow #{flow_num}{RESET}")
                    print(f"  {color}{BOLD}   {label.upper()} detected  —  {conf*100:.1f}% confidence{RESET}")
                    print(f"  {DIM}   N:{probs[0]*100:.1f}%  D:{probs[1]*100:.1f}%  B:{probs[2]*100:.1f}%  |  {phase_desc}{RESET}\n")
                elif cls == 0 and prev_cls is not None and prev_cls != 0:
                    print(f"  {GREEN}\u2713 [{ts}] Traffic restored to Normal{RESET}")
                else:
                    icon = ICONS[cls]
                    print(f"  {icon} [{ts}] #{flow_num:>4}  {label:8s}  {conf*100:.1f}%  {DIM}{phase_desc}{RESET}")

                prev_cls = cls
                time.sleep(FLOW_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}\u23f9 Stopped by user after {flow_num} flows.{RESET}")

    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"  {GREEN}\u2714 Pipeline complete{RESET}")
    print(f"  {DIM}Flows analyzed: {flow_num}  |  Alerts raised: {alerts}{RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}\n")


if __name__ == "__main__":
    run()
