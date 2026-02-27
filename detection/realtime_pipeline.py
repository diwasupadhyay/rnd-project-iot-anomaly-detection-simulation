"""
detection/realtime_pipeline.py
================================
Continuously generates simulated traffic, classifies it in real-time,
and prints live alerts to the console.

This is the core "real-time" demonstration for the project.
Simulates: Normal → DDoS → Normal → Botnet → Normal in a loop.

Run: python detection/realtime_pipeline.py

Press Ctrl+C to stop.
"""

import os, sys, time, queue, threading
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "models"))
sys.path.insert(0, os.path.join(BASE_DIR, "simulation"))

from cnn_lstm         import CNNLSTM
from traffic_simulator import generate_normal, generate_ddos, generate_botnet, FEATURES

MODEL_PATH  = os.path.join(BASE_DIR, "models", "saved", "best_model.pth")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 10
LABELS      = {0: "Normal", 1: "DDoS", 2: "Botnet"}
ICONS       = {0: "🟢", 1: "🔴", 2: "🟣"}

# Alert threshold — only alert when confidence > this
ALERT_THRESHOLD = 0.85

# ── Scenario: sequence of (traffic_type, duration_seconds) ────────────────────
SCENARIO = [
    ("normal",    8,  "Baseline IoT traffic"),
    ("syn_flood", 6,  "DDoS SYN Flood attack"),
    ("normal",    5,  "Traffic normalizing"),
    ("udp_flood", 6,  "DDoS UDP Flood attack"),
    ("normal",    4,  "Recovery"),
    ("botnet",    8,  "Mirai Botnet activity"),
    ("normal",    6,  "Traffic normalizing"),
]

FLOW_INTERVAL = 0.3    # seconds between flows


def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CNNLSTM(
        n_features=ckpt.get("n_features", 38),
        n_classes =ckpt.get("n_classes",  3),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, window: np.ndarray):
    x      = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    cls    = int(probs.argmax())
    return cls, float(probs[cls]), probs


def generate_flow(traffic_type: str) -> np.ndarray:
    if traffic_type == "normal":
        return generate_normal(1)[0]
    elif traffic_type == "syn_flood":
        return generate_ddos(1, attack_type="syn_flood")[0]
    elif traffic_type == "udp_flood":
        return generate_ddos(1, attack_type="udp_flood")[0]
    elif traffic_type == "botnet":
        return generate_botnet(1)[0]
    return generate_normal(1)[0]


def print_alert(cls: int, conf: float, probs: np.ndarray, flow_num: int,
                phase: str, prev_cls: int):
    icon  = ICONS[cls]
    label = LABELS[cls]
    ts    = time.strftime("%H:%M:%S")

    # Attack detected
    if cls != 0 and conf >= ALERT_THRESHOLD:
        print(f"\n{'!'*65}")
        print(f"  🚨 ALERT  [{ts}]  Flow #{flow_num}")
        print(f"  Attack Detected : {icon} {label.upper()}")
        print(f"  Confidence      : {conf*100:.1f}%")
        print(f"  Probabilities   : Normal={probs[0]:.2f}  DDoS={probs[1]:.2f}  Botnet={probs[2]:.2f}")
        print(f"  Phase           : {phase}")
        print(f"{'!'*65}\n")

    # Attack resolved
    elif cls == 0 and prev_cls != 0 and prev_cls is not None:
        print(f"\n  ✅ [{ts}] Attack subsided — traffic back to Normal\n")

    # Normal flow (brief)
    else:
        rate_i = next((i for i, f in enumerate(FEATURES) if "rate" in f.lower()), 0)
        print(f"  {icon} [{ts}] Flow #{flow_num:>4}  {label:8s}  conf={conf*100:.1f}%  "
              f"rate={generate_flow('normal' if cls==0 else 'syn_flood')[rate_i]:.2f}  Phase: {phase}")


def run():
    print("="*65)
    print("🛡️   Real-Time IoT Anomaly Detection Pipeline")
    print(f"    Device: {DEVICE.upper()}  |  Window: {WINDOW_SIZE} flows")
    print(f"    Alert threshold: {ALERT_THRESHOLD*100:.0f}% confidence")
    print("="*65)
    print("\nScenario sequence:")
    for t, dur, desc in SCENARIO:
        print(f"  • {desc:35s} ({dur}s)")
    print("\nStarting in 3 seconds... (Ctrl+C to stop)\n")
    time.sleep(3)

    model    = load_model()
    buffer   = []          # sliding window buffer
    flow_num = 0
    prev_cls = None

    try:
        for traffic_type, duration, phase_desc in SCENARIO:
            phase_start = time.time()
            print(f"\n{'─'*65}")
            print(f"  ▶  {phase_desc.upper()}")
            print(f"{'─'*65}")

            while time.time() - phase_start < duration:
                flow_num += 1
                flow = generate_flow(traffic_type)
                buffer.append(flow)

                if len(buffer) > WINDOW_SIZE:
                    buffer.pop(0)

                if len(buffer) == WINDOW_SIZE:
                    window      = np.array(buffer)
                    cls, conf, probs = predict(model, window)
                    print_alert(cls, conf, probs, flow_num, phase_desc, prev_cls)
                    prev_cls = cls
                else:
                    print(f"  ⏳ Filling window... ({len(buffer)}/{WINDOW_SIZE})")

                time.sleep(FLOW_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n⏹️   Pipeline stopped by user after {flow_num} flows.")

    print("\n✅  Real-time pipeline complete.")
    print("📌  NEXT STEP: python llm/interpreter.py")


if __name__ == "__main__":
    run()
