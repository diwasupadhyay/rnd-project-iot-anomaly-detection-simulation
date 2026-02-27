"""
detection/detector.py
======================
Loads the trained CNN-LSTM model and classifies network flows.
Can run on:
  A) A saved simulation CSV file (--file)
  B) Live real-time stream from simulation (imported as module)

Run on a simulation log:
  python detection/detector.py --file simulation/logs/ddos_simulation.csv
  python detection/detector.py --file simulation/logs/botnet_simulation.csv
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import torch

# Allow imports from models/ folder
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
sys.path.insert(0, MODELS_DIR)

from cnn_lstm import CNNLSTM

MODEL_PATH  = os.path.join(BASE_DIR, "models", "saved", "best_model.pth")
FEAT_PATH   = os.path.join(BASE_DIR, "data", "processed", "feature_names.txt")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 10
LABELS      = {0: "Normal", 1: "DDoS", 2: "Botnet"}
ICONS       = {0: "🟢", 1: "🔴", 2: "🟣"}


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CNNLSTM(
        n_features=ckpt.get("n_features", 38),
        n_classes =ckpt.get("n_classes",  3),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅  Model loaded (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']*100:.2f}%)")
    return model


# ─── LOAD FEATURE NAMES ───────────────────────────────────────────────────────
def load_features():
    with open(FEAT_PATH) as f:
        return f.read().splitlines()


# ─── PREDICT SINGLE WINDOW ────────────────────────────────────────────────────
@torch.no_grad()
def predict_window(model, window: np.ndarray):
    """
    window: (WINDOW_SIZE, N_FEATURES) numpy array
    returns: (predicted_class, confidence, all_probs)
    """
    x     = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    cls    = int(probs.argmax())
    return cls, float(probs[cls]), probs


# ─── RUN DETECTION ON A CSV FILE ──────────────────────────────────────────────
def detect_from_file(csv_path: str, model, features: list, delay: float = 0.1):
    if not os.path.exists(csv_path):
        print(f"❌  File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\n📂  Loaded: {os.path.basename(csv_path)}  ({len(df)} flows)")

    # Keep only known feature columns
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"⚠️   {len(missing)} features missing in file — filling with 0")
    
    X = np.zeros((len(df), len(features)), dtype=np.float32)
    for i, feat in enumerate(features):
        if feat in df.columns:
            X[:, i] = df[feat].fillna(0).values

    true_labels = df["label"].values if "label" in df.columns else None

    print(f"\n{'─'*70}")
    print(f"  {'Flow':>5}  {'Predicted':>10}  {'Conf':>6}  {'True':>10}  {'Match':>5}  Probs")
    print(f"{'─'*70}")

    # Sliding window inference
    results = []
    n_correct, n_total = 0, 0

    for i in range(WINDOW_SIZE - 1, len(X)):
        window = X[i - WINDOW_SIZE + 1 : i + 1]
        cls, conf, probs = predict_window(model, window)

        true_lbl = str(true_labels[i]) if true_labels is not None else "?"
        match    = "✅" if true_lbl == LABELS[cls] else "❌"
        icon     = ICONS[cls]

        print(f"  {i+1:>5}  {icon} {LABELS[cls]:>8}  {conf*100:>5.1f}%  "
              f"{true_lbl:>10}  {match}  "
              f"[N:{probs[0]:.2f} D:{probs[1]:.2f} B:{probs[2]:.2f}]")

        results.append({
            "flow_idx"  : i + 1,
            "predicted" : LABELS[cls],
            "confidence": round(conf, 4),
            "true_label": true_lbl,
            "prob_normal": round(float(probs[0]), 4),
            "prob_ddos"  : round(float(probs[1]), 4),
            "prob_botnet": round(float(probs[2]), 4),
        })

        if true_labels is not None:
            n_correct += int(true_lbl == LABELS[cls])
            n_total   += 1

        time.sleep(delay)

    print(f"{'─'*70}")

    if n_total > 0:
        acc = n_correct / n_total * 100
        print(f"\n🎯  Detection Accuracy: {acc:.2f}%  ({n_correct}/{n_total})")

    # Count alerts
    pred_counts = pd.Series([r["predicted"] for r in results]).value_counts()
    print(f"\n📊  Detection Summary:")
    for label, count in pred_counts.items():
        icon = ICONS[{"Normal":0,"DDoS":1,"Botnet":2}.get(label,0)]
        print(f"    {icon} {label:10s}: {count} flows")

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IoT Anomaly Detector")
    parser.add_argument("--file", type=str, required=True,
                        help="Path to simulation CSV log")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between predictions (seconds)")
    args = parser.parse_args()

    print("="*70)
    print("🛡️   IoT Anomaly Detection — CNN-LSTM Inference")
    print(f"    Device: {DEVICE.upper()}")
    print("="*70)

    model    = load_model()
    features = load_features()
    detect_from_file(args.file, model, features, delay=args.delay)

    print("\n📌  NEXT: python detection/realtime_pipeline.py")


if __name__ == "__main__":
    main()
