"""
STEP 2c: Model Evaluation — CICIoT2023
========================================
Loads best_model.pth and evaluates on the held-out test set.
Outputs:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix plot
  - Per-class accuracy bar chart
  - ROC curves (one-vs-rest)

Run: python models/evaluate.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from cnn_lstm import CNNLSTM

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved", "best_model.pth")
PLOTS_DIR  = os.path.join(BASE_DIR, "models", "plots")
BATCH_SIZE = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LABELS     = ["Normal", "DDoS", "Botnet"]
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}\nRun: python models/train.py first")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CNNLSTM(
        n_features=ckpt.get("n_features", 38),
        n_classes =ckpt.get("n_classes",  3),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅  Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']*100:.2f}%)")
    return model


# ─── PREDICT ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model):
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds  = []
    all_probs  = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits  = model(X_batch)
        probs   = torch.softmax(logits, dim=1).cpu().numpy()
        preds   = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


# ─── PLOT: CONFUSION MATRIX ───────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN-LSTM — Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=axes[0])
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=axes[1])
    axes[1].set_title("Row-Normalized (%)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

    plt.tight_layout()
    path = f"{PLOTS_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾  Saved → {path}")


# ─── PLOT: ROC CURVES ─────────────────────────────────────────────────────────
def plot_roc(y_true, y_probs):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = ["#2ecc71", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(LABELS, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_title("ROC Curves — One-vs-Rest", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾  Saved → {path}")


# ─── PLOT: PER-CLASS ACCURACY ─────────────────────────────────────────────────
def plot_per_class_accuracy(y_true, y_pred):
    colors = ["#2ecc71", "#e74c3c", "#9b59b6"]
    accs   = []
    for cls in range(3):
        mask = y_true == cls
        acc  = (y_pred[mask] == cls).mean() * 100
        accs.append(acc)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(LABELS, accs, color=colors, edgecolor="white", width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Per-Class Detection Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/per_class_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾  Saved → {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("📊  CNN-LSTM Evaluation — CICIoT2023 Test Set")
    print(f"    Device: {DEVICE.upper()}")
    print("="*60)

    model = load_model()
    print("\n🔄  Running inference on test set...")
    y_true, y_pred, y_probs = predict(model)

    # Overall accuracy
    overall_acc = (y_true == y_pred).mean() * 100
    print(f"\n🎯  Overall Test Accuracy : {overall_acc:.2f}%")

    # Classification report
    print("\n📋  Classification Report:")
    print("─"*60)
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))

    # Per-class breakdown
    print("📊  Per-Class Accuracy:")
    for cls, name in enumerate(LABELS):
        mask = y_true == cls
        acc  = (y_pred[mask] == cls).mean() * 100
        n    = mask.sum()
        print(f"    {name:8s}: {acc:6.2f}%  ({n:,} samples)")

    # Plots
    print("\n🖼️   Generating plots...")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc(y_true, y_probs)
    plot_per_class_accuracy(y_true, y_pred)

    print("\n✅  Evaluation complete!")
    print(f"    All plots saved → {PLOTS_DIR}/")
    print("\n📌  NEXT STEP: python simulation/ddos_attack.py")
    print("="*60)


if __name__ == "__main__":
    main()