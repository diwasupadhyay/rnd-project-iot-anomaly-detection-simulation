"""
STEP 2b: Training — CNN-LSTM on CICIoT2023
============================================
Features:
  - CUDA GPU acceleration
  - Weighted CrossEntropyLoss (handles any remaining imbalance)
  - ReduceLROnPlateau scheduler
  - Early stopping (patience=5)
  - Saves best model checkpoint
  - Live epoch progress with tqdm
  - Loss & accuracy plots saved automatically

Run: python models/train.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

from cnn_lstm import build_model

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models", "saved")
PLOTS_DIR   = os.path.join(BASE_DIR, "models", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

BATCH_SIZE   = 512       # Increase to 1024 if your GPU has 8GB+ VRAM
EPOCHS       = 30
LR           = 1e-3
PATIENCE     = 5         # Early stopping patience
N_FEATURES   = 38
N_CLASSES    = 3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_data():
    print("📦  Loading preprocessed data...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")

    print(f"    Train : {X_train.shape}  |  Val : {X_val.shape}")

    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    y_v  = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE=="cuda"))
    val_loader   = DataLoader(TensorDataset(X_v, y_v),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE=="cuda"))
    return train_loader, val_loader


# ─── WEIGHTED LOSS ────────────────────────────────────────────────────────────
def get_loss(data_dir: str):
    w_path = os.path.join(data_dir, "class_weights.npy")
    if os.path.exists(w_path):
        weights = torch.tensor(np.load(w_path), dtype=torch.float32).to(DEVICE)
        print(f"⚖️   Using class weights: {weights.cpu().numpy().round(3)}")
    else:
        weights = None
        print("⚖️   No class weights found — using equal weights")
    return nn.CrossEntropyLoss(weight=weights)


# ─── TRAIN ONE EPOCH ──────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ─── VALIDATE ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ─── PLOT TRAINING CURVES ─────────────────────────────────────────────────────
def save_plots(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN-LSTM Training — CICIoT2023", fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    if history.get("best_epoch"):
        ax1.axvline(history["best_epoch"], color="green", linestyle="--", label="Best")

    # Accuracy
    ax2.plot(epochs, [a*100 for a in history["train_acc"]], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, [a*100 for a in history["val_acc"]],   "r-o", label="Val Acc",   markersize=4)
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n💾  Training curves saved → {path}")


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────
def train():
    print("="*60)
    print("🚀  CNN-LSTM Training — CICIoT2023")
    print(f"    Device : {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"    GPU    : {torch.cuda.get_device_name(0)}")
        print(f"    VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)

    train_loader, val_loader = load_data()
    model     = build_model(n_features=N_FEATURES, n_classes=N_CLASSES, device=DEVICE)
    criterion = get_loss(DATA_DIR)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, verbose=True)

    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_ctr  = 0
    best_epoch    = 1
    start_time    = time.time()

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Loss':>10}  {'Val Acc':>10}  {'LR':>10}")
    print("─" * 65)

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {t_loss:>10.4f}  {t_acc*100:>9.2f}%  {v_loss:>10.4f}  {v_acc*100:>9.2f}%  {lr_now:>10.6f}")

        # Save best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch    = epoch
            patience_ctr  = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_loss"   : v_loss,
                "val_acc"    : v_acc,
                "n_features" : N_FEATURES,
                "n_classes"  : N_CLASSES,
            }, f"{MODELS_DIR}/best_model.pth")
            print(f"         ✅ New best model saved (val_loss={v_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n⏹️   Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    elapsed = time.time() - start_time
    history["best_epoch"] = best_epoch

    print("\n" + "="*60)
    print(f"✅  Training complete in {elapsed/60:.1f} minutes")
    print(f"    Best epoch    : {best_epoch}")
    print(f"    Best val loss : {best_val_loss:.4f}")
    print(f"    Best val acc  : {history['val_acc'][best_epoch-1]*100:.2f}%")
    print(f"    Model saved   → {MODELS_DIR}/best_model.pth")
    print("="*60)

    save_plots(history)
    print("\n📌  NEXT STEP: python models/evaluate.py")


if __name__ == "__main__":
    train()