"""
STEP 1c: Verify Processed Data
================================
Run after preprocess.py to confirm everything is correct
before building the model.

Run: python data/verify.py
"""

import os
import numpy as np
import joblib

OUTPUT_DIR = "data/processed"
LABEL_MAP  = {0: "Normal", 1: "DDoS", 2: "Botnet"}


def check(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    if not os.path.exists(path):
        print(f"  ❌ Missing: {path}")
        return None
    arr = np.load(path)
    size_mb = arr.nbytes / 1e6
    print(f"  ✅ {name:<15} shape={str(arr.shape):<30} dtype={arr.dtype}  ({size_mb:.1f} MB)")
    return arr


def main():
    print("="*65)
    print("🔍  Verifying Preprocessed Data — CICIoT2023")
    print("="*65)

    required = ["X_train.npy", "y_train.npy", "scaler.pkl", "feature_names.txt"]
    missing  = [f for f in required if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    if missing:
        print(f"❌  Missing files: {missing}")
        print("    Run: python data/preprocess.py first")
        return

    print("\n📦  Array shapes & sizes:")
    X_train = check("X_train")
    X_val   = check("X_val")
    X_test  = check("X_test")
    y_train = check("y_train")
    y_val   = check("y_val")
    y_test  = check("y_test")

    print("\n📊  Class distributions:")
    for split_name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        if y is None: continue
        total = len(y)
        print(f"\n  {split_name}:")
        for cls in sorted(np.unique(y)):
            cnt = (y == cls).sum()
            bar = "█" * int(cnt / total * 40)
            print(f"    {LABEL_MAP[cls]:8s} ({cls}): {cnt:>8,}  {cnt/total*100:5.1f}%  {bar}")

    print("\n📐  Model input info:")
    if X_train is not None:
        _, window, n_feats = X_train.shape
        print(f"    Window size    : {window} timesteps")
        print(f"    Feature count  : {n_feats}")
        print(f"    Value range    : [{X_train.min():.4f}, {X_train.max():.4f}]  (should be ~0–1)")

    feat_path = os.path.join(OUTPUT_DIR, "feature_names.txt")
    with open(feat_path) as f:
        features = f.read().splitlines()
    print(f"\n    Feature names  : {features[:3]} ... (total {len(features)})")

    w_path = os.path.join(OUTPUT_DIR, "class_weights.npy")
    if os.path.exists(w_path):
        weights = np.load(w_path)
        print(f"\n⚖️   Class weights  : {dict(zip([LABEL_MAP[i] for i in range(len(weights))], weights.round(3)))}")

    print("\n" + "─"*65)
    if X_train is not None and y_train is not None:
        print("✅  Everything looks good! Data is ready for training.")
        print("📌  NEXT STEP: python models/train.py")
    print("="*65)


if __name__ == "__main__":
    main()
