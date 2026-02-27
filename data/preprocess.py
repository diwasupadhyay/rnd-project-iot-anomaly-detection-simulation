"""
STEP 1b: Preprocessing Pipeline — CICIoT2023
=============================================
Handles 169 CSV files (~3GB) with memory-efficient chunked loading.
Outputs train/val/test .npy files ready for CNN-LSTM training.

Run: python data/preprocess.py

Outputs → data/processed/
  X_train.npy, X_val.npy, X_test.npy    shape: (N, window=10, features=50)
  y_train.npy, y_val.npy, y_test.npy    dtype: int64  (0=Normal,1=DDoS,2=Botnet)
  scaler.pkl                              MinMaxScaler (save for inference)
  feature_names.txt                       ordered feature list
  class_weights.npy                       for weighted loss in training
"""

import os, glob, gc, warnings
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RAW_DATA_DIR  = "data/raw"
OUTPUT_DIR    = "data/processed"
WINDOW_SIZE   = 10           # LSTM timesteps
TOP_N_FEATS   = 50           # top features by correlation
ROWS_PER_FILE = 3000         # rows to sample per CSV file (169 × 3000 = ~507k rows)
                             # increase if you have 16GB+ RAM: try 10000
RANDOM_STATE  = 42
VAL_SIZE      = 0.15
TEST_SIZE     = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Label map (exact strings from CICIoT2023) ─────────────────────────────────
LABEL_MAP = {
    "BenignTraffic"           : 0,   # Normal
    "DDoS-RSTFINFlood"        : 1,   # DDoS
    "DDoS-PSHACK_Flood"       : 1,
    "DDoS-SYN_Flood"          : 1,
    "DDoS-UDP_Flood"          : 1,
    "DDoS-TCP_Flood"          : 1,
    "DDoS-ICMP_Flood"         : 1,
    "DDoS-HTTP_Flood"         : 1,
    "DDoS-SlowLoris"          : 1,
    "DDoS-ICMP_Fragmentation" : 1,
    "DDoS-UDP_Fragmentation"  : 1,
    "DDoS-ACK_Fragmentation"  : 1,
    "DoS-UDP_Flood"           : 1,
    "DoS-SYN_Flood"           : 1,
    "DoS-TCP_Flood"           : 1,
    "DoS-HTTP_Flood"          : 1,
    "Mirai-greeth_flood"      : 2,   # Botnet
    "Mirai-greip_flood"       : 2,
    "Mirai-udpplain"          : 2,
    # All other attack types → -1 (filtered out)
}

# Columns that are identifiers, not features
DROP_COLS = {
    "flow_id", "src_ip", "dst_ip", "source_ip", "destination_ip",
    "src_port", "dst_port", "source_port", "destination_port",
    "timestamp", "protocol",
}


# ─── 1. LOAD ALL FILES ────────────────────────────────────────────────────────
def load_all_files(data_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/")

    print(f"📂  Found {len(csv_files)} CSV files")
    print(f"    Sampling {ROWS_PER_FILE} rows per file → ~{len(csv_files)*ROWS_PER_FILE:,} rows total\n")

    dfs = []
    label_totals = {0: 0, 1: 0, 2: 0}

    for f in tqdm(csv_files, desc="  Loading CSVs", unit="file"):
        try:
            df = pd.read_csv(f, low_memory=False)

            # Map label → numeric, drop unknown
            if "label" not in df.columns:
                continue
            df["label"] = df["label"].astype(str).str.strip().map(LABEL_MAP)
            df = df[df["label"].notna() & (df["label"] >= 0)].copy()
            df["label"] = df["label"].astype(int)

            if len(df) == 0:
                continue

            # Stratified sample from this file
            sampled = df.groupby("label", group_keys=False).apply(
                lambda x: x.sample(min(len(x), ROWS_PER_FILE // 3),
                                   random_state=RANDOM_STATE)
            )
            dfs.append(sampled)

            for lbl, cnt in sampled["label"].value_counts().items():
                label_totals[lbl] = label_totals.get(lbl, 0) + cnt

        except Exception as e:
            print(f"\n  ⚠️  Skipping {os.path.basename(f)}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✅  Total rows loaded : {len(combined):,}")
    name_map = {0: "Normal", 1: "DDoS", 2: "Botnet"}
    for lbl, cnt in sorted(label_totals.items()):
        print(f"    {name_map[lbl]:8s} : {cnt:>10,}")
    return combined


# ─── 2. CLEAN ─────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    print("\n🧹  Cleaning...")

    # Drop identifier columns
    drop = [c for c in df.columns if c.lower().strip() in DROP_COLS]
    df.drop(columns=drop, errors="ignore", inplace=True)

    # Keep only numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c != "label"]
    df = df[feat_cols + ["label"]].copy()

    # Replace inf → NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns with >30% missing
    miss_rate = df[feat_cols].isnull().mean()
    bad = miss_rate[miss_rate > 0.30].index.tolist()
    if bad:
        print(f"    Dropped {len(bad)} columns with >30% missing")
        df.drop(columns=bad, inplace=True)
        feat_cols = [c for c in df.columns if c != "label"]

    # Fill remaining NaN with median
    medians = df[feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(medians)

    # Drop near-zero variance
    var = df[feat_cols].var()
    low_var = var[var < 1e-9].index.tolist()
    if low_var:
        print(f"    Dropped {len(low_var)} near-zero variance columns")
        df.drop(columns=low_var, inplace=True)
        feat_cols = [c for c in df.columns if c != "label"]

    # Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"    Removed {before - len(df):,} duplicate rows")
    print(f"    Remaining features: {len(feat_cols)}, rows: {len(df):,}")

    return df, feat_cols


# ─── 3. FEATURE SELECTION ─────────────────────────────────────────────────────
def select_features(df: pd.DataFrame, feat_cols: list) -> tuple[pd.DataFrame, list]:
    print(f"\n🎯  Selecting top {TOP_N_FEATS} features by |correlation with label|...")
    corr = df[feat_cols].corrwith(df["label"]).abs()
    top  = corr.nlargest(TOP_N_FEATS).index.tolist()
    df   = df[top + ["label"]]
    print(f"    Top 3 features: {top[:3]}")
    with open(os.path.join(OUTPUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(top))
    return df, top


# ─── 4. SPLIT ─────────────────────────────────────────────────────────────────
def split(df, feature_names):
    print("\n✂️   Splitting 70 / 15 / 15 ...")
    X = df[feature_names].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y)

    X_v, X_te, y_v, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_tmp)

    print(f"    Train: {len(X_tr):>7,}  Val: {len(X_v):>7,}  Test: {len(X_te):>7,}")
    return X_tr, X_v, X_te, y_tr, y_v, y_te


# ─── 5. SCALE ─────────────────────────────────────────────────────────────────
def scale(X_tr, X_v, X_te):
    print("\n📐  Scaling with MinMaxScaler...")
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_v  = scaler.transform(X_v)
    X_te = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print("    Scaler saved → data/processed/scaler.pkl")
    return X_tr, X_v, X_te, scaler


# ─── 6. CLASS BALANCE CHECK (no SMOTE — dataset is large enough) ──────────────
def check_balance(y_tr):
    print("\n📊  Training class balance:")
    name_map = {0: "Normal", 1: "DDoS", 2: "Botnet"}
    total = len(y_tr)
    for cls in sorted(np.unique(y_tr)):
        cnt = (y_tr == cls).sum()
        print(f"    {name_map[cls]:8s} : {cnt:>8,}  ({cnt/total*100:.1f}%)")

    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    np.save(os.path.join(OUTPUT_DIR, "class_weights.npy"), weights)
    print(f"\n    Class weights saved (used in training loss)")
    print(f"    Weights: { {name_map[int(c)]: round(w, 3) for c, w in zip(classes, weights)} }")


# ─── 7. SLIDING WINDOW SEQUENCES ─────────────────────────────────────────────
def make_sequences(X: np.ndarray, y: np.ndarray, window: int = WINDOW_SIZE):
    """
    Convert flat rows → overlapping windows for LSTM.
    Input : (N, features)
    Output: (N - window + 1,  window,  features)
    Label : class at the LAST timestep of each window
    """
    n = len(X) - window + 1
    Xs = np.lib.stride_tricks.sliding_window_view(X, (window, X.shape[1]))[..., 0, :, :]
    # sliding_window_view returns (n, 1, window, features) — reshape
    Xs = Xs.reshape(n, window, X.shape[1]).astype(np.float32)
    ys = y[window - 1:].astype(np.int64)
    return Xs, ys


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("🚀  CICIoT2023 — Preprocessing Pipeline")
    print("="*60)

    # 1. Load
    df = load_all_files(RAW_DATA_DIR)

    # 2. Clean
    df, feat_cols = clean(df)

    # 3. Feature selection
    df, feature_names = select_features(df, feat_cols)

    # 4. Split  (before scaling to avoid leakage)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split(df, feature_names)
    del df; gc.collect()

    # 5. Scale
    X_tr, X_v, X_te, _ = scale(X_tr, X_v, X_te)

    # 6. Balance info + class weights
    check_balance(y_tr)

    # 7. Create sequences
    print(f"\n🔄  Creating sliding windows (size={WINDOW_SIZE})...")
    X_tr_s, y_tr_s = make_sequences(X_tr, y_tr)
    X_v_s,  y_v_s  = make_sequences(X_v,  y_v)
    X_te_s, y_te_s = make_sequences(X_te, y_te)

    print(f"\n    X_train : {X_tr_s.shape}   ← (samples, timesteps, features)")
    print(f"    X_val   : {X_v_s.shape}")
    print(f"    X_test  : {X_te_s.shape}")

    # 8. Save
    print(f"\n💾  Saving to {OUTPUT_DIR}/...")
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_tr_s)
    np.save(f"{OUTPUT_DIR}/X_val.npy",   X_v_s)
    np.save(f"{OUTPUT_DIR}/X_test.npy",  X_te_s)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_tr_s)
    np.save(f"{OUTPUT_DIR}/y_val.npy",   y_v_s)
    np.save(f"{OUTPUT_DIR}/y_test.npy",  y_te_s)

    print("\n✅  Preprocessing complete!")
    print("📌  NEXT STEP: python data/verify.py  →  then  python models/train.py")
    print("="*60)


if __name__ == "__main__":
    main()