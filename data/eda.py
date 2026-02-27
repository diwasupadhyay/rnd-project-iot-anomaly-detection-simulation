"""
STEP 1a: EDA — CICIoT2023 (UNB/CIC Official Dataset)
=======================================================
Dataset : https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset
Structure: 169 CSV files in data/raw/, label column = 'label'
Focus    : DDoS (11 subtypes + DoS), Botnet/Mirai (3 subtypes), Normal (BenignTraffic)

Run: python data/eda.py
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

RAW_DATA_DIR = "data/raw"
PLOTS_DIR    = "data/eda_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── EXACT label strings from CICIoT2023 paper (Neto et al.) ──────────────────
LABEL_MAP = {
    "BenignTraffic"           : "Normal",
    # DDoS variants
    "DDoS-RSTFINFlood"        : "DDoS",
    "DDoS-PSHACK_Flood"       : "DDoS",
    "DDoS-SYN_Flood"          : "DDoS",
    "DDoS-UDP_Flood"          : "DDoS",
    "DDoS-TCP_Flood"          : "DDoS",
    "DDoS-ICMP_Flood"         : "DDoS",
    "DDoS-HTTP_Flood"         : "DDoS",
    "DDoS-SlowLoris"          : "DDoS",
    "DDoS-ICMP_Fragmentation" : "DDoS",
    "DDoS-UDP_Fragmentation"  : "DDoS",
    "DDoS-ACK_Fragmentation"  : "DDoS",
    # DoS (treated same as DDoS for our model)
    "DoS-UDP_Flood"           : "DDoS",
    "DoS-SYN_Flood"           : "DDoS",
    "DoS-TCP_Flood"           : "DDoS",
    "DoS-HTTP_Flood"          : "DDoS",
    # Botnet — Mirai variants
    "Mirai-greeth_flood"      : "Botnet",
    "Mirai-greip_flood"       : "Botnet",
    "Mirai-udpplain"          : "Botnet",
    # Everything else is out of scope for our 3-class model
    "Recon-HostDiscovery"     : "Other",
    "Recon-OSScan"            : "Other",
    "Recon-PortScan"          : "Other",
    "VulnerabilityScan"       : "Other",
    "Recon-PingSweep"         : "Other",
    "BrowserHijacking"        : "Other",
    "CommandInjection"        : "Other",
    "SqlInjection"            : "Other",
    "XSS"                     : "Other",
    "Uploading_Attack"        : "Other",
    "Backdoor_Malware"        : "Other",
    "MITM-ArpSpoofing"        : "Other",
    "DNS_Spoofing"            : "Other",
    "DictionaryBruteForce"    : "Other",
}

NUMERIC_MAP  = {"Normal": 0, "DDoS": 1, "Botnet": 2}
CLASS_COLORS = {"Normal": "#2ecc71", "DDoS": "#e74c3c", "Botnet": "#9b59b6"}


def load_sample(data_dir, max_files=20, rows_per_file=5000):
    """Load a representative sample for EDA (full load happens in preprocess.py)."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))

    if not csv_files:
        raise FileNotFoundError(
            "\n❌  No CSV files found in data/raw/\n"
            "    Make sure you unzipped the Kaggle download into data/raw/\n"
            "    Expected files like: data/raw/part-00000-....csv"
        )

    total = len(csv_files)
    sample_files = csv_files[:max_files]
    print(f"📂  Total CSV files : {total}")
    print(f"    Loading sample  : {len(sample_files)} files × {rows_per_file} rows")

    dfs = []
    for f in sample_files:
        df = pd.read_csv(f, nrows=rows_per_file, low_memory=False)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"    Sample size     : {len(combined):,} rows, {combined.shape[1]} columns")
    return combined


def inspect_labels(df):
    print("\n" + "─"*60)
    print("🏷️   RAW LABELS in your dataset:")
    print("─"*60)
    counts = df["label"].value_counts()
    for lbl, cnt in counts.items():
        mapped = LABEL_MAP.get(str(lbl).strip(), "❓ NOT IN MAP — check spelling")
        print(f"  {str(lbl):<35} {cnt:>7,}   →  {mapped}")
    return counts


def map_labels(df):
    df["label_name"] = df["label"].astype(str).str.strip().map(LABEL_MAP).fillna("Other")
    df["label_num"]  = df["label_name"].map(NUMERIC_MAP).fillna(-1).astype(int)
    df = df[df["label_name"] != "Other"].copy()
    print(f"\n✅  Rows after filtering to Normal/DDoS/Botnet: {len(df):,}")
    return df


def run_eda(df):
    counts = df["label_name"].value_counts()
    print("\n📊  Class distribution in sample:")
    for name, cnt in counts.items():
        bar = "█" * int(cnt / counts.max() * 35)
        print(f"    {name:8s}  {cnt:>7,}  {bar}")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in ["label_num"]]

    # Correlation with label
    corr = df[feat_cols].corrwith(df["label_num"]).abs().sort_values(ascending=False)
    top20 = corr.nlargest(20).index.tolist()
    top6  = corr.nlargest(6).index.tolist()

    # ── Plot 1: Class distribution ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CICIoT2023 — Class Distribution", fontsize=14, fontweight="bold")

    bar_colors = [CLASS_COLORS.get(n, "#95a5a6") for n in counts.index]
    counts.plot(kind="bar", ax=ax1, color=bar_colors, edgecolor="white")
    ax1.set_title("Count per Class"); ax1.set_xlabel(""); ax1.tick_params(axis="x", rotation=0)
    for p in ax1.patches:
        ax1.annotate(f"{int(p.get_height()):,}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=10)

    counts.plot(kind="pie", ax=ax2, autopct="%1.1f%%",
                colors=bar_colors, startangle=90)
    ax2.set_title("Proportion"); ax2.set_ylabel("")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n💾  Saved → {PLOTS_DIR}/01_class_distribution.png")

    # ── Plot 2: Correlation heatmap (top 20 features) ────────────────────────
    corr_matrix = df[top20].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm",
                center=0, linewidths=0.3, ax=ax)
    ax.set_title("Top-20 Features — Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾  Saved → {PLOTS_DIR}/02_correlation_heatmap.png")

    # ── Plot 3: Feature distributions per class ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    for ax, feat in zip(axes, top6):
        for lname, grp in df.groupby("label_name"):
            v = grp[feat].replace([np.inf, -np.inf], np.nan).dropna()
            v = v.clip(v.quantile(0.01), v.quantile(0.99))
            ax.hist(v, bins=60, alpha=0.55, label=lname,
                    density=True, color=CLASS_COLORS.get(lname, "#95a5a6"))
        ax.set_title(feat[:28], fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
    fig.suptitle("Top-6 Feature Distributions by Attack Class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾  Saved → {PLOTS_DIR}/03_feature_distributions.png")

    # ── Plot 4: Top-20 feature importances ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    corr.nlargest(20).sort_values().plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title("Top-20 Features — Correlation with Label", fontweight="bold")
    ax.set_xlabel("Absolute Correlation")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾  Saved → {PLOTS_DIR}/04_feature_importance.png")

    print(f"\n✅  EDA complete! Open data/eda_plots/ to view all charts.")
    print(f"\n    Top 5 most discriminating features:")
    for f in corr.nlargest(5).index:
        print(f"    • {f}  (|corr| = {corr[f]:.3f})")

    return corr.nlargest(50).index.tolist()


if __name__ == "__main__":
    print("="*60)
    print("🔬 CICIoT2023 — Exploratory Data Analysis")
    print("="*60)

    df = load_sample(RAW_DATA_DIR)
    inspect_labels(df)
    df = map_labels(df)
    run_eda(df)

    print("\n📌  NEXT STEP: python data/preprocess.py")
