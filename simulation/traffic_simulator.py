"""
traffic_simulator.py — Real Data-Based Traffic Sampler
========================================================
Instead of synthetic feature generation, this samples REAL
feature vectors from the processed CICIoT2023 dataset.

This guarantees the model sees the same feature distributions
it was trained on → correct DDoS/Botnet/Normal detection.

Falls back to synthetic generation if no data files are available.

Used by: realtime_pipeline.py, ddos_attack.py, botnet_attack.py
"""

import numpy as np
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
FEAT_PATH = os.path.join(DATA_DIR, "feature_names.txt")
DEMO_POOL_PATH = os.path.join(DATA_DIR, "demo_pool.npz")

# ── Load feature names ────────────────────────────────────────────────────────
def get_feature_names():
    try:
        with open(FEAT_PATH) as f:
            names = [line.strip() for line in f if line.strip()]
            return names if names else _default_feature_names()
    except FileNotFoundError:
        return _default_feature_names()


def _default_feature_names():
    """Fallback feature names if feature_names.txt is missing."""
    return [
        "Variance", "ack_flag_number", "Protocol Type", "TCP", "HTTPS",
        "Duration", "syn_count", "Min", "rst_count", "Std", "Radius",
        "urg_count", "Max", "flow_duration", "UDP", "Covariance",
        "Header_Length", "Magnitue", "HTTP", "ack_count", "AVG",
        "Tot size", "Tot sum", "psh_flag_number", "Rate", "Srate",
        "DNS", "IPv", "LLC", "fin_count", "ARP", "Number", "Weight",
        "IAT", "ICMP", "syn_flag_number", "rst_flag_number", "fin_flag_number",
    ]


FEATURES = get_feature_names()
N_FEATS  = len(FEATURES)

# ── Load real test samples grouped by class ───────────────────────────────────
_pools  = {}   # {0: array of Normal rows, 1: DDoS rows, 2: Botnet rows}
_use_synthetic = False

def _load_pool():
    global _pools, _use_synthetic
    if _pools:
        return

    if os.path.exists(DEMO_POOL_PATH):
        try:
            print("📦  Loading compact demo pool for simulation...")
            demo = np.load(DEMO_POOL_PATH)
            _pools[0] = demo["normal"].astype(np.float32)
            _pools[1] = demo["ddos"].astype(np.float32)
            _pools[2] = demo["botnet"].astype(np.float32)
        except Exception as e:
            print(f"⚠️  Failed to load demo pool: {e}")
            _use_synthetic = True
            _build_synthetic_pool()
            return
    else:
        # Try full test arrays as fallback
        x_path = os.path.join(DATA_DIR, "X_test.npy")
        y_path = os.path.join(DATA_DIR, "y_test.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            try:
                print("📦  Loading full test pool for simulation...")
                X = np.load(x_path)   # (N, 10, 38)
                y = np.load(y_path)   # (N,)
                X_flat = X[:, -1, :]  # (N, 38)
                for cls in [0, 1, 2]:
                    mask = y == cls
                    _pools[cls] = X_flat[mask]
                del X, y, X_flat  # free memory immediately
            except Exception as e:
                print(f"⚠️  Failed to load test data: {e}")
                _use_synthetic = True
                _build_synthetic_pool()
                return
        else:
            print("⚠️  No data files found — using synthetic simulation data.")
            _use_synthetic = True
            _build_synthetic_pool()
            return

    label_map = {0: "Normal", 1: "DDoS", 2: "Botnet"}
    for cls, name in label_map.items():
        print(f"   {name:8s}: {len(_pools[cls]):,} real samples available")


def _build_synthetic_pool():
    """Generate synthetic feature vectors for each class when no real data is available."""
    global _pools
    rng = np.random.default_rng(42)
    n = 500

    # Normal traffic: low variance, moderate values
    normal = rng.uniform(0.0, 0.3, size=(n, N_FEATS)).astype(np.float32)
    normal[:, 0] = rng.uniform(0.0, 0.15, n)   # Variance
    normal[:, 3] = rng.uniform(0.5, 0.9, n)     # TCP
    normal[:, 5] = rng.uniform(0.01, 0.1, n)    # Duration

    # DDoS traffic: high rate, high syn count, high packet sizes
    ddos = rng.uniform(0.3, 0.9, size=(n, N_FEATS)).astype(np.float32)
    ddos[:, 6] = rng.uniform(0.7, 1.0, n)   # syn_count high
    ddos[:, 24] = rng.uniform(0.8, 1.0, n)  # Rate high
    ddos[:, 25] = rng.uniform(0.7, 1.0, n)  # Srate high
    ddos[:, 16] = rng.uniform(0.6, 1.0, n)  # Header_Length high

    # Botnet traffic: mixed patterns, C2 indicators
    botnet = rng.uniform(0.2, 0.7, size=(n, N_FEATS)).astype(np.float32)
    botnet[:, 14] = rng.uniform(0.6, 1.0, n)  # UDP high
    botnet[:, 11] = rng.uniform(0.3, 0.8, n)  # urg_count
    botnet[:, 31] = rng.uniform(0.5, 0.9, n)  # Number
    botnet[:, 34] = rng.uniform(0.4, 0.8, n)  # ICMP

    _pools[0] = np.clip(normal, 0.0, 1.0)
    _pools[1] = np.clip(ddos, 0.0, 1.0)
    _pools[2] = np.clip(botnet, 0.0, 1.0)

    print("   Synthetic pool built: 500 samples per class")


def _sample(cls: int, n: int = 1, noise: float = 0.01) -> np.ndarray:
    """Sample n real feature vectors from the given class pool."""
    _load_pool()
    pool = _pools[cls]
    idx  = np.random.randint(0, len(pool), size=n)
    samples = pool[idx].copy()
    samples += np.random.normal(0, noise, samples.shape)
    return np.clip(samples, 0.0, 1.0).astype(np.float32)


# ── Public API (same interface as before) ─────────────────────────────────────

def generate_normal(n_samples: int = 1, **kwargs) -> np.ndarray:
    """Returns n real Normal traffic feature vectors."""
    return _sample(cls=0, n=n_samples)


def generate_ddos(n_samples: int = 1, attack_type: str = "syn_flood", **kwargs) -> np.ndarray:
    """Returns n real DDoS traffic feature vectors."""
    return _sample(cls=1, n=n_samples)


def generate_botnet(n_samples: int = 1, variant: str = "udpplain", **kwargs) -> np.ndarray:
    """Returns n real Botnet (Mirai) traffic feature vectors."""
    return _sample(cls=2, n=n_samples)


def get_sample_dataframe(n_per_class: int = 5) -> "pd.DataFrame":
    """Generate a sample DataFrame showing expected CSV format."""
    import pandas as pd
    _load_pool()
    rows = []
    for cls, label in [(0, "Normal"), (1, "DDoS"), (2, "Botnet")]:
        samples = _sample(cls, n_per_class, noise=0.005)
        for row in samples:
            d = {feat: round(float(row[i]), 4) for i, feat in enumerate(FEATURES)}
            d["label"] = label  # optional label column
            rows.append(d)
    return pd.DataFrame(rows)