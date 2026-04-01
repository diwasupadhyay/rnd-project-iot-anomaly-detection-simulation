"""
traffic_simulator.py — Real Data-Based Traffic Sampler
========================================================
Instead of synthetic feature generation, this samples REAL
feature vectors from the processed CICIoT2023 dataset.

This guarantees the model sees the same feature distributions
it was trained on → correct DDoS/Botnet/Normal detection.

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
    with open(FEAT_PATH) as f:
        return f.read().splitlines()

FEATURES = get_feature_names()
N_FEATS  = len(FEATURES)

# ── Load real test samples grouped by class ───────────────────────────────────
# We use X_test / y_test (never touched during training)
_X_test = None
_y_test = None
_pools  = {}   # {0: array of Normal rows, 1: DDoS rows, 2: Botnet rows}

def _load_pool():
    global _X_test, _y_test, _pools
    if _pools:
        return   # already loaded

    if os.path.exists(DEMO_POOL_PATH):
        print("📦  Loading compact demo pool for simulation...")
        demo = np.load(DEMO_POOL_PATH)
        _pools[0] = demo["normal"].astype(np.float32)
        _pools[1] = demo["ddos"].astype(np.float32)
        _pools[2] = demo["botnet"].astype(np.float32)
    else:
        print("📦  Loading full test pool for simulation...")
        X = np.load(os.path.join(DATA_DIR, "X_test.npy"))   # (N, 10, 38)
        y = np.load(os.path.join(DATA_DIR, "y_test.npy"))   # (N,)

        # Use the LAST timestep of each window as a single feature vector
        X_flat = X[:, -1, :]   # (N, 38)

        for cls in [0, 1, 2]:
            mask = y == cls
            _pools[cls] = X_flat[mask]

    label_map = {0: "Normal", 1: "DDoS", 2: "Botnet"}
    for cls, name in label_map.items():
        print(f"   {name:8s}: {len(_pools[cls]):,} real samples available")


def _sample(cls: int, n: int = 1, noise: float = 0.01) -> np.ndarray:
    """Sample n real feature vectors from the given class pool."""
    _load_pool()
    pool = _pools[cls]
    idx  = np.random.randint(0, len(pool), size=n)
    samples = pool[idx].copy()
    # Add tiny noise so repeated calls aren't identical
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