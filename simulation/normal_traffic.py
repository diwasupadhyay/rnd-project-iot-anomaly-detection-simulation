"""
simulation/normal_traffic.py
==============================
Simulates a stream of normal IoT network flows.
Prints feature vectors to console and saves to a CSV for inspection.

Run: python simulation/normal_traffic.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_simulator import generate_normal, FEATURES
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "simulation", "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERVAL    = 0.2    # seconds between flows
N_FLOWS     = 50     # total flows to generate


def run():
    print("="*55)
    print("🟢  Normal IoT Traffic Simulation")
    print(f"    Generating {N_FLOWS} flows  |  interval={INTERVAL}s")
    print("="*55)

    records = []
    for i in range(1, N_FLOWS + 1):
        flow = generate_normal(n_samples=1)[0]
        record = dict(zip(FEATURES, flow))
        record["label"] = "Normal"
        record["timestamp"] = time.time()
        records.append(record)

        # Print a brief summary each flow
        rate_col = next((f for f in FEATURES if "rate" in f.lower()), FEATURES[0])
        iat_col  = next((f for f in FEATURES if "iat"  in f.lower()), FEATURES[1])
        print(f"  Flow {i:>3} | {rate_col[:20]}: {flow[FEATURES.index(rate_col)]:.3f} | "
              f"{iat_col[:15]}: {flow[FEATURES.index(iat_col)]:.3f} | Label: Normal ✅")
        time.sleep(INTERVAL)

    # Save log
    df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, "normal_traffic.csv")
    df.to_csv(out_path, index=False)
    print(f"\n💾  Saved {N_FLOWS} flows → {out_path}")
    print("✅  Normal traffic simulation complete")


if __name__ == "__main__":
    run()
