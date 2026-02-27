"""
simulation/ddos_attack.py
==========================
Simulates a DDoS attack scenario:
  Phase 1 — Normal traffic (baseline)
  Phase 2 — DDoS ramp-up (attack begins)
  Phase 3 — Full DDoS flood
  Phase 4 — Attack subsides

Run: python simulation/ddos_attack.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_simulator import generate_normal, generate_ddos, FEATURES
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "simulation", "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERVAL = 0.15

# Attack scenario phases: (label, generator_fn, n_flows, description)
SCENARIO = [
    ("Normal",  "normal",    15, "🟢  Phase 1: Normal baseline traffic"),
    ("DDoS",    "syn_flood",  5, "🟡  Phase 2: DDoS ramp-up (SYN flood begins)"),
    ("DDoS",    "syn_flood", 20, "🔴  Phase 3: Full DDoS SYN flood"),
    ("DDoS",    "udp_flood", 10, "🔴  Phase 3b: Switching to UDP flood"),
    ("Normal",  "normal",    10, "🟢  Phase 4: Attack subsides — traffic normalizes"),
]


def run():
    print("="*60)
    print("💥  DDoS Attack Simulation — CICIoT2023 Style")
    print("    Attack types: SYN Flood → UDP Flood")
    print("="*60)

    all_records = []
    flow_num = 0

    for true_label, gen_type, n_flows, phase_desc in SCENARIO:
        print(f"\n{phase_desc}  ({n_flows} flows)")
        print("─" * 55)

        for _ in range(n_flows):
            flow_num += 1

            if gen_type == "normal":
                features = generate_normal(n_samples=1)[0]
            else:
                features = generate_ddos(n_samples=1, attack_type=gen_type)[0]

            record = dict(zip(FEATURES, features))
            record["label"]     = true_label
            record["timestamp"] = time.time()
            all_records.append(record)

            rate_col = next((f for f in FEATURES if "rate" in f.lower()), FEATURES[0])
            syn_col  = next((f for f in FEATURES if "syn"  in f.lower()), FEATURES[2])
            icon = "🔴" if true_label == "DDoS" else "🟢"
            print(f"  {icon} Flow {flow_num:>3} | {rate_col[:18]}: {features[FEATURES.index(rate_col)]:.3f} | "
                  f"{syn_col[:12]}: {features[FEATURES.index(syn_col)]:.3f} | {true_label}")
            time.sleep(INTERVAL)

    # Save
    df = pd.DataFrame(all_records)
    out_path = os.path.join(OUTPUT_DIR, "ddos_simulation.csv")
    df.to_csv(out_path, index=False)

    n_ddos   = (df["label"] == "DDoS").sum()
    n_normal = (df["label"] == "Normal").sum()
    print(f"\n{'='*60}")
    print(f"✅  Simulation complete: {len(df)} flows total")
    print(f"    Normal: {n_normal}  |  DDoS: {n_ddos}")
    print(f"💾  Saved → {out_path}")
    print("\n📌  Run detection on this: python detection/detector.py --file simulation/logs/ddos_simulation.csv")


if __name__ == "__main__":
    run()
