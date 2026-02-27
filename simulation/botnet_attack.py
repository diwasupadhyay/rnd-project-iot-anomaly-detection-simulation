"""
simulation/botnet_attack.py
============================
Simulates a Mirai Botnet infection scenario:
  Phase 1 — Normal traffic
  Phase 2 — Initial scanning (low-rate botnet probing)
  Phase 3 — Botnet flood (full Mirai behaviour)
  Phase 4 — Mixed: some bots still active

Run: python simulation/botnet_attack.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_simulator import generate_normal, generate_botnet, FEATURES
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "simulation", "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERVAL = 0.15

SCENARIO = [
    ("Normal", "normal",   15, "🟢  Phase 1: Normal IoT baseline"),
    ("Botnet", "udpplain", 10, "🟡  Phase 2: Mirai scanning — device probing"),
    ("Botnet", "greeth",   20, "🔴  Phase 3: Mirai GRE-Ethernet flood"),
    ("Botnet", "greip",    10, "🔴  Phase 3b: Mirai GRE-IP flood"),
    ("Normal", "normal",    5, "🟢  Phase 4: Some bots idle — partial recovery"),
    ("Botnet", "udpplain",  8, "🔴  Phase 4b: Botnet still active in background"),
]


def run():
    print("="*60)
    print("🤖  Botnet (Mirai) Attack Simulation — CICIoT2023 Style")
    print("    Variants: UDP Plain → GRE-Ethernet → GRE-IP")
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
                features = generate_botnet(n_samples=1, variant=gen_type)[0]

            record = dict(zip(FEATURES, features))
            record["label"]     = true_label
            record["timestamp"] = time.time()
            all_records.append(record)

            rate_col = next((f for f in FEATURES if "rate"    in f.lower()), FEATURES[0])
            rst_col  = next((f for f in FEATURES if "rst"     in f.lower()), FEATURES[3])
            icon = "🔴" if true_label == "Botnet" else "🟢"
            print(f"  {icon} Flow {flow_num:>3} | {rate_col[:18]}: {features[FEATURES.index(rate_col)]:.3f} | "
                  f"{rst_col[:12]}: {features[FEATURES.index(rst_col)]:.3f} | {true_label}")
            time.sleep(INTERVAL)

    df = pd.DataFrame(all_records)
    out_path = os.path.join(OUTPUT_DIR, "botnet_simulation.csv")
    df.to_csv(out_path, index=False)

    n_botnet = (df["label"] == "Botnet").sum()
    n_normal = (df["label"] == "Normal").sum()
    print(f"\n{'='*60}")
    print(f"✅  Simulation complete: {len(df)} flows total")
    print(f"    Normal: {n_normal}  |  Botnet: {n_botnet}")
    print(f"💾  Saved → {out_path}")
    print("\n📌  Run detection: python detection/detector.py --file simulation/logs/botnet_simulation.csv")


if __name__ == "__main__":
    run()
