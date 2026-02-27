# 🛡️ Real-Time Anomaly Detection in IoT Networks
### Using CNN-LSTM Deep Learning + Local LLM (Ollama)

---

## Project Status
- [x] **Step 1** — Data Preprocessing ← YOU ARE HERE
- [ ] **Step 2** — CNN-LSTM Model Training
- [ ] **Step 3** — Attack Simulation (DDoS / Botnet)
- [ ] **Step 4** — Real-Time Detection Pipeline
- [ ] **Step 5** — Ollama LLM Integration
- [ ] **Step 6** — Streamlit Dashboard

---

## Setup (One-Time)

### 1. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify GPU (CUDA)
```python
import torch
print(torch.cuda.is_available())   # Should print: True
print(torch.cuda.get_device_name()) # Your GPU name
```

---

## Step 1: Data Preprocessing

### Download Dataset
```bash
# Option A: Kaggle CLI (recommended)
python data/download_dataset.py

# Option B: Manual
# Visit: https://www.kaggle.com/datasets/madhavmalhotraofficial/cic-iot-dataset-2023
# Download → unzip into data/raw/
```

### Run EDA
```bash
python data/eda.py
# Outputs plots to: data/eda_plots/
```

### Run Preprocessing
```bash
python data/preprocess.py
# Outputs to: data/processed/
#   X_train.npy, X_val.npy, X_test.npy   (samples, window=10, features=50)
#   y_train.npy, y_val.npy, y_test.npy   (0=Normal, 1=DDoS, 2=Botnet)
#   scaler.pkl                            (for inference)
#   feature_names.txt                     (50 selected features)
#   class_weights.npy                     (for balanced training)
```

### Verify Output
```bash
python data/verify.py
```

---

## Project Structure
```
iot-anomaly-detection/
├── data/
│   ├── raw/                    ← Kaggle CSVs go here
│   ├── processed/              ← Preprocessed .npy files (auto-created)
│   ├── eda_plots/              ← EDA visualizations (auto-created)
│   ├── download_dataset.py     ← Kaggle download helper
│   ├── eda.py                  ← Step 1a: Exploratory analysis
│   ├── preprocess.py           ← Step 1b: Full preprocessing pipeline
│   └── verify.py               ← Step 1c: Validate output
│
├── models/                     ← (Coming in Step 2)
├── simulation/                 ← (Coming in Step 3)
├── detection/                  ← (Coming in Step 4)
├── llm/                        ← (Coming in Step 5)
├── dashboard/                  ← (Coming in Step 6)
└── requirements.txt
```

---

## Dataset: CIC-IoT2023

| Property | Value |
|---|---|
| Source | Canadian Institute for Cybersecurity |
| Classes | Normal, DDoS, Botnet (+ others mapped to Normal) |
| Features | 78 network flow features |
| Format | Pre-extracted CSV (no PCAP needed) |
| Size | ~1GB |

**Key Features Used:**
- Flow Duration, Packet Length (mean/max/min/std)
- Inter-Arrival Times (IAT)
- TCP Flag Counts (SYN, FIN, RST, PSH)
- Forward/Backward Packet rates
- Active/Idle time statistics

---

## Architecture Overview

```
Input (batch=32, window=10, features=50)
        │
   ┌────▼────┐
   │  Conv1D  │  kernel=3, filters=64  — spatial feature patterns
   │  Conv1D  │  kernel=3, filters=128
   │ MaxPool  │
   │ Dropout  │  0.3
   └────┬────┘
        │
   ┌────▼────┐
   │  LSTM   │  128 units — temporal attack sequences
   │  LSTM   │  64 units
   │ Dropout │  0.3
   └────┬────┘
        │
   ┌────▼────┐
   │  Dense  │  64 → 3
   │ Softmax │  Normal / DDoS / Botnet
   └─────────┘
```
