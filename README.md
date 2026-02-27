# 🛡️ Real-Time Anomaly Detection in IoT Networks
### CNN-LSTM Deep Learning + Local LLM (Ollama)

---

## Tech Stack
| Component | Tool |
|---|---|
| Dataset | CICIoT2023 — 169 CSV files, 3GB |
| DL Model | CNN-LSTM (PyTorch) |
| LLM | Ollama — LLaMA 3.2:3b (local, offline) |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.11 / 3.12 |

---

## Prerequisites
- Python 3.11 or 3.12
- NVIDIA GPU with CUDA (recommended)
- [Ollama](https://ollama.com/download) installed
- ~6GB free disk space

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull LLM (one time, ~2GB)
ollama pull llama3.2:3b
```

---

## Dataset

1. Download: https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset
2. Unzip all 169 CSV files into `data/raw/`

---

## Run Order

### Step 1 — Preprocessing
```bash
python data/eda.py            # EDA plots saved to data/eda_plots/
python data/preprocess.py     # Creates train/val/test .npy arrays
python data/verify.py         # Confirm shapes and class balance
```

### Step 2 — Train Model
```bash
python models/train.py        # ~4 min on RTX 3050, saves best_model.pth
python models/evaluate.py     # Prints accuracy + saves confusion matrix/ROC plots
```

### Step 3 — Console Demo (optional)
```bash
python detection/realtime_pipeline.py
```

### Step 4 — Full Dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501 → press ▶ Start in sidebar
```

---

## Project Structure
```
iot-anomaly-detection/
├── data/
│   ├── raw/                  ← Place Kaggle CSVs here
│   ├── processed/            ← Auto-generated .npy files
│   ├── eda.py
│   ├── preprocess.py
│   └── verify.py
├── models/
│   ├── cnn_lstm.py           ← Model architecture
│   ├── train.py              ← Training loop (GPU, early stopping)
│   ├── evaluate.py           ← Metrics + plots
│   └── saved/best_model.pth  ← Trained weights
├── simulation/
│   ├── traffic_simulator.py  ← Real data sampler
│   ├── ddos_attack.py
│   └── botnet_attack.py
├── detection/
│   ├── detector.py           ← File-based inference
│   └── realtime_pipeline.py  ← Console real-time demo
├── llm/
│   └── interpreter.py        ← Ollama LLaMA wrapper
├── dashboard/
│   └── app.py                ← Streamlit dashboard
└── requirements.txt
```

---

## Results
| Metric | Score |
|---|---|
| Overall Accuracy | **99.92%** |
| Normal Detection | **100.00%** |
| DDoS Detection | **99.81%** |
| Botnet Detection | **99.94%** |
| F1-Score (macro) | **0.9992** |
| Training Time | ~4 min (RTX 3050 4GB) |

---

## Detected Attack Classes
- **Normal** — BenignTraffic
- **DDoS** — SYN Flood, UDP Flood, ICMP Flood, HTTP Flood, SlowLoris + 6 more variants
- **Botnet** — Mirai greeth_flood, greip_flood, udpplain
