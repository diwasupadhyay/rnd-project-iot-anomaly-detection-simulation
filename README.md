# 🛡️ IoT Anomaly Shield — Real-Time Network Threat Detection

**CNN-LSTM Deep Learning + LLM (Gemini 2.5 Flash / Ollama)**

Detects DDoS attacks and Mirai botnet infections in IoT network traffic in real-time using a hybrid CNN-LSTM model, with live AI-powered incident analysis via Gemini API (deployment) or Ollama (local).

---

## Overview

| Component | Details |
|---|---|
| **Dataset** | CICIoT2023 — 169 CSV files (~3 GB, 507K samples) |
| **Model** | CNN-LSTM hybrid (218K params, 99.92% accuracy) |
| **LLM** | Gemini 2.5 Flash (deploy) or Ollama LLaMA 3.2:3b (local) |
| **Dashboard** | Streamlit + Plotly (dark theme, real-time) |
| **Detection** | 3 classes: Normal, DDoS (SYN/UDP), Botnet (Mirai) |
| **Language** | Python 3.11+ / PyTorch 2.x (CUDA supported) |

---

## Features

- **Real-time classification** — Sliding-window CNN-LSTM classifies flows every ~0.3s
- **Live streaming LLM analysis** — Token-chunk incident reports via Gemini or Ollama
- **CSV Upload testing mode** — Run inference on your own traffic CSV files
- **Network health monitoring** — Dynamic health score with threat severity
- **Professional SOC dashboard** — Dark theme, live charts, probability visualization
- **Attack simulation** — Multi-phase scenarios using real CICIoT2023 data vectors
- **Console pipeline** — CLI-based real-time detection with colored alerts

---

## Prerequisites

- Python 3.11 or 3.12
- NVIDIA GPU with CUDA (recommended, CPU also works)
- Gemini API key (recommended for deployment) or [Ollama](https://ollama.com/download) for local mode
- ~6 GB free disk space

---

## Quick Start

```bash
# 1. Clone & enter project
cd iot-anomaly-detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Windows PowerShell:
copy .env.example .env
# then set GEMINI_API_KEY in .env (or in your deployment platform env settings)

# 5. Optional local LLM mode (instead of Gemini)
ollama pull llama3.2:3b
ollama serve
```

---

## Dataset

1. Download from [Kaggle — CICIoT2023](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset)
2. Unzip all 169 CSV files into `data/raw/`

---

## Pipeline

### Step 1 — Data Preprocessing
```bash
python data/eda.py            # Exploratory analysis → data/eda_plots/
python data/preprocess.py     # Feature engineering + train/val/test split → data/processed/
python data/verify.py         # Validate shapes and class distribution
```

### Step 2 — Model Training & Evaluation
```bash
python models/train.py        # CNN-LSTM training (early stopping, LR scheduling)
python models/evaluate.py     # Test metrics, confusion matrix, ROC curves
```

### Step 3 — Real-Time Detection (CLI)
```bash
python detection/realtime_pipeline.py    # Console-based live detection demo
```

### Step 4 — Dashboard
```bash
streamlit run dashboard/app.py
# Open http://localhost:8501 → press ▶ Start in sidebar
# For custom data: switch Input Source to CSV Upload and upload your CSV
```

---

## Project Structure

```
iot-anomaly-detection/
├── dashboard/
│   └── app.py                ← Streamlit dashboard (main showcase)
├── data/
│   ├── raw/                  ← Place Kaggle CSVs here (169 files)
│   ├── processed/            ← Auto-generated .npy files + scaler
│   ├── eda.py                ← Exploratory data analysis
│   ├── preprocess.py         ← Full preprocessing pipeline
│   └── verify.py             ← Data validation
├── models/
│   ├── cnn_lstm.py           ← CNN-LSTM architecture definition
│   ├── train.py              ← Training loop (GPU, early stopping)
│   ├── evaluate.py           ← Test metrics + plot generation
│   ├── saved/
│   │   └── best_model.pth    ← Trained model weights
│   └── plots/                ← Training curves, confusion matrix, ROC
├── simulation/
│   ├── traffic_simulator.py  ← Real data-based traffic sampler
│   ├── ddos_attack.py        ← DDoS attack simulation script
│   ├── botnet_attack.py      ← Botnet infection simulation script
│   └── normal_traffic.py     ← Normal traffic generation
├── detection/
│   ├── detector.py           ← File-based inference on CSV logs
│   └── realtime_pipeline.py  ← Console real-time detection pipeline
├── llm/
│   └── interpreter.py        ← Gemini/Ollama LLM wrapper
├── requirements.txt
├── METHODOLOGY.md
└── README.md
```

---

## Model Architecture

```
Input: (batch, 10 timesteps, 38 features)
    ↓
Conv1D(38→64)  → BatchNorm → ReLU
Conv1D(64→128) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
LSTM(128→128)  → LSTM(128→64) → Dropout(0.3)
    ↓
Dense(64→64)   → ReLU → Dropout(0.3) → Dense(64→3)
    ↓
Output: [Normal, DDoS, Botnet] probabilities
```

**Trainable Parameters:** 218,563

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

| Class | Attacks Covered |
|---|---|
| **Normal** | BenignTraffic |
| **DDoS** | SYN Flood, UDP Flood, ICMP Flood, HTTP Flood, SlowLoris, TCP Flood, and 6 more |
| **Botnet** | Mirai greeth_flood, greip_flood, udpplain |

---

## Dashboard Features

- **Status header** with animated threat level indicator
- **Probability strip** showing real-time model output distribution
- **Network health** score that degrades during attacks
- **Live confidence chart** with attack region highlighting
- **Terminal-style flow log** with color-coded entries
- **Active threat banner** with pulsing animation
- **AI analysis cards** with live LLM streaming (token-by-token)
- **Attack event log** table with timestamps and confidence
- **CSV Upload** mode with downloadable prediction outputs

---

## Tech Details

- **Traffic Simulation**: Samples real feature vectors from the CICIoT2023 test set (not synthetic), adds minimal Gaussian noise
- **Sliding Window**: 10 consecutive flows form one classification input
- **LLM Integration**: Gemini API (deployment) or Ollama NDJSON stream (local)
- **Preprocessing**: MinMaxScaler, top-50 features by correlation → 38 after cleaning, stratified 70/15/15 split

---

## Deployment

See full deployment instructions in DEPLOYMENT.md.
For lightweight hosting, simulation uses data/processed/demo_pool.npz (compact sample pool) instead of full test arrays.
