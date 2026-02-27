# Real-Time Anomaly Detection in IoT Networks Using Deep Learning and LLM

**R&D Project — Methodology & Implementation Plan**

---

## 1. Problem Statement

IoT networks are highly vulnerable to cyber attacks such as DDoS and Botnet infections due to limited device security. Traditional rule-based intrusion detection systems fail to generalize against evolving attack patterns. This project builds a real-time anomaly detection system using Deep Learning (CNN-LSTM) for detection and a Large Language Model (LLM) for intelligent, human-readable threat interpretation.

---

## 2. Objectives

- Detect DDoS and Botnet attacks in IoT network traffic in real time
- Achieve high accuracy (>95%) using a CNN-LSTM deep learning model
- Integrate a local LLM to explain detected threats and recommend mitigations
- Build a live monitoring dashboard suitable for practical demonstration

---

## 3. Dataset

**CICIoT2023** — Canadian Institute for Cybersecurity IoT Dataset (Neto et al., 2023)

- **Source:** University of New Brunswick (available on Kaggle)
- **Size:** ~3GB, 169 CSV files, ~507,000 flow samples used
- **Features:** 78 pre-extracted network flow features per record
- **Label column:** `label` (string class name)

| Class | Attack Types | Samples Used |
|---|---|---|
| Normal | BenignTraffic | ~169,000 |
| DDoS | SYN Flood, UDP Flood, ICMP Flood, HTTP Flood, SlowLoris + 7 more | ~169,000 |
| Botnet | Mirai greeth_flood, greip_flood, udpplain | ~169,000 |

---

## 4. Methodology

### 4.1 Data Preprocessing Pipeline

1. **Loading** — Stratified sampling across all 169 CSV files (3,000 rows/file)
2. **Label Mapping** — Raw label strings mapped to 3 numeric classes (Normal=0, DDoS=1, Botnet=2)
3. **Cleaning** — Remove null values, infinity values, duplicate rows, near-zero variance columns
4. **Feature Selection** — Top 50 features selected by absolute Pearson correlation with label (reduced to 38 after cleaning)
5. **Splitting** — 70% train / 15% validation / 15% test (stratified)
6. **Scaling** — MinMaxScaler fitted on training set only (prevents data leakage)
7. **Sequence Creation** — Sliding window of size 10 converts flat rows into temporal sequences for LSTM input → shape: `(samples, 10, 38)`

Final dataset: 354,891 training samples, 76,041 validation, 76,041 test. Perfectly balanced (33.3% each class).

---

### 4.2 Deep Learning Model — CNN-LSTM Hybrid

**Rationale:** CNN extracts local spatial patterns within a flow window (e.g., repeated SYN flags); LSTM captures temporal dependencies across sequential flows (attack build-up over time). Their combination outperforms standalone models on network traffic classification.

**Architecture:**

```
Input  →  (batch, 10 timesteps, 38 features)
           │
    ┌──────▼──────┐
    │   Conv1D     │  64 filters, kernel=3, BatchNorm, ReLU
    │   Conv1D     │  128 filters, kernel=3, BatchNorm, ReLU
    │   MaxPool1D  │
    │   Dropout    │  0.3
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   LSTM       │  128 units
    │   LSTM       │  64 units
    │   Dropout    │  0.3
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Dense      │  64 units, ReLU
    │   Dense      │  3 units, Softmax
    └─────────────┘
Output →  [P(Normal), P(DDoS), P(Botnet)]
```

**Training Configuration:**
- Optimizer: Adam (lr=1e-3, weight decay=1e-4)
- Loss: Weighted CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early Stopping: patience=5
- Batch size: 512
- Hardware: NVIDIA RTX 3050 Laptop GPU (4GB VRAM)
- Training time: ~4 minutes

---

### 4.3 Attack Simulation

Since the system must demonstrate real-time capability, a simulation pipeline is built using real flow samples from the held-out test set:

- **Normal phase** — samples from test set class 0
- **DDoS phase** — samples from test set class 1
- **Botnet phase** — samples from test set class 2

Small Gaussian noise (σ=0.01) is added to ensure each flow is unique. The simulation runs the scenario: Normal → DDoS → Recovery → Botnet → Normal.

---

### 4.4 Real-Time Detection Pipeline

1. Incoming flows are buffered into a sliding window of 10 samples
2. Each complete window is passed to the trained CNN-LSTM model
3. Model outputs softmax probabilities for all 3 classes
4. If confidence > 85% for an attack class, an alert is triggered
5. Results are streamed to the dashboard in real time

---

### 4.5 LLM Integration — Ollama (LLaMA 3.2:3b)

When an attack is detected, the model's output (attack type, confidence, probabilities) is formatted into a structured prompt and sent to a locally-running LLaMA 3.2:3b model via the Ollama API (http://localhost:11434).

**Prompt includes:**
- Attack type and confidence score
- Short technical description of the attack
- Key network indicators

**LLM outputs:**
- Plain-language explanation of what is happening
- Immediate mitigation actions
- Long-term prevention recommendations
- Severity rating

This layer transforms raw model predictions into actionable security intelligence without any cloud dependency.

---

### 4.6 Dashboard

Built with Streamlit and Plotly. Displays:
- Live confidence chart with attack region highlighting
- Real-time flow log (terminal style)
- Active attack alert banner
- LLM analysis cards (persistent, one per attack event)
- Attack event log table

---

## 5. Implementation Plan

| Phase | Task | Status |
|---|---|---|
| Phase 1 | Dataset download, EDA, preprocessing pipeline | ✅ Complete |
| Phase 2 | CNN-LSTM model design, training, evaluation | ✅ Complete |
| Phase 3 | Attack simulation scripts (DDoS, Botnet, Normal) | ✅ Complete |
| Phase 4 | Real-time detection pipeline | ✅ Complete |
| Phase 5 | LLM integration via Ollama | ✅ Complete |
| Phase 6 | Streamlit dashboard | ✅ Complete |

---

## 6. Results

| Metric | Value |
|---|---|
| Overall Test Accuracy | **99.92%** |
| Normal (Precision / Recall) | 99.98% / 100.00% |
| DDoS (Precision / Recall) | 99.94% / 99.81% |
| Botnet (Precision / Recall) | 99.82% / 99.94% |
| Macro F1-Score | **0.9992** |
| Trainable Parameters | 218,563 |

---

## 7. Tools & Libraries

| Purpose | Library |
|---|---|
| Data processing | pandas, numpy, scikit-learn |
| Deep learning | PyTorch (CUDA) |
| Visualization | matplotlib, seaborn, plotly |
| LLM inference | Ollama (local), requests |
| Dashboard | Streamlit |

---

## 8. References

- Neto, E.C.P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A.A. (2023). *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environments.*
- Kim, J. et al. (2020). *Network Intrusion Detection based on CNN and LSTM.* IEEE Access.
- Mirai Botnet Analysis — CISA ICS-CERT Advisory (2016).
