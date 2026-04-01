# Deployment Guide

This project is deployment-ready for Streamlit-compatible hosts (Render, Railway, VM, container).

## 1) Required Artifacts

Keep these in the repository for showcase mode:
- models/saved/best_model.pth
- data/processed/feature_names.txt
- data/processed/scaler.pkl
- data/processed/demo_pool.npz

These allow live simulation and CSV testing without retraining while keeping repo size deployment-friendly.

## 2) Environment Variables

Set these in your deployment platform:

- LLM_PROVIDER=gemini
- GEMINI_API_KEY=your_real_key
- GEMINI_MODEL=gemini-2.5-flash

Optional local fallback:
- LLM_PROVIDER=ollama
- OLLAMA_URL=http://localhost:11434/api/generate
- OLLAMA_MODEL=llama3.2:3b

## 3) Build and Start Commands

Build command:
- pip install -r requirements.txt

Start command:
- streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0

Procfile is already included with the same command.

## 4) Recommended Host Setup (Render Example)

1. Push repository to GitHub.
2. Create a new Web Service on Render from the repo.
3. Runtime: Python.
4. Build command: pip install -r requirements.txt
5. Start command: streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
6. Add environment variables listed in section 2.
7. Deploy.

## 5) Cost and API Usage Control

The dashboard already limits LLM calls:
- LLM only triggers for attack predictions above confidence threshold.
- LLM has cooldown between calls.
- LLM does not run for every single detection event.

Tune from dashboard sidebar:
- LLM min confidence %
- LLM cooldown (sec)

## 6) Smoke Test Checklist

After deployment:
1. Open dashboard URL.
2. Test Live Simulation mode.
3. Test CSV Upload mode with your own CSV.
4. Confirm predictions appear.
5. Trigger at least one high-confidence attack and confirm Gemini response.
