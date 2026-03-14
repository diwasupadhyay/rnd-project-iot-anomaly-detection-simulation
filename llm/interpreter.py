"""
llm/interpreter.py — Ollama LLM Interpreter
=============================================
Streams real-time attack analysis from a local LLaMA 3.2:3b model.
Supports token-by-token streaming via Ollama's NDJSON API.
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

_CONTEXT = {
    "DDoS": {
        "desc":       "Distributed Denial-of-Service attack flooding IoT gateway with high-rate SYN/UDP packets",
        "indicators": "Abnormal packet rate, SYN flag surges, bandwidth saturation, connection timeouts",
        "impact":     "Gateway overload, device unresponsiveness, service disruption",
        "severity":   "CRITICAL",
    },
    "Botnet": {
        "desc":       "Mirai-variant botnet infection — compromised IoT devices executing C2-directed attacks",
        "indicators": "Outbound UDP/GRE floods, C2 beacon patterns, port-scanning from internal IPs",
        "impact":     "Devices weaponized for DDoS, lateral spread risk, potential data exfiltration",
        "severity":   "CRITICAL",
    },
}


def _build_prompt(attack_type: str, confidence: float,
                  prob_normal: float, prob_ddos: float, prob_botnet: float) -> str:
    ctx = _CONTEXT.get(attack_type, _CONTEXT["DDoS"])
    return f"""You are a senior IoT network security analyst at a SOC. A CNN-LSTM anomaly detection model raised a real-time alert.

DETECTION REPORT:
  Attack    : {attack_type} ({ctx['desc']})
  Confidence: {confidence:.1f}%
  Model out : Normal={prob_normal*100:.1f}% | DDoS={prob_ddos*100:.1f}% | Botnet={prob_botnet*100:.1f}%
  Indicators: {ctx['indicators']}
  Impact    : {ctx['impact']}

Write a concise incident response. Use EXACTLY this format:

THREAT ASSESSMENT:
[2 sentences: what is happening and how severe it is]

IMMEDIATE ACTIONS:
1. [First containment step]
2. [Second mitigation step]
3. [Third protective measure]

RECOMMENDATIONS:
- [Infrastructure hardening]
- [Monitoring improvement]

SEVERITY: {ctx['severity']}"""


_OPTS = {"temperature": 0.2, "num_predict": 280, "top_p": 0.85, "repeat_penalty": 1.15}


def stream_attack_analysis(attack_type: str, confidence: float,
                           prob_normal: float, prob_ddos: float,
                           prob_botnet: float):
    """Generator — yields text tokens one at a time as Ollama streams them."""
    prompt = _build_prompt(attack_type, confidence, prob_normal, prob_ddos, prob_botnet)
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": True, "options": _OPTS},
            stream=True, timeout=90,
        ) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield (
            "\n⚠️  Cannot reach Ollama.\n\n"
            "Start it:   ollama serve\n"
            "Pull model: ollama pull llama3.2:3b\n\n"
            "LLM will auto-retry on the next detected attack."
        )
    except requests.exceptions.Timeout:
        yield "\n⚠️  Ollama timed out (90s). The model may be loading — next attack will retry."
    except Exception as e:
        yield f"\n⚠️  LLM error: {type(e).__name__}: {e}"


def get_attack_analysis(attack_type: str, confidence: float,
                        prob_normal: float, prob_ddos: float,
                        prob_botnet: float) -> dict:
    """Non-streaming fallback — joins all tokens. Used by CLI tools."""
    response = "".join(
        stream_attack_analysis(attack_type, confidence, prob_normal, prob_ddos, prob_botnet)
    )
    return {"attack_type": attack_type, "confidence": confidence, "response": response.strip()}