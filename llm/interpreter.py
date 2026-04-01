"""
llm/interpreter.py
===================
Provider-agnostic LLM interpreter for attack explanation.

Supports:
    - Ollama (local) for offline development
    - Gemini API for deployment environments

Environment variables:
    LLM_PROVIDER=gemini|ollama   (default: gemini)
    OLLAMA_URL=http://localhost:11434/api/generate
    OLLAMA_MODEL=llama3.2:3b
    GEMINI_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash
"""

import json
import os
from typing import Generator

import requests

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

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


def _build_prompt(attack_type: str, confidence: float,
                  prob_normal: float, prob_ddos: float, prob_botnet: float) -> str:
    ctx = _CONTEXT.get(attack_type, _CONTEXT["DDoS"])
    return f"""You are a senior SOC analyst for IoT networks. Keep response short and actionable.

DETECTION REPORT:
  Attack    : {attack_type} ({ctx['desc']})
  Confidence: {confidence:.1f}%
  Model out : Normal={prob_normal*100:.1f}% | DDoS={prob_ddos*100:.1f}% | Botnet={prob_botnet*100:.1f}%
  Indicators: {ctx['indicators']}
  Impact    : {ctx['impact']}

Return at most 130 words. Use EXACTLY this format:

THREAT ASSESSMENT:
[max 2 short sentences]

IMMEDIATE ACTIONS:
1. [containment]
2. [mitigation]
3. [monitoring]

RECOMMENDATIONS:
- [hardening]
- [prevention]

SEVERITY: {ctx['severity']}"""


_OPTS = {"temperature": 0.1, "num_predict": 180, "top_p": 0.8, "repeat_penalty": 1.15}


def _chunk_text(text: str, chunk_size: int = 32) -> Generator[str, None, None]:
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _stream_from_ollama(prompt: str):
    with requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "options": _OPTS},
        stream=True,
        timeout=90,
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


def _stream_from_gemini(prompt: str):
    if not GEMINI_API_KEY:
        yield (
            "\nWARNING: Gemini key missing. Set GEMINI_API_KEY in environment.\n"
            "Tip: For local mode you can switch to Ollama by setting LLM_PROVIDER=ollama."
        )
        return

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 220,
            "topP": 0.8,
        },
    }
    r = requests.post(url, json=payload, timeout=45)
    r.raise_for_status()
    data = r.json()

    text = ""
    candidates = data.get("candidates") or []
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)

    if not text.strip():
        text = "No LLM response generated for this alert."

    yield from _chunk_text(text.strip())


def stream_attack_analysis(attack_type: str, confidence: float,
                           prob_normal: float, prob_ddos: float,
                           prob_botnet: float):
    """Generator yielding text chunks for dashboard streaming UI."""
    prompt = _build_prompt(attack_type, confidence, prob_normal, prob_ddos, prob_botnet)
    try:
        if LLM_PROVIDER == "ollama":
            yield from _stream_from_ollama(prompt)
        else:
            yield from _stream_from_gemini(prompt)
    except requests.exceptions.ConnectionError:
        yield (
            "\nWARNING: Cannot reach the configured LLM provider.\n"
            "Check provider settings and network."
        )
    except requests.exceptions.Timeout:
        yield "\nWARNING: LLM request timed out. Next alert will retry."
    except Exception as e:
        yield f"\nWARNING: LLM error: {type(e).__name__}: {e}"


def get_attack_analysis(attack_type: str, confidence: float,
                        prob_normal: float, prob_ddos: float,
                        prob_botnet: float) -> dict:
    """Non-streaming fallback — joins all tokens. Used by CLI tools."""
    response = "".join(
        stream_attack_analysis(attack_type, confidence, prob_normal, prob_ddos, prob_botnet)
    )
    return {"attack_type": attack_type, "confidence": confidence, "response": response.strip()}