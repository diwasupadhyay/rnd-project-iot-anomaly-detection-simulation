"""
llm/interpreter.py
===================
Provider-agnostic LLM interpreter for attack explanation.

Supports:
    - Gemini API for deployment environments
    - Ollama (local) for offline development
    - Built-in fallback when no provider is available

Environment variables (loaded from .env automatically):
    LLM_PROVIDER=gemini|ollama   (default: gemini)
    GEMINI_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash
    OLLAMA_URL=http://localhost:11434/api/generate
    OLLAMA_MODEL=llama3.2:3b
"""

import json
import os
import time
from typing import Generator

import requests

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

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
        yield from _stream_fallback(prompt)
        return

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"
        f"?alt=sse&key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 220,
            "topP": 0.8,
        },
    }
    try:
        with requests.post(url, json=payload, timeout=45, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="ignore")
                if line_str.startswith("data: "):
                    json_str = line_str[6:]
                    try:
                        data = json.loads(json_str)
                        candidates = data.get("candidates") or []
                        if candidates:
                            parts = (candidates[0].get("content") or {}).get("parts") or []
                            for p in parts:
                                text = p.get("text", "")
                                if text:
                                    yield text
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.RequestException:
        # Fall back to non-streaming if SSE fails
        try:
            url_sync = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
                f"?key={GEMINI_API_KEY}"
            )
            r = requests.post(url_sync, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            text = ""
            candidates = data.get("candidates") or []
            if candidates:
                parts = (candidates[0].get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts)
            if text.strip():
                yield from _chunk_text(text.strip())
            else:
                yield "No response generated."
        except Exception:
            yield from _stream_fallback(prompt)


def _stream_fallback(prompt: str):
    """Built-in fallback analysis when no LLM provider is available."""
    # Parse attack type from the prompt
    attack_type = "DDoS"
    if "Botnet" in prompt:
        attack_type = "Botnet"

    ctx = _CONTEXT.get(attack_type, _CONTEXT["DDoS"])

    if attack_type == "DDoS":
        response = """THREAT ASSESSMENT:
CNN-LSTM model detected a Distributed Denial-of-Service attack with high confidence. Traffic patterns show SYN/UDP flood characteristics consistent with volumetric DDoS targeting IoT gateway infrastructure.

IMMEDIATE ACTIONS:
1. Enable rate-limiting on gateway ports and activate SYN cookie protection
2. Deploy upstream traffic scrubbing and blackhole affected source IPs
3. Monitor bandwidth utilization and connection table saturation

RECOMMENDATIONS:
- Implement network segmentation to isolate IoT subnets from critical infrastructure
- Deploy dedicated DDoS mitigation appliance or cloud-based protection service

SEVERITY: CRITICAL"""
    else:
        response = """THREAT ASSESSMENT:
CNN-LSTM model identified Mirai-variant botnet infection signatures across IoT endpoints. Compromised devices exhibit C2 beacon patterns and outbound attack traffic typical of botnet recruitment.

IMMEDIATE ACTIONS:
1. Quarantine infected devices and block C2 communication channels immediately
2. Reset credentials on all potentially compromised IoT endpoints
3. Monitor for lateral movement and additional infection indicators

RECOMMENDATIONS:
- Enforce firmware updates and disable default credentials on all IoT devices
- Implement network-level bot detection using flow-based behavioral analysis

SEVERITY: CRITICAL"""

    # Stream it chunk by chunk with realistic delay
    for chunk in _chunk_text(response, chunk_size=24):
        time.sleep(0.03)
        yield chunk


def stream_attack_analysis(attack_type: str, confidence: float,
                           prob_normal: float, prob_ddos: float,
                           prob_botnet: float):
    """Generator yielding text chunks for dashboard streaming UI."""
    prompt = _build_prompt(attack_type, confidence, prob_normal, prob_ddos, prob_botnet)
    try:
        if LLM_PROVIDER == "ollama":
            yield from _stream_from_ollama(prompt)
        elif GEMINI_API_KEY:
            yield from _stream_from_gemini(prompt)
        else:
            yield from _stream_fallback(prompt)
    except requests.exceptions.ConnectionError:
        yield from _stream_fallback(prompt)
    except requests.exceptions.Timeout:
        yield from _stream_fallback(prompt)
    except Exception as e:
        yield f"\nAnalysis error: {type(e).__name__}: {e}\n"
        yield from _stream_fallback(prompt)


def get_attack_analysis(attack_type: str, confidence: float,
                        prob_normal: float, prob_ddos: float,
                        prob_botnet: float) -> dict:
    """Non-streaming fallback — joins all tokens. Used by CLI tools."""
    response = "".join(
        stream_attack_analysis(attack_type, confidence, prob_normal, prob_ddos, prob_botnet)
    )
    return {"attack_type": attack_type, "confidence": confidence, "response": response.strip()}