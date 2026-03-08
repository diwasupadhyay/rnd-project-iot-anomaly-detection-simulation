"""
llm/interpreter.py — Ollama LLM Interpreter
Supports real-time token streaming via Ollama's stream API.
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

_CONTEXT = {
    "DDoS": {
        "attack_desc" : "Distributed Denial-of-Service flooding IoT network with high-rate SYN/UDP packets.",
        "indicators"  : "Abnormally high packet rate, SYN flag surges, bandwidth saturation, connection timeouts.",
        "impact"      : "Gateway overload, IoT device unresponsiveness, service outage.",
        "severity"    : "Critical",
    },
    "Botnet": {
        "attack_desc" : "Mirai-variant botnet infection — compromised IoT devices executing C2-directed flood attacks.",
        "indicators"  : "Repeated outbound UDP/GRE traffic, C2 beacon patterns, port-scanning probes from IoT IPs.",
        "impact"      : "Devices weaponized for DDoS, lateral spread to vulnerable endpoints, data exfiltration risk.",
        "severity"    : "Critical",
    },
}


def _build_prompt(attack_type: str, confidence: float,
                  prob_normal: float, prob_ddos: float, prob_botnet: float) -> str:
    ctx = _CONTEXT.get(attack_type, _CONTEXT["DDoS"])
    return f"""You are a senior IoT network security analyst. A CNN-LSTM deep learning model just raised a live alert.

=== LIVE DETECTION REPORT ===
Attack Type  : {attack_type}
Confidence   : {confidence:.1f}%
Model Output : Normal={prob_normal*100:.1f}%  DDoS={prob_ddos*100:.1f}%  Botnet={prob_botnet*100:.1f}%
Description  : {ctx['attack_desc']}
Indicators   : {ctx['indicators']}
Impact       : {ctx['impact']}

Respond ONLY in this exact structure. Be specific and concise:

WHAT IS HAPPENING:
[One sentence describing what is occurring on the network right now]

IMMEDIATE ACTIONS:
• [First thing the operator must do RIGHT NOW]
• [Second containment action]
• [Third mitigation step]

LONG-TERM PREVENTION:
• [Infrastructure hardening recommendation]
• [Monitoring/policy improvement]

SEVERITY: {ctx['severity']} — [one-line justification]"""


_OPTS = {"temperature": 0.15, "num_predict": 320, "top_p": 0.9, "repeat_penalty": 1.1}


def stream_attack_analysis(attack_type: str, confidence: float,
                           prob_normal: float, prob_ddos: float,
                           prob_botnet: float):
    """
    Generator — yields text tokens one at a time as Ollama streams them.
    Caller appends each token to card["response"] for live display.
    """
    prompt = _build_prompt(attack_type, confidence, prob_normal, prob_ddos, prob_botnet)
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": True, "options": _OPTS},
            stream=True, timeout=60,
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
            "⚠️  Ollama is not running.\n\n"
            "Start it:  ollama serve\n"
            "Pull model: ollama pull llama3.2:3b\n\n"
            "New attacks will auto-trigger LLM once Ollama is running."
        )
    except requests.exceptions.Timeout:
        yield (
            "⚠️  Ollama timed out (60 s).\n\n"
            "It may be loading the model for the first time — next attack will retry."
        )
    except Exception as e:
        yield f"⚠️  LLM error: {type(e).__name__}: {e}"


def get_attack_analysis(attack_type: str, confidence: float,
                        prob_normal: float, prob_ddos: float,
                        prob_botnet: float) -> dict:
    """Non-streaming fallback (joins all tokens). Used by CLI tools."""
    response = "".join(
        stream_attack_analysis(attack_type, confidence, prob_normal, prob_ddos, prob_botnet)
    )
    return {"attack_type": attack_type, "confidence": confidence, "response": response.strip()}