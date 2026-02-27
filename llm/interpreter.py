"""
llm/interpreter.py — Fast Ollama LLM Interpreter
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def get_attack_analysis(attack_type: str, confidence: float,
                        prob_normal: float, prob_ddos: float,
                        prob_botnet: float) -> dict:

    if attack_type == "DDoS":
        context = "A flood of malicious packets (SYN/UDP) overwhelming IoT devices, causing service disruption."
    else:
        context = "Mirai botnet malware controlling IoT devices for coordinated attacks and scanning."

    prompt = f"""IoT network security alert: {attack_type} detected at {confidence:.0f}% confidence.
{context}

Reply in this exact format (be brief, 2-3 lines max per section):

WHAT IS HAPPENING: [one sentence]

IMMEDIATE ACTIONS:
- [action 1]
- [action 2]
- [action 3]

PREVENTION:
- [action 1]
- [action 2]

SEVERITY: [Critical/High/Medium]"""

    try:
        r = requests.post(OLLAMA_URL, json={
            "model"  : MODEL_NAME,
            "prompt" : prompt,
            "stream" : False,
            "options": {"temperature": 0.2, "num_predict": 250},
        }, timeout=45)
        response = r.json().get("response", "").strip()
    except Exception as e:
        response = f"LLM unavailable: {e}"

    return {
        "attack_type": attack_type,
        "confidence" : confidence,
        "response"   : response,
    }