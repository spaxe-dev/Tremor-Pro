"""
Test script for the Kaggle-hosted MedGemma API.
Usage:
    1. Set your Kaggle ngrok URL:
         set KAGGLE_MEDGEMMA_URL=https://xxxx.ngrok-free.app
    2. Run:
         python test_kaggle_api.py
"""

import os
import sys
import json
import time
import httpx

# ── Configuration ──────────────────────────────────────
BASE_URL = os.environ.get("KAGGLE_MEDGEMMA_URL", "").rstrip("/")

if not BASE_URL:
    print("❌ KAGGLE_MEDGEMMA_URL is not set.")
    print("   Run:  set KAGGLE_MEDGEMMA_URL=https://xxxx.ngrok-free.app")
    sys.exit(1)

HEALTH_URL  = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    icon = "✅" if ok else "❌"
    print(f"  {icon} {name}")
    if detail:
        print(f"     ↳ {detail}")
    if ok:
        passed += 1
    else:
        failed += 1


# ── Test 1: Health Check ──────────────────────────────
print("\n🔹 Test 1 — Health Check")
try:
    r = httpx.get(HEALTH_URL, timeout=15)
    data = r.json()
    report("Status code 200", r.status_code == 200, f"Got {r.status_code}")
    report("Response has 'status' field", "status" in data, json.dumps(data))
except Exception as e:
    report("Connection to /health", False, str(e))


# ── Test 2: Simple Prediction ────────────────────────
print("\n🔹 Test 2 — Simple Prediction")
try:
    payload = {
        "system_prompt": "You are a helpful medical assistant.",
        "user_prompt": "What is a resting tremor? Answer in one sentence.",
        "max_new_tokens": 128,
    }
    start = time.time()
    r = httpx.post(PREDICT_URL, json=payload, timeout=120)
    elapsed = time.time() - start

    data = r.json()
    text = data.get("generated_text", "")

    report("Status code 200", r.status_code == 200, f"Got {r.status_code}")
    report("Response has 'generated_text'", bool(text), f"Length: {len(text)} chars")
    report(f"Latency", True, f"{elapsed:.1f}s")
    print(f"\n     Response preview:\n     \"{text[:200]}{'...' if len(text) > 200 else ''}\"")
except Exception as e:
    report("Connection to /predict", False, str(e))


# ── Test 3: Clinical Tremor Analysis ─────────────────
print("\n🔹 Test 3 — Full Tremor Session Analysis")
try:
    payload = {
        "system_prompt": (
            "You are a clinical reasoning assistant specialized in tremor "
            "pattern interpretation. You do not diagnose disease. You interpret "
            "structured tremor biomarkers and generate professional clinical summaries."
        ),
        "user_prompt": (
            "SESSION METADATA:\n"
            "  Session ID: test-001\n"
            "  Duration: 5.0 minutes\n"
            "  Condition: rest\n"
            "  Medication status: off\n\n"
            "FREQUENCY PROFILE:\n"
            "  Band power mean – 4–6 Hz: 0.450, 6–8 Hz: 0.120, 8–12 Hz: 0.080\n"
            "  Dominant band: 4–6 Hz\n"
            "  Dominance ratio: 3.75\n\n"
            "INTENSITY PROFILE:\n"
            "  Mean tremor score: 4.20\n"
            "  Std: 1.10  |  Min: 1.50  |  Max: 7.80\n"
            "  RMS mean: 0.350\n\n"
            "VARIABILITY PROFILE:\n"
            "  Coefficient of variation: 0.262\n"
            "  Stability index: 0.780\n"
            "  Spectral entropy: 0.310\n\n"
            "WITHIN-SESSION TREND:\n"
            "  Linear slope (score units/min): 0.0150\n"
            "  Fatigue pattern detected: Yes\n\n"
            "Based on the above biomarkers, provide:\n"
            "1. Likely tremor phenotype\n"
            "2. Severity interpretation\n"
            "3. Stability and variability analysis\n"
            "4. Confidence level (Low / Moderate / High)\n"
            "5. Non-diagnostic advisory disclaimer"
        ),
        "max_new_tokens": 1024,
    }
    start = time.time()
    r = httpx.post(PREDICT_URL, json=payload, timeout=180)
    elapsed = time.time() - start

    data = r.json()
    text = data.get("generated_text", "")

    report("Status code 200", r.status_code == 200, f"Got {r.status_code}")
    report("Response has content", len(text) > 50, f"Length: {len(text)} chars")
    report("Mentions tremor", "tremor" in text.lower(), "Keyword check")
    report(f"Latency", True, f"{elapsed:.1f}s")
    print(f"\n     Full response:\n{'='*60}")
    print(text)
    print(f"{'='*60}")
except Exception as e:
    report("Connection to /predict", False, str(e))


# ── Test 4: Error Handling (empty prompt) ─────────────
print("\n🔹 Test 4 — Empty Prompt (Error Handling)")
try:
    payload = {"system_prompt": "", "user_prompt": ""}
    r = httpx.post(PREDICT_URL, json=payload, timeout=60)
    report("Server didn't crash", r.status_code in (200, 400, 422), f"Status: {r.status_code}")
except Exception as e:
    report("Connection to /predict", False, str(e))


# ── Summary ───────────────────────────────────────────
print(f"\n{'='*40}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*40}\n")
sys.exit(1 if failed else 0)
