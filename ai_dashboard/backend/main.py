"""
Layer 3 – Clinical AI Reasoning Backend (FastAPI)
Lives in: ai_dashboard/backend/
Completely separated from ESP32 firmware.

Receives structured tremor session summaries from the AI dashboard frontend,
builds a clinical prompt, calls MedGemma 4B via HuggingFace Inference API,
and returns a structured clinical report.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()  # Load .env file (KAGGLE_MEDGEMMA_URL, etc.)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# ML module (Random Forest tremor classifier)
try:
    from ml.model import get_classifier, classify_window, classify_batch, is_model_available
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[ML] ⚠ ML module not available (missing dependencies?)")

# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class BandPower(BaseModel):
    hz_4_6: float = 0.0
    hz_6_8: float = 0.0
    hz_8_12: float = 0.0

class Metadata(BaseModel):
    session_id: str = "unknown"
    timestamp: str = ""
    duration_minutes: float = 0
    sampling_rate_hz: float = 50
    condition: str = "rest"
    medication_status: str = "unknown"
    tremor_score_scale: str = "0_to_10_log_scaled"

class FrequencyProfile(BaseModel):
    band_power_mean: BandPower = BandPower()
    band_power_std: BandPower = BandPower()
    dominant_band: str = ""
    dominance_ratio: float = 0
    dominant_band_percentage: float = 0
    band_switch_count: int = 0

class TremorScoreStats(BaseModel):
    mean: float = 0
    std: float = 0
    min: float = 0
    max: float = 0
    p25: float = 0
    p50: float = 0
    p75: float = 0
    p90: float = 0

class IntensityProfile(BaseModel):
    tremor_score: TremorScoreStats = TremorScoreStats()
    rms_mean: float = 0
    noise_floor_adjusted_intensity: float = 0

class IntensityDistribution(BaseModel):
    low_fraction: float = 0
    moderate_fraction: float = 0
    high_fraction: float = 0
    very_high_fraction: float = 0

class VariabilityProfile(BaseModel):
    coefficient_of_variation: float = 0
    stability_index: float = 0
    spectral_entropy: float = 0
    window_to_window_variance: float = 0

class WithinSessionTrend(BaseModel):
    linear_slope_per_minute_score_units: float = 0
    early_vs_late_change_percent: float = 0
    fatigue_pattern_detected: bool = False

class MultiSessionTrend(BaseModel):
    dominant_band_consistency_last_3: str = ""
    tremor_score_weekly_slope: str = ""
    severity_change_percent: str = ""
    band_shift_detected: bool = False

class SessionSummary(BaseModel):
    metadata: Metadata = Metadata()
    frequency_profile: FrequencyProfile = FrequencyProfile()
    intensity_profile: IntensityProfile = IntensityProfile()
    intensity_distribution: IntensityDistribution = IntensityDistribution()
    variability_profile: VariabilityProfile = VariabilityProfile()
    within_session_trend: WithinSessionTrend = WithinSessionTrend()
    multi_session_trend: MultiSessionTrend = MultiSessionTrend()

class ClinicalResponse(BaseModel):
    clinical_summary: str
    confidence_level: str
    advisory_note: str


class StoredSession(BaseModel):
    id: int
    session_id: str
    timestamp: str
    duration_minutes: float
    mean_score: float
    dominant_band: str
    confidence_level: str


class StoreSessionRequest(BaseModel):
    session: SessionSummary
    clinical_summary: str
    confidence_level: str

class TestPhaseResult(BaseModel):
    phase: str                     # 'rest' | 'postural' | 'movement'
    duration_seconds: float
    session: SessionSummary = SessionSummary()


class DirectSessionRequest(BaseModel):
    session: SessionSummary

# ──────────────────────────────────────────────
# SQLite (built-in) setup (local-only profiles DB)
# ──────────────────────────────────────────────

DB_PATH = os.environ.get("TREMOR_PROFILE_DB_PATH", "./profiles.db")


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              timestamp TEXT NOT NULL,
              duration_minutes REAL NOT NULL,
              mean_score REAL NOT NULL,
              dominant_band TEXT NOT NULL,
              confidence_level TEXT NOT NULL,
              clinical_summary TEXT NOT NULL,
              raw_summary_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id)")
        conn.commit()
    finally:
        conn.close()

# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(title="Tremor Clinical AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    # Ensure the local SQLite database and tables exist
    init_db()
    # Log MedGemma configuration
    if KAGGLE_MEDGEMMA_URL:
        print(f"✅ KAGGLE_MEDGEMMA_URL configured: {KAGGLE_MEDGEMMA_URL}")
        print("   AI reports will use MedGemma via Kaggle ngrok tunnel")
    elif HF_API_TOKEN:
        print("✅ HF_API_TOKEN configured — using HuggingFace Inference API")
    else:
        print("⚠️  No MedGemma backend configured — using rule-based analysis only")
        print("   Set KAGGLE_MEDGEMMA_URL=https://xxxx.ngrok-free.app to enable AI")
    # Load Random Forest model
    if ML_AVAILABLE:
        clf = get_classifier()
        if clf:
            print(f"🌲 Random Forest model loaded ({clf.n_estimators} trees)")
        else:
            print("🌲 No trained RF model found — train with: python -m ml.train_model")

# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

SYSTEM_INSTRUCTION = (
    "You are a clinical reasoning assistant specialized in tremor pattern interpretation. "
    "You do not diagnose disease. You interpret structured tremor biomarkers and generate "
    "professional clinical summaries including severity assessment, phenotype likelihood, "
    "stability analysis, and uncertainty explanation."
)


def format_band(label: Optional[str]) -> str:
    """4_6_hz -> 4–6 Hz"""
    if not label:
        return "unknown"
    return label.replace("_", "–").replace("hz", "Hz").replace("–Hz", " Hz")


def build_user_prompt(s: SessionSummary) -> str:
    """Build a readable clinical text from the structured session summary."""

    bp = s.frequency_profile.band_power_mean
    bp_std = s.frequency_profile.band_power_std
    ts = s.intensity_profile.tremor_score
    dist = s.intensity_distribution
    var = s.variability_profile
    wt = s.within_session_trend
    mt = s.multi_session_trend

    lines = [
        "SESSION METADATA:",
        f"  Session ID: {s.metadata.session_id}",
        f"  Timestamp: {s.metadata.timestamp}",
        f"  Duration: {s.metadata.duration_minutes} minutes",
        f"  Condition: {s.metadata.condition}",
        f"  Medication status: {s.metadata.medication_status}",
        f"  Tremor score scale: {s.metadata.tremor_score_scale}",
        "",
        "FREQUENCY PROFILE:",
        f"  Band power mean – 4–6 Hz: {bp.hz_4_6:.3f}, 6–8 Hz: {bp.hz_6_8:.3f}, 8–12 Hz: {bp.hz_8_12:.3f}",
        f"  Band power std  – 4–6 Hz: {bp_std.hz_4_6:.3f}, 6–8 Hz: {bp_std.hz_6_8:.3f}, 8–12 Hz: {bp_std.hz_8_12:.3f}",
        f"  Dominant band: {format_band(s.frequency_profile.dominant_band)}",
        f"  Dominance ratio: {s.frequency_profile.dominance_ratio:.2f}",
        f"  Dominant band percentage: {s.frequency_profile.dominant_band_percentage*100:.1f}%",
        f"  Band switch count: {s.frequency_profile.band_switch_count}",
        "",
        "INTENSITY PROFILE:",
        f"  Mean tremor score: {ts.mean:.2f}",
        f"  Std: {ts.std:.2f}  |  Min: {ts.min:.2f}  |  Max: {ts.max:.2f}",
        f"  Percentiles – p25: {ts.p25:.2f}, p50: {ts.p50:.2f}, p75: {ts.p75:.2f}, p90: {ts.p90:.2f}",
        f"  RMS mean: {s.intensity_profile.rms_mean:.3f}",
        f"  Noise-floor adjusted intensity: {s.intensity_profile.noise_floor_adjusted_intensity:.3f}",
        "",
        "INTENSITY DISTRIBUTION:",
        f"  Low: {dist.low_fraction*100:.1f}%  |  Moderate: {dist.moderate_fraction*100:.1f}%",
        f"  High: {dist.high_fraction*100:.1f}%  |  Very high: {dist.very_high_fraction*100:.1f}%",
        "",
        "VARIABILITY PROFILE:",
        f"  Coefficient of variation: {var.coefficient_of_variation:.3f}",
        f"  Stability index: {var.stability_index:.3f}",
        f"  Spectral entropy: {var.spectral_entropy:.3f}",
        f"  Window-to-window variance: {var.window_to_window_variance:.3f}",
        "",
        "WITHIN-SESSION TREND:",
        f"  Linear slope (score units/min): {wt.linear_slope_per_minute_score_units:.4f}",
        f"  Early vs late change: {wt.early_vs_late_change_percent:.1f}%",
        f"  Fatigue pattern detected: {'Yes' if wt.fatigue_pattern_detected else 'No'}",
        "",
        "MULTI-SESSION TREND:",
        f"  Dominant band consistency (last 3): {mt.dominant_band_consistency_last_3}",
        f"  Weekly slope: {mt.tremor_score_weekly_slope}",
        f"  Severity change: {mt.severity_change_percent}",
        f"  Band shift detected: {'Yes' if mt.band_shift_detected else 'No'}",
    ]

    user_input = "\n".join(lines)

    request_section = (
        "\n\nBased on the above biomarkers, provide:\n"
        "1. Likely tremor phenotype\n"
        "2. Severity interpretation\n"
        "3. Stability and variability analysis\n"
        "4. Within-session trend interpretation\n"
        "5. Multi-session trend interpretation\n"
        "6. Confidence level (Low / Moderate / High)\n"
        "7. Non-diagnostic advisory disclaimer"
    )

    return user_input + request_section

# ──────────────────────────────────────────────
# MedGemma Inference
# ──────────────────────────────────────────────

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
MEDGEMMA_MODEL = "google/medgemma-4b-it"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MEDGEMMA_MODEL}"

# Kaggle-hosted MedGemma via ngrok tunnel
KAGGLE_MEDGEMMA_URL = os.environ.get("KAGGLE_MEDGEMMA_URL", "").rstrip("/")


async def call_kaggle_medgemma(system_prompt: str, user_prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Call MedGemma via the Kaggle notebook exposed through ngrok.
    The Kaggle Flask API expects:
      POST /predict  {"system_prompt": ..., "user_prompt": ..., "max_new_tokens": ...}
    Returns: {"generated_text": "..."}
    """
    predict_url = KAGGLE_MEDGEMMA_URL + "/predict"
    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_new_tokens": max_new_tokens,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(predict_url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        return result.get("generated_text", "No response generated.")


async def call_medgemma(system_prompt: str, user_prompt: str) -> str:
    """
    Call MedGemma 4B — tries Kaggle ngrok first, then HuggingFace API,
    then falls back to placeholder.
    """

    # 1. Try Kaggle-hosted MedGemma via ngrok
    if KAGGLE_MEDGEMMA_URL:
        try:
            print(f"[MedGemma] Calling Kaggle endpoint: {KAGGLE_MEDGEMMA_URL}/predict")
            result = await call_kaggle_medgemma(system_prompt, user_prompt)
            print(f"[MedGemma] Kaggle response received ({len(result)} chars)")
            return result
        except Exception as e:
            print(f"[MedGemma] Kaggle API error: {e} — falling back…")

    # 2. Try HuggingFace Inference API
    if HF_API_TOKEN:
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n",
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.4,
                "top_p": 0.9,
                "return_full_text": False,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(HF_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response generated.")
                return str(result)
        except Exception as e:
            print(f"[MedGemma HF API error] {e}")

    # 3. Placeholder fallback
    return _placeholder_response(user_prompt)


def _placeholder_response(user_prompt: str) -> str:
    """Generate a structured placeholder when API is unavailable."""
    return (
        "## Tremor Phenotype Assessment\n\n"
        "**Likely Phenotype:** Based on the dominant 4–6 Hz band activity and the "
        "frequency profile provided, the pattern is most consistent with a resting tremor "
        "phenotype, commonly associated with Parkinsonian-spectrum conditions. However, "
        "essential tremor cannot be excluded without postural/kinetic testing.\n\n"
        "**Severity Interpretation:** The mean tremor score indicates moderate tremor "
        "intensity. The score distribution suggests periods of significant tremor activity "
        "interspersed with lower-intensity windows.\n\n"
        "**Stability & Variability:** The stability index and coefficient of variation "
        "suggest a relatively consistent tremor pattern with low spectral entropy, "
        "indicating a narrow-band, well-defined oscillatory pattern.\n\n"
        "**Within-Session Trend:** A mild upward trend in tremor score over the session "
        "may indicate fatigue-related amplification, which is clinically relevant for "
        "medication timing assessment.\n\n"
        "**Multi-Session Trend:** Consistent dominant band across sessions with a slight "
        "upward weekly slope warrants monitoring for progressive changes.\n\n"
        "**Confidence Level:** Moderate — sufficient data quality for pattern interpretation, "
        "but clinical context (medication, activity) is incomplete.\n\n"
        "⚠️ **Advisory:** This analysis is for informational purposes only and does not "
        "constitute a medical diagnosis. A qualified neurologist should interpret these "
        "findings in conjunction with clinical examination."
    )

# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    ai_backend = "kaggle_medgemma" if KAGGLE_MEDGEMMA_URL else ("huggingface" if HF_API_TOKEN else "rule_based")
    return {"status": "ok", "service": "Tremor Clinical AI Backend", "ai_backend": ai_backend}


@app.post("/analyze", response_model=ClinicalResponse)
async def analyze(session: SessionSummary):
    """
    Accepts a structured session summary, builds a clinical prompt,
    calls MedGemma 4B, and returns a structured clinical report.
    """
    user_prompt = build_user_prompt(session)
    raw_response = await call_medgemma(SYSTEM_INSTRUCTION, user_prompt)

    # Determine confidence from the model response or default
    confidence = "Moderate"
    for level in ["High", "Low", "Moderate"]:
        if level.lower() in raw_response.lower():
            confidence = level
            break

    advisory = (
        "This analysis is generated by an AI clinical reasoning assistant and does not "
        "constitute a medical diagnosis. Results should be reviewed by a qualified "
        "healthcare professional. Tremor patterns may vary based on medication, fatigue, "
        "stress, and other clinical factors not captured by sensor data alone."
    )

    return ClinicalResponse(
        clinical_summary=raw_response,
        confidence_level=confidence,
        advisory_note=advisory,
    )


@app.post("/tests/analyze-phase", response_model=ClinicalResponse)
async def analyze_test_phase(payload: TestPhaseResult):
    """
    Accepts a single standardized-test phase (rest / postural / movement),
    stamps the correct condition into metadata, calls MedGemma, and returns
    the clinical analysis for that phase.
    """
    # Overwrite the condition field so the AI prompt carries the correct context
    payload.session.metadata.condition = payload.phase
    if payload.session.metadata.duration_minutes == 0 and payload.duration_seconds > 0:
        payload.session.metadata.duration_minutes = round(payload.duration_seconds / 60, 3)

    user_prompt = build_user_prompt(payload.session)
    # Prepend a short phase-context header so MedGemma knows the test type
    phase_header = (
        f"STANDARDIZED TEST — PHASE: {payload.phase.upper()}\n"
        f"Duration: {payload.duration_seconds:.0f} seconds\n"
        "The patient performed the standardized clinical tremor test as described below.\n"
        "Focus analysis specifically on the tremor characteristics relevant to this test type:\n"
        " • Rest: resting tremor (Parkinsonian-type characteristics)\n"
        " • Postural: postural/sustained-position tremor (essential tremor / dystonic features)\n"
        " • Movement: action/kinetic tremor (cerebellar / essential tremor features)\n\n"
    )
    raw_response = await call_medgemma(SYSTEM_INSTRUCTION, phase_header + user_prompt)

    confidence = "Moderate"
    for level in ["High", "Low", "Moderate"]:
        if level.lower() in raw_response.lower():
            confidence = level
            break

    advisory = (
        f"[{payload.phase.capitalize()} Phase] This AI analysis is generated for the standardized "
        f"{payload.phase} tremor test and does not constitute a medical diagnosis. "
        "Results must be corroborated with clinical examination by a qualified neurologist."
    )

    return ClinicalResponse(
        clinical_summary=raw_response,
        confidence_level=confidence,
        advisory_note=advisory,
    )


# ──────────────────────────────────────────────
# Rule-Based Clinical Interpretation (no AI)
# ──────────────────────────────────────────────

def _rule_based_analysis(s: SessionSummary, phase_label: str = "") -> ClinicalResponse:
    """
    Generate a structured clinical report from raw biomarkers using
    deterministic rules. No external API call needed.
    """
    ts = s.intensity_profile.tremor_score
    fp = s.frequency_profile
    dist = s.intensity_distribution
    var = s.variability_profile
    wt = s.within_session_trend
    mt = s.multi_session_trend
    condition = s.metadata.condition or "rest"

    # ── Severity ──────────────────────────────────
    mean = ts.mean
    if mean < 1.5:
        sev_label, sev_desc = "Minimal", "Tremor intensity is within the minimal range, suggesting negligible motor disruption."
    elif mean < 2.5:
        sev_label, sev_desc = "Mild", "Tremor intensity is mild. This level is typically perceptible but unlikely to interfere with daily activities."
    elif mean < 5.0:
        sev_label, sev_desc = "Moderate", "Tremor intensity is moderate. This may cause noticeable difficulty with fine motor tasks such as writing or eating."
    elif mean < 7.5:
        sev_label, sev_desc = "Moderate-Severe", "Tremor intensity is in the moderate-to-severe range. Functional impairment is likely during most manual activities."
    else:
        sev_label, sev_desc = "Severe", "Tremor intensity is severe. Significant functional impairment is expected across daily living tasks."

    # ── Dominant band → phenotype hints ───────────
    dom = format_band(fp.dominant_band)
    bp = fp.band_power_mean
    if fp.dominant_band == "4_6_hz":
        band_hint = (
            f"The dominant frequency band is {dom} (power {bp.hz_4_6:.3f}), "
            "which is characteristic of a resting/Parkinsonian-type tremor pattern."
        )
    elif fp.dominant_band == "6_8_hz":
        band_hint = (
            f"The dominant frequency band is {dom} (power {bp.hz_6_8:.3f}), "
            "which overlaps with the typical range for essential tremor."
        )
    else:
        band_hint = (
            f"The dominant frequency band is {dom} (power {bp.hz_8_12:.3f}), "
            "which may indicate a higher-frequency physiological or enhanced essential tremor."
        )

    # ── Phenotype reasoning by condition ──────────
    if condition == "rest":
        pheno = (
            "During the rest condition, the detected tremor is most consistent with a **resting tremor** phenotype. "
            "Resting tremor is commonly associated with Parkinsonian-spectrum conditions, though other aetiologies exist."
        )
    elif condition == "postural":
        pheno = (
            "During the postural hold, tremor activity is consistent with a **postural tremor** phenotype. "
            "This pattern is frequently seen in essential tremor and enhanced physiological tremor."
        )
    elif condition == "movement":
        pheno = (
            "During the movement (kinetic) test, tremor characteristics suggest a **kinetic/intention tremor** phenotype. "
            "This pattern can be associated with cerebellar dysfunction or advanced essential tremor."
        )
    else:
        pheno = (
            f"Under the '{condition}' condition, no specific phenotype mapping is available. "
            "Clinical correlation is recommended."
        )

    # ── Stability / Variability ───────────────────
    cv = var.coefficient_of_variation
    se = var.spectral_entropy
    if cv < 0.15:
        stab = "Tremor pattern is **highly stable** with low variability, indicating a consistent oscillatory source."
    elif cv < 0.35:
        stab = "Tremor shows **moderate variability**, which is common in functional tremor states."
    else:
        stab = "Tremor is **highly variable**, suggesting fluctuating motor drive or possible mixed aetiology."

    if se < 0.4:
        entropy_note = "Low spectral entropy indicates a narrow-band, well-defined oscillation."
    elif se < 0.7:
        entropy_note = "Moderate spectral entropy suggests energy is spread across multiple frequency bands."
    else:
        entropy_note = "High spectral entropy indicates broad-spectrum activity without a clear dominant oscillation."

    # ── Within-session trend ──────────────────────
    slope = wt.linear_slope_per_minute_score_units
    fatigue = wt.fatigue_pattern_detected
    if fatigue or slope > 0.1:
        trend_note = (
            f"Within-session trend shows an **upward drift** (slope {slope:+.4f} score-units/min), "
            "suggesting possible fatigue-related amplification. This is clinically relevant for medication timing."
        )
    elif slope < -0.1:
        trend_note = (
            f"Within-session trend shows a **downward drift** (slope {slope:+.4f} score-units/min), "
            "indicating tremor attenuation over time — possibly adaptation or relaxation effect."
        )
    else:
        trend_note = "Within-session intensity remained **stable** throughout the recording."

    # ── Distribution summary ──────────────────────
    dist_note = (
        f"Intensity distribution: {dist.low_fraction*100:.0f}% low, "
        f"{dist.moderate_fraction*100:.0f}% moderate, "
        f"{dist.high_fraction*100:.0f}% high, "
        f"{dist.very_high_fraction*100:.0f}% very-high."
    )

    stats_note = (
        f"Score statistics — mean: {ts.mean:.2f}, std: {ts.std:.2f}, "
        f"range: [{ts.min:.2f} – {ts.max:.2f}], "
        f"median (p50): {ts.p50:.2f}, p90: {ts.p90:.2f}."
    )

    # ── Confidence ────────────────────────────────
    dur = s.metadata.duration_minutes
    if dur >= 0.5 and cv < 0.3:
        conf = "High"
    elif dur >= 0.25:
        conf = "Moderate"
    else:
        conf = "Low"

    # ── Assemble report ───────────────────────────
    phase_str = f" [{phase_label.upper()} PHASE]" if phase_label else ""
    sections = [
        f"## Tremor Analysis Report{phase_str}\n",
        f"**Severity: {sev_label}**\n{sev_desc}\n",
        f"**Frequency Profile**\n{band_hint}\n",
        f"**Phenotype Assessment**\n{pheno}\n",
        f"**Stability & Variability**\n{stab} {entropy_note}\n",
        f"**Intensity Distribution**\n{dist_note}\n{stats_note}\n",
        f"**Within-Session Trend**\n{trend_note}\n",
    ]

    if mt.severity_change_percent and mt.severity_change_percent != "N/A (first session)":
        sections.append(
            f"**Multi-Session Trend**\n"
            f"Weekly slope: {mt.tremor_score_weekly_slope}, "
            f"severity change: {mt.severity_change_percent}. "
            f"Band consistency: {mt.dominant_band_consistency_last_3}. "
            f"Band shift detected: {'Yes' if mt.band_shift_detected else 'No'}.\n"
        )

    advisory = (
        "This analysis is generated from raw sensor biomarkers using deterministic rules "
        "and does not constitute a medical diagnosis. Results should be reviewed by a "
        "qualified healthcare professional. Tremor patterns may vary based on medication, "
        "fatigue, stress, and other clinical factors not captured by sensor data alone."
    )

    return ClinicalResponse(
        clinical_summary="\n".join(sections),
        confidence_level=conf,
        advisory_note=advisory,
    )


@app.post("/profile/session/direct", response_model=StoredSession)
async def store_session_direct(payload: DirectSessionRequest):
    """
    Analyse a session and persist it into the local SQLite profile database.
    Tries MedGemma (via Kaggle ngrok) first for AI-powered analysis;
    falls back to rule-based biomarker interpretation if unavailable.
    """
    report = None
    used_ai = False

    # Try MedGemma via Kaggle ngrok (call directly so failures raise)
    if KAGGLE_MEDGEMMA_URL:
        try:
            user_prompt = build_user_prompt(payload.session)
            print(f"[Session] Calling Kaggle MedGemma at {KAGGLE_MEDGEMMA_URL}/predict …")
            raw_response = await call_kaggle_medgemma(SYSTEM_INSTRUCTION, user_prompt)
            print(f"[Session] ✅ MedGemma AI response received ({len(raw_response)} chars)")

            confidence = "Moderate"
            for level in ["High", "Low", "Moderate"]:
                if level.lower() in raw_response.lower():
                    confidence = level
                    break

            advisory = (
                "This analysis is generated by MedGemma AI clinical reasoning assistant "
                "and does not constitute a medical diagnosis. Results should be reviewed "
                "by a qualified healthcare professional."
            )
            report = ClinicalResponse(
                clinical_summary=raw_response,
                confidence_level=confidence,
                advisory_note=advisory,
            )
            used_ai = True
        except Exception as e:
            print(f"[Session] ❌ Kaggle MedGemma failed: {e}")
            print("[Session] Falling back to rule-based analysis")

    # Fallback to rule-based (uses your actual sensor data)
    if report is None:
        report = _rule_based_analysis(payload.session)
        print("[Session] 📊 Using rule-based analysis")

    s = payload.session
    mean_score = s.intensity_profile.tremor_score.mean
    dom_band = s.frequency_profile.dominant_band or ""
    ts_val = s.metadata.timestamp or datetime.utcnow().isoformat()
    session_id = s.metadata.session_id or f"S{int(datetime.now(tz=timezone.utc).timestamp())}"
    created_at = datetime.now(tz=timezone.utc).isoformat()

    conn = _connect_db()
    try:
        cur = conn.execute(
            """
            INSERT INTO sessions
              (session_id, timestamp, duration_minutes, mean_score, dominant_band, confidence_level, clinical_summary, raw_summary_json, created_at)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                ts_val,
                float(s.metadata.duration_minutes),
                float(mean_score),
                str(dom_band),
                str(report.confidence_level),
                str(report.clinical_summary),
                json.dumps(s.model_dump()),
                created_at,
            ),
        )
        conn.commit()
        new_id = int(cur.lastrowid)
    finally:
        conn.close()

    return StoredSession(
        id=new_id,
        session_id=session_id,
        timestamp=ts_val,
        duration_minutes=float(s.metadata.duration_minutes),
        mean_score=float(mean_score),
        dominant_band=str(dom_band),
        confidence_level=str(report.confidence_level),
    )


@app.post("/tests/analyze-phase/direct", response_model=ClinicalResponse)
async def analyze_test_phase_direct(payload: TestPhaseResult):
    """
    Analyse a single standardized test phase.
    Tries MedGemma (via Kaggle ngrok) first; falls back to rule-based.
    """
    payload.session.metadata.condition = payload.phase
    if payload.session.metadata.duration_minutes == 0 and payload.duration_seconds > 0:
        payload.session.metadata.duration_minutes = round(payload.duration_seconds / 60, 3)

    # Try MedGemma via Kaggle ngrok (call directly so failures raise)
    if KAGGLE_MEDGEMMA_URL:
        try:
            user_prompt = build_user_prompt(payload.session)
            phase_header = (
                f"STANDARDIZED TEST — PHASE: {payload.phase.upper()}\n"
                f"Duration: {payload.duration_seconds:.0f} seconds\n"
                "The patient performed the standardized clinical tremor test as described below.\n"
                "Focus analysis specifically on the tremor characteristics relevant to this test type:\n"
                " • Rest: resting tremor (Parkinsonian-type characteristics)\n"
                " • Postural: postural/sustained-position tremor (essential tremor / dystonic features)\n"
                " • Movement: action/kinetic tremor (cerebellar / essential tremor features)\n\n"
            )
            print(f"[Test Phase] Calling Kaggle MedGemma for {payload.phase} …")
            raw_response = await call_kaggle_medgemma(SYSTEM_INSTRUCTION, phase_header + user_prompt)
            print(f"[Test Phase] ✅ MedGemma response for {payload.phase} ({len(raw_response)} chars)")

            confidence = "Moderate"
            for level in ["High", "Low", "Moderate"]:
                if level.lower() in raw_response.lower():
                    confidence = level
                    break

            advisory = (
                f"[{payload.phase.capitalize()} Phase] This AI analysis is generated by MedGemma "
                f"for the standardized {payload.phase} tremor test and does not constitute a medical "
                "diagnosis. Results must be corroborated with clinical examination by a qualified neurologist."
            )
            return ClinicalResponse(
                clinical_summary=raw_response,
                confidence_level=confidence,
                advisory_note=advisory,
            )
        except Exception as e:
            print(f"[Test Phase] ❌ Kaggle MedGemma failed for {payload.phase}: {e}")
            print(f"[Test Phase] Falling back to rule-based for {payload.phase}")

    return _rule_based_analysis(payload.session, phase_label=payload.phase)


# ──────────────────────────────────────────────
# Profile + session history endpoints (local SQLite)
# ──────────────────────────────────────────────

@app.post("/profile/session", response_model=StoredSession)
async def store_session(
    payload: StoreSessionRequest
):
    """
    Persist a single session's biomarkers and AI interpretation
    into a local SQLite database. This is intentionally single-profile
    (no multi-user accounts) and runs only on the local machine.
    """

    s = payload.session
    mean_score = s.intensity_profile.tremor_score.mean
    dom_band = s.frequency_profile.dominant_band or ""
    ts = s.metadata.timestamp or datetime.utcnow().isoformat()
    session_id = s.metadata.session_id or f"S{int(datetime.now(tz=timezone.utc).timestamp())}"
    created_at = datetime.now(tz=timezone.utc).isoformat()

    conn = _connect_db()
    try:
        cur = conn.execute(
            """
            INSERT INTO sessions
              (session_id, timestamp, duration_minutes, mean_score, dominant_band, confidence_level, clinical_summary, raw_summary_json, created_at)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                ts,
                float(s.metadata.duration_minutes),
                float(mean_score),
                str(dom_band),
                str(payload.confidence_level),
                str(payload.clinical_summary),
                json.dumps(s.model_dump()),
                created_at,
            ),
        )
        conn.commit()
        new_id = int(cur.lastrowid)
    finally:
        conn.close()

    return StoredSession(
        id=new_id,
        session_id=session_id,
        timestamp=ts,
        duration_minutes=float(s.metadata.duration_minutes),
        mean_score=float(mean_score),
        dominant_band=str(dom_band),
        confidence_level=str(payload.confidence_level),
    )


@app.get("/profile/sessions", response_model=List[StoredSession])
async def list_sessions():
    """
    Return a lightweight list of all stored sessions for the single
    local profile. Frontend uses this for graphs and trend detection.
    """

    conn = _connect_db()
    try:
        rows = conn.execute(
            """
            SELECT id, session_id, timestamp, duration_minutes, mean_score, dominant_band, confidence_level
            FROM sessions
            ORDER BY timestamp ASC, id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    return [
        StoredSession(
            id=int(r["id"]),
            session_id=str(r["session_id"]),
            timestamp=str(r["timestamp"]),
            duration_minutes=float(r["duration_minutes"]),
            mean_score=float(r["mean_score"]),
            dominant_band=str(r["dominant_band"]),
            confidence_level=str(r["confidence_level"]),
        )
        for r in rows
    ]


@app.get("/profile/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """
    Return full stored JSON + AI summary for a single session.
    Useful for deep dives from the profiles screen.
    """

    conn = _connect_db()
    try:
        row = conn.execute(
            """
            SELECT *
            FROM sessions
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    clinical_text = str(row["clinical_summary"])
    # Detect if rule-based (starts with ## Tremor Analysis Report) or AI-generated
    is_rule_based = clinical_text.startswith("## Tremor Analysis Report")
    advisory = (
        "This analysis is generated from raw sensor biomarkers using deterministic rules "
        "and does not constitute a medical diagnosis. Results should be reviewed by a "
        "qualified healthcare professional."
    ) if is_rule_based else (
        "This analysis is generated by MedGemma AI clinical reasoning assistant "
        "and does not constitute a medical diagnosis. Results should be reviewed "
        "by a qualified healthcare professional."
    )

    return {
        "id": int(row["id"]),
        "session_id": str(row["session_id"]),
        "timestamp": str(row["timestamp"]),
        "duration_minutes": float(row["duration_minutes"]),
        "mean_score": float(row["mean_score"]),
        "dominant_band": str(row["dominant_band"]),
        "confidence_level": str(row["confidence_level"]),
        "clinical_summary": clinical_text,
        "advisory_note": advisory,
        "raw_summary": json.loads(row["raw_summary_json"] or "{}"),
    }


# ──────────────────────────────────────────────
# ML Classification Endpoints (Random Forest)
# ──────────────────────────────────────────────

class MLWindowRequest(BaseModel):
    b1: float
    b2: float
    b3: float
    meanNorm: float = 0.0

class MLBatchRequest(BaseModel):
    windows: List[dict]


@app.post("/ml/classify")
async def ml_classify_single(req: MLWindowRequest):
    """
    Classify a single ESP32 window using the Random Forest model.
    Falls back to rule-based if no model is trained.
    """
    if not ML_AVAILABLE:
        raise HTTPException(503, "ML module not available")
    result = classify_window(req.b1, req.b2, req.b3, req.meanNorm)
    return result


@app.post("/ml/classify-batch")
async def ml_classify_batch(req: MLBatchRequest):
    """
    Classify a batch of windows (e.g. a full session) and return
    aggregate predictions with per-window breakdown.
    """
    if not ML_AVAILABLE:
        raise HTTPException(503, "ML module not available")
    result = classify_batch(req.windows)
    return result


@app.get("/ml/status")
async def ml_status():
    """Check if the ML model is loaded and ready."""
    if not ML_AVAILABLE:
        return {"available": False, "reason": "ML module not installed"}
    model_ready = is_model_available()
    clf = get_classifier() if model_ready else None
    return {
        "available": model_ready,
        "model_loaded": clf is not None,
        "n_estimators": clf.n_estimators if clf else None,
        "n_features": clf.n_features_in_ if clf else None,
        "classes": ["no_tremor", "parkinsonian", "essential", "physiological"],
    }


# ──────────────────────────────────────────────
# Serve AI Dashboard frontend (../index.html)
# Mount AFTER API routes so /analyze is not shadowed
# ──────────────────────────────────────────────

import pathlib
FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ──────────────────────────────────────────────
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
