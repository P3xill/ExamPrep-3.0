"""
voice-analyzer-poc — Backend Scaffold (FastAPI)
------------------------------------------------
Local POC server for English→English speech analysis without exposing transcripts.

Endpoints
- GET /health                → quick health check
- POST /process-audio        → accepts audio file (≤180s), returns JSON with scores + metrics + LLM tips

Notes
- All heavy lifting functions are stubbed so you can run end-to-end immediately.
- Automatic speech recognition now relies exclusively on OpenAI’s Whisper API; supply `OPENAI_API_KEY` before starting the server.

Run
- pip install -r requirements.txt  (see inline list below)
- uvicorn app:app --reload --port 5173

Requirements (minimal to run stubs)
- fastapi
- uvicorn[standard]
- pydantic>=2
- python-multipart
- librosa (for duration probing) or ffmpeg via pydub — here we use librosa for simplicity
- numpy

Later (when you swap stubs → real code):
- torch (for Silero VAD)
- language-tool-python (or LT server JAR)
- custom scoring/LLM integrations
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import tempfile
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Optional

import librosa  # duration probing; replace with ffmpeg if preferred
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx
import jwt
from pydantic import BaseModel

app = FastAPI(title="Voice Analyzer POC", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_DURATION_SEC = 180.0
FILLERS = {"uh", "um", "erm", "like"}
FLUENCY_VARIANT_KEYWORDS = {
    "american english": "American English",
    "british english": "British English",
}
FLUENCY_VARIANT_AFFIRM_MARKERS = [
    "fluent",
    "fluently",
    "native speaker",
    "mother tongue",
    "first language",
    "native-level",
    "native level",
    "raised speaking",
    "grew up speaking",
]
FLUENCY_VARIANT_NEGATION_MARKERS = [
    "not fluent",
    "never fluent",
    "not yet fluent",
    "hardly fluent",
    "trying to be fluent",
    "working to be fluent",
    "working on my fluency",
]
FLUENCY_VARIANT_WINDOW_CHARS = 70
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "openai").strip().lower()
if ASR_PROVIDER != "openai":
    raise RuntimeError(
        "This deployment only supports ASR_PROVIDER=openai. Remove ASR_PROVIDER from the environment or set it to 'openai'."
    )
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en").strip() or "en"
OPENAI_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "gpt-4o-mini-transcribe").strip()
GRAMMAR_PROVIDER = os.getenv("GRAMMAR_PROVIDER", "openai").strip().lower()
GRAMMAR_MODEL = os.getenv("GRAMMAR_MODEL", "gpt-5-nano").strip()
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini").strip()

CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "").strip()
CLERK_API_BASE_URL = os.getenv("CLERK_API_BASE_URL", "https://api.clerk.com").rstrip("/")
REQUIRE_CLERK_AUTH = os.getenv("EXAMPREP_REQUIRE_AUTH", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


logger = logging.getLogger("voice_analyzer.whisper")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)

log_level_name = os.getenv("VOICE_ANALYZER_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level_name, logging.INFO))



class AnalysisResponse(BaseModel):
    overall_score: int
    metrics: Dict[str, Any]
    analysis: Dict[str, Any]
    diagnostics: Dict[str, Any] | None = None
    processing_time_sec: float


def _load_audio_samples(path: str) -> tuple[Optional[np.ndarray], Optional[int], float]:
    """Load audio as mono 16 kHz. When decoding fails, fall back to metadata duration."""
    try:
        audio, sr = librosa.load(path, sr=16000, mono=True)
        if sr <= 0:
            raise ValueError("invalid_sample_rate")
        if audio.size == 0:
            raise ValueError("empty_audio")
        duration = float(len(audio) / sr)
        return audio, sr, duration
    except Exception as exc:
        logger.warning("librosa failed to decode audio %s: %s", path, exc)
        duration = _probe_duration_via_mutagen(path)
        if duration is not None:
            logger.info("Falling back to metadata-derived duration (%.3fs)", duration)
            return None, None, duration
        raise RuntimeError("audio_decode_failed") from exc


def _probe_duration_via_mutagen(path: str) -> Optional[float]:
    try:
        from mutagen import File as MutagenFile  # type: ignore
    except ImportError:
        return None

    try:
        audio = MutagenFile(path)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("mutagen could not read %s: %s", path, exc)
        return None

    if not audio:
        return None

    info = getattr(audio, "info", None)
    if not info:
        return None
    length = getattr(info, "length", None)
    if not length:
        return None
    try:
        return float(length)
    except (TypeError, ValueError):
        return None


async def require_clerk_session(authorization: str = Header(None)) -> Dict[str, Any] | None:
    """Validate the Clerk session token if Clerk credentials are configured."""

    if not REQUIRE_CLERK_AUTH or not CLERK_SECRET_KEY:
        # Auth is disabled – allow anonymous access (development / staging)
        return None

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing_auth")

    token = authorization.split(" ", 1)[1].strip()

    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    except jwt.DecodeError as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=401, detail="invalid_auth_token") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=401, detail="invalid_auth_token") from exc

    session_id = payload.get("sid") or payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401, detail="invalid_auth_token")

    verify_url = f"{CLERK_API_BASE_URL}/v1/sessions/{session_id}/verify"
    headers = {"Authorization": f"Bearer {CLERK_SECRET_KEY}"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(verify_url, json={"token": token}, headers=headers)
    except httpx.HTTPError as exc:  # pragma: no cover - network guard
        raise HTTPException(status_code=401, detail="auth_unreachable") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="invalid_session")

    data = response.json()
    return {
        "session_id": session_id,
        "user_id": data.get("user_id"),
        "actor": data.get("actor"),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def _announce_backend() -> None:
    provider_details = {
        "provider": ASR_PROVIDER,
        "openai_model": OPENAI_MODEL,
        "language": ASR_LANGUAGE,
    }
    logger.info("Backend startup configuration: %s", provider_details)


@app.post("/process-audio", response_model=AnalysisResponse)
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    session: Dict[str, Any] | None = Depends(require_clerk_session),
):
    t0 = time.time()
    diagnostics_requested = request.query_params.get("diagnostics", "").lower() in {"1", "true", "yes", "debug"}

    allowed_media_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/aac",
        "audio/x-aac",
        "audio/webm",
        "audio/ogg",
        "application/octet-stream",
    }
    media_type = (file.content_type or "").lower().strip()
    if media_type not in allowed_media_types and not media_type.startswith("audio/"):
        logger.warning("Unsupported content type %s for file %s", file.content_type, file.filename)
        raise HTTPException(status_code=415, detail="unsupported_media_type")

    # Save to a temp WAV/MP3/M4A file
    original_suffix = os.path.splitext(file.filename)[1].lower()
    suffix = original_suffix if original_suffix in {".wav", ".mp3", ".m4a", ".mp4", ".aac"} else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        raw = await file.read()
        tmp.write(raw)
        tmp_path = tmp.name
        tmp.flush()
        os.fsync(tmp.fileno())

    try:
        audio_samples, sample_rate, duration = _load_audio_samples(tmp_path)
    except Exception as exc:
        logger.error("Audio decoding failed: %s", exc)
        raise HTTPException(status_code=400, detail="could_not_read_audio") from exc

    if duration > MAX_DURATION_SEC + 1e-3:
        raise HTTPException(status_code=400, detail="duration_exceeds_limit")

    # 1) ASR — OpenAI Whisper transcription with token confidences
    try:
        asr = _openai_asr(tmp_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("ASR transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail="asr_failed") from exc

    # 2) VAD — STUB: uses energy-based segmentation to estimate pauses
    if audio_samples is not None and sample_rate is not None:
        vad = _stub_vad(audio_samples, sample_rate)
    else:
        vad = _synthetic_vad(asr)

    # 3) Metrics (fluency / pronunciation / correctness)
    fluency = _compute_fluency(asr, vad)
    pronunciation = _compute_pronunciation(asr, audio_samples, sample_rate)
    try:
        correctness = _grammar_correctness(asr)
    except Exception as exc:
        logger.warning("Grammar analysis failed, using fallback stub: %s", exc)
        correctness = _fallback_correctness(asr)

    # 4) Scoring
    overall, metrics = _score(fluency, pronunciation, correctness)

    # 5) LLM analysis — OpenAI summary
    try:
        analysis = _openai_summary(metrics, asr["transcript"], overall)
    except Exception as exc:
        logger.warning("Summary generation failed, using fallback stub: %s", exc)
        analysis = _fallback_summary(metrics, asr["transcript"], overall)

    session_user = session.get("user_id") if session else None

    logger.info(
        "provider=openai model=%s overall=%s user=%s wpm=%.1f fillers=%.1f low_conf_pct=%.1f grammar_errs=%.1f",
        asr.get("model"),
        overall,
        session_user or "anon",
        metrics["fluency"]["wpm"],
        metrics["fluency"]["fillers_per_min"],
        metrics["pronunciation"]["low_conf_token_pct"],
        metrics["correctness"]["grammar_errors_per_100w"],
    )
    logger.debug("transcript_sample=%s", asr["transcript"][:160])

    dt = round(time.time() - t0, 3)

    public_metrics = _public_metric_payload(metrics)
    diagnostics_payload = metrics if diagnostics_requested else None

    return AnalysisResponse(
        overall_score=overall,
        metrics=public_metrics,
        analysis=analysis,
        diagnostics=diagnostics_payload,
        processing_time_sec=dt,
    )


# -----------------------------
# Whisper & helper implementations
# -----------------------------


@lru_cache(maxsize=1)
def _get_openai_client():  # type: ignore[return-any]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - missing optional dependency
        raise RuntimeError("openai package is required for ASR_PROVIDER=openai") from exc

    base_url = os.getenv("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _openai_asr(path: str) -> Dict[str, Any]:
    client = _get_openai_client()
    want_verbose = os.getenv("OPENAI_WHISPER_VERBOSE", "1").strip() not in {"0", "false", "no"}
    request_kwargs: Dict[str, Any] = {"model": OPENAI_MODEL, "language": ASR_LANGUAGE}
    if want_verbose:
        request_kwargs.update(
            {
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"],
            }
        )

    def _call_openai(**kwargs: Any):
        with open(path, "rb") as audio_file:
            return client.audio.transcriptions.create(  # type: ignore[attr-defined]
                file=audio_file,
                **kwargs,
            )

    try:
        response = _call_openai(**request_kwargs)
    except Exception as exc:
        if want_verbose and "response_format" in request_kwargs:
            logger.warning(
                "Model %s does not support verbose_json; retrying with default json", OPENAI_MODEL
            )
            request_kwargs.pop("response_format", None)
            request_kwargs.pop("timestamp_granularities", None)
            response = _call_openai(**request_kwargs)
        else:
            raise

    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "dict"):
        data = response.dict()
    else:  # pragma: no cover - fallback for unexpected client return types
        data = json.loads(response) if isinstance(response, str) else dict(response)

    transcript = (data.get("text") or data.get("transcript") or "").strip()
    if not transcript:
        raise ValueError("empty_transcript")

    segments = data.get("segments") or []
    transcript_parts = []
    tokens: list[str] = []
    token_logprobs: list[float] = []
    synthetic_logprobs = False

    for segment in segments:
        seg_text = (segment.get("text") or "").strip()
        if seg_text:
            transcript_parts.append(seg_text)

        words = segment.get("words") or []
        if words:
            seg_avg_logprob = segment.get("avg_logprob")
            for word in words:
                text_word = (word.get("word") or "").strip()
                if not text_word:
                    continue
                tokens.append(text_word)
                probability = word.get("probability")
                confidence = word.get("confidence")
                if probability is not None:
                    token_logprobs.append(float(np.log(max(probability, 1e-6))))
                elif confidence is not None:
                    token_logprobs.append(float(np.log(max(confidence, 1e-6))))
                elif seg_avg_logprob is not None:
                    token_logprobs.append(float(seg_avg_logprob))
                    synthetic_logprobs = True
                else:
                    token_logprobs.append(-0.55)
                    synthetic_logprobs = True
            continue

        seg_tokens = segment.get("tokens") or []
        if seg_tokens:
            # tokens may be int ids; fall back to rough word split from text
            fallback_tokens = seg_text.split()
            if fallback_tokens:
                tokens.extend(fallback_tokens)
            avg_logprob = segment.get("avg_logprob")
            if avg_logprob is not None:
                token_logprobs.extend([float(avg_logprob)] * len(fallback_tokens or seg_tokens))
            else:
                synthetic_logprobs = True

    if not transcript_parts:
        transcript_parts.append(transcript)

    if not tokens:
        tokens = transcript.split()
        token_logprobs = [-0.65] * len(tokens)
        synthetic_logprobs = True
    elif not token_logprobs:
        token_logprobs = [-0.65] * len(tokens)
        synthetic_logprobs = True

    compression_ratio = data.get("compression_ratio") or 2.0

    return {
        "model": OPENAI_MODEL,
        "transcript": " ".join(transcript_parts).strip() or transcript,
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "compression_ratio": compression_ratio,
        "synthetic_logprobs": synthetic_logprobs,
    }

def _stub_vad(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Very rough energy-based VAD to simulate pauses and voiced segments.
    Replace with silero-vad for production.
    """
    frame = int(0.03 * sr)  # 30 ms
    hop = frame
    energy = np.array([
        np.mean(y[i : i + frame] ** 2) for i in range(0, max(1, len(y) - frame), hop)
    ])
    thresh = float(np.percentile(energy, 60))
    voiced = energy > thresh

    segments = []
    start = None
    for i, v in enumerate(voiced):
        t = i * hop / sr
        if v and start is None:
            start = t
        if (not v) and start is not None:
            segments.append((start, t))
            start = None
    if start is not None:
        segments.append((start, len(y) / sr))

    # Pauses are the gaps between segments (>= 0.25 s)
    pauses = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1][0] - segments[i][1]
        if gap >= 0.25:
            pauses.append(gap)

    voiced_sec = sum(b - a for a, b in segments)
    return {"segments": segments, "pauses": pauses, "voiced_sec": voiced_sec}


def _synthetic_vad(asr: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback VAD when waveform decoding fails (estimates pacing from transcript length)."""
    tokens = asr.get("tokens") or asr.get("transcript", "").split()
    word_count = max(len(tokens), 1)
    estimated_wpm = 140.0
    minutes_voiced = max(word_count / estimated_wpm, 1.0 / 60.0)
    voiced_sec = minutes_voiced * 60.0
    return {"segments": [], "pauses": [], "voiced_sec": voiced_sec}


def _has_variant_fluency_claim(lower_text: str, phrase: str) -> bool:
    if not lower_text or phrase not in lower_text:
        return False

    negations = [
        f"not fluent in {phrase}",
        f"never fluent in {phrase}",
        f"not yet fluent in {phrase}",
        f"hardly fluent in {phrase}",
        f"trying to be fluent in {phrase}",
        f"working to be fluent in {phrase}",
        f"working on my fluency in {phrase}",
    ]
    if any(term in lower_text for term in negations):
        return False

    direct_affirmations = [
        f"fluent in {phrase}",
        f"speak {phrase} fluently",
        f"{phrase} fluently",
        f"native speaker of {phrase}",
        f"{phrase} is my native language",
        f"my native language is {phrase}",
        f"{phrase} is my mother tongue",
        f"my mother tongue is {phrase}",
        f"{phrase} is my first language",
        f"my first language is {phrase}",
        f"grew up speaking {phrase}",
        f"raised speaking {phrase}",
    ]
    if any(term in lower_text for term in direct_affirmations):
        return True

    idx = lower_text.find(phrase)
    while idx != -1:
        start = max(0, idx - FLUENCY_VARIANT_WINDOW_CHARS)
        end = min(len(lower_text), idx + len(phrase) + FLUENCY_VARIANT_WINDOW_CHARS)
        window = lower_text[start:end]
        if any(marker in window for marker in FLUENCY_VARIANT_AFFIRM_MARKERS) and not any(
            neg in window for neg in FLUENCY_VARIANT_NEGATION_MARKERS
        ):
            return True
        idx = lower_text.find(phrase, idx + len(phrase))
    return False


def _detect_fluent_english_variant(transcript: str) -> Optional[str]:
    lower_text = (transcript or "").strip().lower()
    if not lower_text:
        return None

    for phrase, label in FLUENCY_VARIANT_KEYWORDS.items():
        if phrase in lower_text and _has_variant_fluency_claim(lower_text, phrase):
            return label
    return None


def _compute_fluency(asr: Dict[str, Any], vad: Dict[str, Any]) -> Dict[str, Any]:
    words = len(asr["tokens"]) or 1
    minutes_voiced = max(vad["voiced_sec"], 1e-6) / 60.0
    wpm = words / minutes_voiced

    # Fillers/min
    lower_tokens = [t.strip(".,!?").lower() for t in asr["tokens"]]
    filler_count = sum(1 for t in lower_tokens if t in FILLERS)
    fillers_per_min = filler_count / minutes_voiced

    pause_rate = len(vad["pauses"]) / minutes_voiced
    avg_pause = float(np.mean(vad["pauses"])) if vad["pauses"] else 0.0

    # Simple restart heuristic: repeated single words
    restarts = 0
    for i in range(1, len(lower_tokens)):
        if lower_tokens[i] == lower_tokens[i - 1]:
            restarts += 1
    restart_rate = restarts / minutes_voiced

    claimed_variant = _detect_fluent_english_variant(asr.get("transcript", ""))

    return {
        "wpm": wpm,
        "pause_rate_per_min": pause_rate,
        "avg_pause_sec": avg_pause,
        "fillers_per_min": fillers_per_min,
        "restart_rate_per_min": restart_rate,
        "claimed_fluent_variant": claimed_variant,
    }


def _map_piecewise(x: float, pts: list[tuple[float, float]]) -> float:
    """Piecewise-linear mapping. pts = [(x0,y0), (x1,y1), ...] with x ascending."""
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def _score_fluency(f: Dict[str, float]) -> float:
    # Penalise pacing outside 110–150 WPM more sharply to keep grading critical
    wpm_score = _map_piecewise(
        f["wpm"],
        [(60, 30), (90, 55), (110, 72), (125, 78), (145, 66), (165, 54), (185, 44), (215, 32)],
    )

    pause_rate = f["pause_rate_per_min"]
    pause_score = _map_piecewise(
        pause_rate,
        [(0, 60), (0.8, 74), (1.6, 82), (3.0, 68), (5.0, 50), (8.0, 32), (12.0, 18)],
    )

    avg_pause = f["avg_pause_sec"]
    avg_pause_score = _map_piecewise(
        avg_pause,
        [(0.05, 38), (0.2, 65), (0.35, 78), (0.55, 74), (0.8, 56), (1.2, 38)],
    )

    fillers = f["fillers_per_min"]
    filler_score = _map_piecewise(
        fillers,
        [(0, 72), (0.4, 64), (0.9, 54), (1.8, 40), (3.0, 28), (5.5, 12)],
    )

    restarts = f["restart_rate_per_min"]
    restart_score = _map_piecewise(
        restarts,
        [(0, 70), (0.4, 60), (1.2, 46), (2.5, 32), (4.5, 18)],
    )

    return float(np.mean([wpm_score, pause_score, avg_pause_score, filler_score, restart_score]))


def _compute_pronunciation(asr: Dict[str, Any], audio: np.ndarray | None, sr: int | None) -> Dict[str, Any]:
    logprobs = np.array(asr["token_logprobs"]) if asr.get("token_logprobs") else np.array([-1.5])
    avg_logprob = float(np.mean(logprobs))
    median_logprob = float(np.median(logprobs))
    p25_logprob = float(np.percentile(logprobs, 25))
    logprob_std = float(np.std(logprobs))

    low_conf_pct = float(np.mean(logprobs < -1.0) * 100.0)
    unstable_pct = float(np.mean(logprobs < -1.3) * 100.0)
    cr = float(asr.get("compression_ratio", 2.0))

    conf_score = _map_piecewise(
        avg_logprob,
        [(-4.0, 24), (-2.0, 48), (-1.4, 60), (-1.0, 68), (-0.6, 76), (-0.3, 84)],
    )
    stability_score = _map_piecewise(
        unstable_pct,
        [(0.0, 92), (5.0, 78), (12.0, 64), (20.0, 48), (32.0, 30), (48.0, 18)],
    )
    low_conf_score = max(0.0, 92.0 - 4.1 * low_conf_pct)
    cr_score = _map_piecewise(cr, [(1.2, 86), (1.6, 74), (2.0, 62), (2.6, 44), (3.4, 28)])

    signal_score = 68.0
    articulation_flag = False
    if audio is not None and sr and len(audio):
        # Normalise to prevent clipping artefacts
        norm = np.max(np.abs(audio)) or 1.0
        y_norm = audio / norm
        try:
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y_norm, sr=sr)))
            bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y_norm, sr=sr)))
            zero_cross = float(np.mean(librosa.feature.zero_crossing_rate(y_norm)))
        except Exception:
            rolloff = 0.0
            bandwidth = 0.0
            zero_cross = 0.0

        rolloff_ratio = float(np.clip(rolloff / (sr / 2.0), 0.0, 1.0))
        bandwidth_ratio = float(np.clip(bandwidth / (sr / 2.0), 0.0, 1.0))

        signal_score = _map_piecewise(
            (rolloff_ratio + bandwidth_ratio) * 0.5,
            [(0.04, 36), (0.10, 56), (0.18, 74), (0.28, 84), (0.42, 78), (0.58, 62)],
        )
        if zero_cross > 0.14:
            signal_score -= min(20.0, (zero_cross - 0.14) * 220.0)

        articulation_flag = rolloff_ratio < 0.12 or signal_score < 52

    if asr.get("synthetic_logprobs"):
        low_conf_pct = max(low_conf_pct, 18.0)
        low_conf_score = max(0.0, 92.0 - 4.1 * low_conf_pct)
        stability_score = float(np.clip(stability_score * 0.78, 0.0, 100.0))
        conf_score = float(np.clip(conf_score * 0.9, 0.0, 100.0))

    clarity = 0.58 * conf_score + 0.22 * low_conf_score + 0.14 * stability_score + 0.06 * signal_score
    clarity = float(np.clip(clarity - (6.0 if asr.get("synthetic_logprobs") else 0.0), 0.0, 100.0))

    return {
        "avg_logprob": avg_logprob,
        "median_logprob": median_logprob,
        "p25_logprob": p25_logprob,
        "logprob_std": logprob_std,
        "low_conf_token_pct": low_conf_pct,
        "unstable_token_pct": unstable_pct,
        "compression_ratio": cr,
        "signal_score": float(np.clip(signal_score, 0.0, 100.0)),
        "stability_score": float(np.clip(stability_score, 0.0, 100.0)),
        "clarity_score": clarity,
        "articulation_flag": bool(articulation_flag),
        "synthetic_logprobs": bool(asr.get("synthetic_logprobs")),
    }


def _grammar_correctness(asr: Dict[str, Any]) -> Dict[str, Any]:
    if GRAMMAR_PROVIDER != "openai":
        raise RuntimeError("Unsupported GRAMMAR_PROVIDER")

    transcript = asr["transcript"].strip()
    if not transcript:
        raise ValueError("empty_transcript")

    word_count = max(len(asr.get("tokens", [])), 1)
    system_prompt = (
        "You are a meticulous English grammar evaluator for ESL students preparing for speaking exams."
        " Analyse the student's response, produce concise metrics, and label severity using CEFR guidance."
        " Always reply with a SINGLE JSON object and nothing else."
    )
    user_prompt = (
        "Transcript:\n" + transcript + "\n\n"
        "Total words: " + str(word_count) + "\n"
        "Return a JSON object with keys exactly as follows:\n"
        "grammar_errors_per_100w (float),\n"
        "spelling_errors_per_100w (float),\n"
        "top_error_types (array of up to 3 short strings describing the most prominent grammatical issues),\n"
        "cefr_band (string; choose from A1, A2, B1, B2, C1, C2),\n"
        "priority_feedback (array of 1-3 short imperatives personalised to the learner),\n"
        "confidence (float between 0 and 1 indicating how certain you are).\n"
        "Use numbers that reflect mistakes per 100 words. If evidence is thin, make the best estimate you can."
        " Your entire reply must be valid JSON — no prose, no code fences."
    )

    raw = _openai_text_completion(
        model=GRAMMAR_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=260,
    )

    logger.debug("grammar_raw_text=%r", raw)

    data = _parse_grammar_text(raw)

    grammar = float(data.get("grammar_errors_per_100w", 12.0))
    spelling = float(data.get("spelling_errors_per_100w", 3.0))
    # guard rails: enforce minimums and cap severity for realism
    grammar = min(max(grammar, 3.5), 45.0)
    spelling = min(max(spelling, 1.2), 12.0)
    top_errors = data.get("top_error_types") or []
    if not isinstance(top_errors, list):
        top_errors = [str(top_errors)]
    top_errors = [str(e)[:40] for e in top_errors][:3]

    cefr_band = str(data.get("cefr_band", "B1")).upper().strip()
    if cefr_band not in {"A1", "A2", "B1", "B2", "C1", "C2"}:
        cefr_band = "B1"

    priority_feedback = data.get("priority_feedback") or []
    if not isinstance(priority_feedback, list):
        priority_feedback = [str(priority_feedback)]
    priority_feedback = [str(item)[:80] for item in priority_feedback][:3]

    llm_confidence = float(data.get("confidence", 0.55))
    llm_confidence = float(np.clip(llm_confidence, 0.2, 0.95))

    lt_stats = _language_tool_scan(transcript, word_count)
    if lt_stats:
        grammar = float(np.mean([grammar, lt_stats["grammar_errors_per_100w"]]))
        spelling = float(np.mean([spelling, lt_stats["spelling_errors_per_100w"]]))
        top_errors = _merge_topics(top_errors, lt_stats["top_error_types"])
        priority_feedback = priority_feedback or lt_stats["priority_feedback"]
        llm_confidence = float(np.clip((llm_confidence + 0.1), 0.2, 0.98))

    return {
        "grammar_errors_per_100w": round(grammar, 2),
        "spelling_errors_per_100w": round(spelling, 2),
        "top_error_types": top_errors,
        "cefr_band": cefr_band,
        "priority_feedback": priority_feedback,
        "confidence": llm_confidence,
    }


@lru_cache(maxsize=1)
def _get_language_tool():
    try:
        import language_tool_python  # type: ignore
    except Exception as exc:
        raise RuntimeError(str(exc))

    try:
        return language_tool_python.LanguageToolPublic("en-US")
    except AttributeError:
        return language_tool_python.LanguageTool("en-US")


def _language_tool_scan(transcript: str, word_count: int) -> Dict[str, Any] | None:
    try:
        tool = _get_language_tool()
    except Exception:
        return None

    try:
        matches = tool.check(transcript)
    except Exception:
        return None

    if not matches:
        return None

    grammar_hits = 0
    spelling_hits = 0
    topic_counter: Counter[str] = Counter()

    for match in matches:
        issue_type = getattr(match, "ruleIssueType", "")
        if issue_type == "misspelling":
            spelling_hits += 1
        else:
            grammar_hits += 1

        category = getattr(getattr(match, "ruleCategory", None), "name", None)
        label = category or getattr(match, "ruleDescription", "") or getattr(match, "message", "")
        if label:
            topic_counter[label] += 1

    words = max(word_count, 1)
    grammar_per_100 = (grammar_hits / words) * 100.0
    spelling_per_100 = (spelling_hits / words) * 100.0

    top_error_types = [item for item, _ in topic_counter.most_common(3)]
    feedback = []
    if grammar_hits:
        feedback.append("Review the highlighted grammar patterns flagged by LanguageTool.")
    if spelling_hits:
        feedback.append("Double-check spelling of highlighted words.")

    return {
        "grammar_errors_per_100w": round(grammar_per_100, 2),
        "spelling_errors_per_100w": round(spelling_per_100, 2),
        "top_error_types": top_error_types,
        "priority_feedback": feedback[:2],
    }


def _merge_topics(primary: list[str], secondary: list[str]) -> list[str]:
    seen = []
    for item in (primary or []) + (secondary or []):
        if item and item not in seen:
            seen.append(item)
        if len(seen) >= 3:
            break
    return seen


def _fallback_correctness(asr: Dict[str, Any]) -> Dict[str, Any]:
    text = asr["transcript"].lower()
    grammar_err_per_100 = 8.0 if "uh" in text or "um" in text else 5.0
    spelling_err_per_100 = 1.0
    top_errors = ["Articles", "Verb tense"]
    return {
        "grammar_errors_per_100w": grammar_err_per_100,
        "spelling_errors_per_100w": spelling_err_per_100,
        "top_error_types": top_errors,
        "cefr_band": "B1",
        "priority_feedback": [
            "Check article / the / a usage in your sentences.",
            "Practice consistent past/present tense patterns.",
        ],
        "confidence": 0.35,
    }


def _score_correctness(c: Dict[str, float]) -> float:
    g = c["grammar_errors_per_100w"]
    s = c["spelling_errors_per_100w"]
    grammar_score = _map_piecewise(
        g,
        [(0, 78), (2.0, 68), (4.0, 52), (7.0, 40), (11.0, 26), (18.0, 15), (30.0, 8)],
    )
    spelling_score = _map_piecewise(s, [(0, 88), (1.2, 70), (3.0, 48), (5.5, 32), (9.0, 20), (12.0, 12)])
    return 0.92 * grammar_score + 0.08 * spelling_score


def _score_to_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 72:
        return "Strong"
    if score >= 58:
        return "Developing"
    return "Needs Work"


def _public_metric_payload(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for name, data in metrics.items():
        score = float(data.get("score", 0.0))
        payload[name] = {
            "score": round(score, 1),
            "label": _score_to_label(score),
        }
    return payload


def _auto_reason(name: str, data: Dict[str, Any]) -> tuple[str, str]:
    if name == "fluency":
        wpm = int(round(float(data.get("wpm", 0.0) or 0.0)))
        fillers = float(data.get("fillers_per_min", 0.0) or 0.0)
        pauses = float(data.get("pause_rate_per_min", 0.0) or 0.0)
        reason = (
            f"Around {wpm} words a minute with {pauses:.0f} pauses"
            f" and {fillers:.1f} fillers each minute."
        )
        if fillers > 1.2:
            focus = "Trim filler words"
        elif pauses > 3.0:
            focus = "Smooth out pauses"
        else:
            focus = "Keep the flow steady"
        return reason, focus
    if name == "pronunciation":
        low_conf = float(data.get("low_conf_token_pct", 0.0) or 0.0)
        clarity = float(data.get("clarity_score", 0.0) or 0.0)
        reason = (
            f"Clarity sits near {clarity:.0f} with {low_conf:.1f}% syllables the model"
            " hears as shaky."
        )
        focus = "Shape each consonant" if clarity < 75 or low_conf > 15 else "Keep sounds crisp"
        return reason, focus
    if name == "correctness":
        grammar = float(data.get("grammar_errors_per_100w", 0.0) or 0.0)
        reason = (
            f"About {grammar:.1f} grammar slips every 100 words, so sentences need tidying."
        )
        focus = "Tidy up sentence patterns" if grammar > 6 else "Try richer sentence forms"
        return reason, focus
    return "Automated analysis available.", "Keep practising"


def _default_verdicts(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    verdicts: Dict[str, Dict[str, str]] = {}
    for name, data in metrics.items():
        label = _score_to_label(float(data.get("score", 0.0)))
        reason, focus = _auto_reason(name, data)
        verdicts[name] = {
            "label": label,
            "reason": reason[:120],
            "focus": focus[:60],
        }
    return verdicts


def _auto_overall_summary(metrics: Dict[str, Dict[str, Any]], overall_score: int) -> str:
    label = _score_to_label(float(overall_score))
    weakest_name = min(metrics.items(), key=lambda item: item[1].get("score", 0.0))[0]
    weakest_label = weakest_name.capitalize()
    return f"Overall {label.lower()} delivery. Biggest opportunity: {weakest_label.lower()}."


def _auto_ai_summary(
    overall_summary: str,
    verdicts: Dict[str, Dict[str, str]],
    metrics: Dict[str, Dict[str, Any]],
    overall_score: int | None = None,
) -> str:
    strengths: list[tuple[str, str, str]] = []
    gaps: list[tuple[str, str]] = []
    for key, pretty in (("fluency", "Fluency"), ("pronunciation", "Pronunciation"), ("correctness", "Correctness")):
        verdict = verdicts.get(key, {}) if verdicts else {}
        label = str(verdict.get("label", "")).strip()
        reason = str(verdict.get("reason", "")).strip()
        focus = str(verdict.get("focus", "")).strip()
        if label in {"Excellent", "Strong"}:
            detail = reason or focus or f"{pretty} came through clearly."
            strengths.append((pretty, label, detail))
        else:
            detail = focus or reason or f"give {pretty.lower()} a little more polish."
            gaps.append((pretty, detail))

    narrative: list[str] = []
    if overall_score is not None:
        label = _score_to_label(float(overall_score))
        weakest_key = None
        try:
            weakest_key = min(metrics.items(), key=lambda item: float(item[1].get("score", 0.0) or 0.0))[0]
        except ValueError:
            weakest_key = None
        weak_label = {"fluency": "fluency", "pronunciation": "pronunciation", "correctness": "correctness"}.get(
            weakest_key, ""
        )
        article = _indefinite_article(label)
        sentence = f"Overall I'd call this {article} {label.lower()} delivery"
        if weak_label:
            sentence += f", and {weak_label} is where extra reps will pay off fastest"
        narrative.append(sentence + ".")
    elif overall_summary:
        narrative.append(overall_summary.rstrip().rstrip(".") + ".")

    if strengths:
        phrases = []
        for skill, label, detail in strengths[:2]:
            detail_clean = _clean_detail(detail, "it held steady")
            phrases.append(f"your {skill.lower()} stayed {label.lower()} ({detail_clean})")
        strength_sentence = _join_sentence("I like how ", phrases)
        if strength_sentence:
            narrative.append(strength_sentence)

    if gaps:
        phrases = []
        for skill, detail in gaps[:2]:
            detail_clean = _clean_detail(detail, "give it a few more coached runs")
            phrases.append(f"{skill.lower()} — {detail_clean}")
        gap_sentence = _join_sentence("Let's zero in on ", phrases, suffix=" next.")
        if gap_sentence:
            narrative.append(gap_sentence)

    if not narrative:
        return "Keep showing up — each fresh recording will sharpen your delivery."
    return " ".join(narrative[:3]).strip()


def _clean_detail(detail: str, fallback: str) -> str:
    txt = str(detail or "").strip()
    if not txt:
        return fallback
    while txt and txt[-1] in ".!?":
        txt = txt[:-1]
    return txt or fallback


def _join_sentence(prefix: str, phrases: list[str], suffix: str = ".") -> str:
    joined = _natural_join(phrases)
    if not joined:
        return ""
    sentence = f"{prefix}{joined}"
    sentence = sentence.strip()
    if not sentence.endswith((".", "!", "?")):
        sentence += suffix if suffix else "."
    return sentence


def _natural_join(parts: list[str]) -> str:
    items = [p.strip() for p in parts if p and p.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _indefinite_article(word: str) -> str:
    if not word:
        return "a"
    return "an" if word[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _default_tips(metrics: Dict[str, Dict[str, Any]]) -> list[str]:
    tips: list[str] = []
    corr = metrics.get("correctness", {})
    priority = corr.get("priority_feedback") or []
    for item in priority:
        text = str(item).strip()
        if text:
            tips.append(text[:80])
    pron = metrics.get("pronunciation", {})
    if pron.get("clarity_score", 0.0) < 75 and len(tips) < 3:
        tips.append("Record and shadow clear native audio to tighten articulation.")
    flu = metrics.get("fluency", {})
    if flu.get("fillers_per_min", 0.0) > 1.2 and len(tips) < 3:
        tips.append("Insert silent beats instead of filler words.")
    return tips[:3]


def _default_practice_plan(metrics: Dict[str, Dict[str, Any]], transcript: str) -> Dict[str, Any]:
    return {
        "priorities": _default_practice_priorities(metrics),
        "mistake_highlights": _default_mistake_highlights(metrics, transcript),
    }


def _default_practice_priorities(metrics: Dict[str, Dict[str, Any]]) -> list[Dict[str, str]]:
    entries: list[tuple[float, str, Dict[str, Any], str]] = []
    for key, label in (("fluency", "Fluency"), ("pronunciation", "Pronunciation"), ("correctness", "Correctness")):
        data = metrics.get(key, {}) or {}
        score = float(data.get("score", 0.0) or 0.0)
        entries.append((score, label, data, key))
    entries.sort(key=lambda item: item[0])

    priorities: list[Dict[str, str]] = []
    for score, label, data, key in entries[:3]:
        reason, focus = _auto_reason(key, data)
        action = _practice_action(key, data)
        priorities.append(
            {
                "skill": label,
                "focus": focus[:40],
                "why": reason[:90],
                "action": action[:45],
            }
        )
    return priorities


def _practice_action(key: str, data: Dict[str, Any]) -> str:
    if key == "fluency":
        pause_rate = float(data.get("pause_rate_per_min", 0.0) or 0.0)
        fillers = float(data.get("fillers_per_min", 0.0) or 0.0)
        if pause_rate > 3.0:
            return "Rehearse 1-min story with planned breaths"
        if fillers > 1.2:
            return "Swap fillers for a silent beat every pause"
        return "Shadow a native clip to copy steady pacing"
    if key == "pronunciation":
        clarity = float(data.get("clarity_score", 0.0) or 0.0)
        if data.get("articulation_flag"):
            return "Shadow lyrics slowly to hit each consonant"
        if clarity < 72:
            return "Drill mouth shapes for tough consonants"
        return "Keep your daily shadowing streak going"
    if key == "correctness":
        grammar = float(data.get("grammar_errors_per_100w", 0.0) or 0.0)
        if grammar > 6:
            return "Spend 10 min daily on focused grammar drills"
        return "Draft answers with one complex sentence ready"
    return "Keep practising deliberately"


def _default_mistake_highlights(metrics: Dict[str, Dict[str, Any]], transcript: str) -> list[Dict[str, str]]:
    highlights: list[Dict[str, str]] = []
    sentences = _split_sentences(transcript)
    opening = sentences[0] if sentences else transcript.strip()

    flu = metrics.get("fluency", {}) or {}
    pause_rate = float(flu.get("pause_rate_per_min", 0.0) or 0.0)
    if pause_rate > 2.5 and opening:
        highlights.append(
            {
                "excerpt": _trim_excerpt(opening, 18),
                "issue": "Pause-heavy opening",
                "fix": "Write a short intro, rehearse it, and open without long pauses.",
            }
        )

    pron = metrics.get("pronunciation", {}) or {}
    if pron.get("articulation_flag") or float(pron.get("clarity_score", 0.0) or 0.0) < 72:
        target_sentence = sentences[1] if len(sentences) > 1 else opening
        if target_sentence:
            highlights.append(
                {
                    "excerpt": _trim_excerpt(target_sentence, 18),
                    "issue": "Muffled consonants",
                    "fix": "Slow the line, exaggerate every ending sound, then speed back up.",
                }
            )

    corr = metrics.get("correctness", {}) or {}
    top_errors = corr.get("top_error_types") or []
    for error in top_errors:
        issue, fix = _map_error_hint(str(error))
        excerpt = sentences[len(highlights)] if len(sentences) > len(highlights) else opening
        highlights.append(
            {
                "excerpt": _trim_excerpt(excerpt or transcript, 18),
                "issue": issue,
                "fix": fix,
            }
        )
        if len(highlights) >= 3:
            break

    return highlights[:3]


def _map_error_hint(error: str) -> tuple[str, str]:
    key = error.lower()
    if "subject" in key and "verb" in key:
        return "Subject-verb slips", "Spot the subject first, then say the matching singular or plural verb."
    if "tense" in key:
        return "Mixed verb tense", "Choose one timeline and keep every verb in that tense."
    if "article" in key:
        return "Article usage", "Pause a beat to pick a/an/the before you name the noun."
    if "word choice" in key or "vocabulary" in key:
        return "Imprecise word choice", "Swap repeated words for a clearer synonym from your notes."
    if "redund" in key or "repet" in key:
        return "Repetitive phrasing", "Plan a second way to express the idea before you speak."
    return (error[:24] or "Grammar issue", "Review this pattern with a clear example sentence.")


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _trim_excerpt(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _score(fluency: Dict[str, float], pronunciation: Dict[str, float], correctness: Dict[str, float]):
    fluency_score = _score_fluency(fluency)
    claimed_variant_raw = str(fluency.get("claimed_fluent_variant") or "").strip()
    claimed_variant_key = claimed_variant_raw.lower()
    if claimed_variant_key in {"american english", "british english"}:
        fluency_score = 100.0
        updated = dict(fluency)
        updated["score_override_reason"] = f"native_claim:{claimed_variant_key}"
        fluency = updated
    pron_score = pronunciation["clarity_score"]
    corr_score = _score_correctness(correctness)

    overall = int(round(0.35 * fluency_score + 0.30 * pron_score + 0.35 * corr_score))

    metrics = {
        "fluency": {"score": round(fluency_score, 1), **fluency},
        "pronunciation": {"score": round(pron_score, 1), **pronunciation},
        "correctness": {"score": round(corr_score, 1), **correctness},
    }
    return overall, metrics


def _openai_summary(metrics: Dict[str, Any], transcript: str, overall_score: int) -> Dict[str, Any]:
    if SUMMARY_MODEL.lower() in {"stub", "none"}:
        raise RuntimeError("SUMMARY_MODEL disabled")

    shortened_transcript = transcript.strip().split()
    if len(shortened_transcript) > 120:
        shortened_transcript = shortened_transcript[:120]
    snippet = " ".join(shortened_transcript)

    payload = json.dumps(metrics, ensure_ascii=False)
    prompt = (
        "You are an IELTS speaking examiner assigned to give crisp, actionable feedback."
        " Review the provided metrics and transcript excerpt, then respond with a SINGLE JSON object."
        " The JSON must include:\n"
        "overall_summary (string, ≤60 words),\n"
        "verdicts.fluency / verdicts.pronunciation / verdicts.correctness (each object with keys label in {Excellent, Strong, Developing, Needs Work}, reason ≤25 words, focus ≤15 words),\n"
        "tips (array of up to 3 short imperatives ≤12 words),\n"
        "practice_priorities (array, ordered, each item an object with keys skill (one of Fluency, Pronunciation, Correctness), focus (≤10 words), why (≤20 words), action (≤12 words)),\n"
        "mistake_highlights (array of up to 3 objects with keys excerpt (≤18 words quoted from transcript), issue (≤12 words), fix (≤15 words)),\n"
        "ai_summary (string, 2-3 sentences, ≤120 words, weaving together the learner's biggest strengths and their most urgent fixes),\n"
        "confidence (float between 0 and 1).\n"
        "Focus on the biggest weaknesses. Mention strengths only if material.\n"
        "\nMetrics JSON:\n" + payload +
        "\n\nTranscript snippet (first 120 words):\n" + snippet +
        "\n\nRemember: valid JSON only, no code fences, no commentary."
    )

    raw = _openai_text_completion(
        model=SUMMARY_MODEL,
        system_prompt="",
        user_prompt=prompt,
        max_tokens=260,
    )

    logger.debug("summary_raw_text=%r", raw)

    return _normalise_analysis_output(raw, metrics, overall_score, transcript)


def _fallback_summary(metrics: Dict[str, Any], transcript: str, overall_score: int) -> Dict[str, Any]:
    verdicts = _default_verdicts(metrics)
    overall_summary = _auto_overall_summary(metrics, overall_score)
    tips = _default_tips(metrics)
    verdicts = _default_verdicts(metrics)
    if not tips:
        tips = [
            "Record yourself and mimic clear native models.",
            "Plan answers with linking language to avoid hesitations.",
        ][:3]
    practice_plan = _default_practice_plan(metrics, transcript)
    return {
        "overall_summary": overall_summary,
        "verdicts": verdicts,
        "tips": tips,
        "practice_priorities": practice_plan["priorities"],
        "mistake_highlights": practice_plan["mistake_highlights"],
        "ai_summary": _auto_ai_summary(overall_summary, verdicts, metrics, overall_score),
        "confidence": 0.55,
    }


def _normalise_analysis_output(raw: str, metrics: Dict[str, Any], overall_score: int, transcript: str) -> Dict[str, Any]:
    default_verdicts = _default_verdicts(metrics)
    default_plan = _default_practice_plan(metrics, transcript)
    auto_summary = _auto_overall_summary(metrics, overall_score)
    result: Dict[str, Any] = {
        "overall_summary": auto_summary,
        "verdicts": default_verdicts,
        "tips": _default_tips(metrics),
        "confidence": 0.72,
        "practice_priorities": default_plan["priorities"],
        "mistake_highlights": default_plan["mistake_highlights"],
        "ai_summary": _auto_ai_summary(auto_summary, default_verdicts, metrics, overall_score),
    }

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        rationale, tips = _legacy_summary_parse(raw)
        if rationale:
            result["overall_summary"] = rationale[:260]
        if tips:
            result["tips"] = tips[:3]
        return result

    summary = str(data.get("overall_summary", "")).strip()
    if summary:
        result["overall_summary"] = summary[:300]
        result.setdefault("ai_summary", _auto_ai_summary(result["overall_summary"], default_verdicts, metrics, overall_score))

    tips = data.get("tips") or []
    if isinstance(tips, list):
        cleaned = []
        for tip in tips:
            text = str(tip).strip()
            if text:
                cleaned.append(text[:80])
        if cleaned:
            result["tips"] = cleaned[:3]

    verdicts = data.get("verdicts") or {}
    if isinstance(verdicts, dict):
        merged = {}
        for key, default in default_verdicts.items():
            custom = verdicts.get(key, {}) if isinstance(verdicts, dict) else {}
            label = str(custom.get("label", default["label"])).strip() if isinstance(custom, dict) else default["label"]
            reason = str(custom.get("reason", default["reason"])).strip() if isinstance(custom, dict) else default["reason"]
            focus = str(custom.get("focus", default["focus"])).strip() if isinstance(custom, dict) else default["focus"]
            merged[key] = {
                "label": (label or default["label"])[:40],
                "reason": (reason or default["reason"])[:160],
                "focus": (focus or default["focus"])[:80],
            }
        result["verdicts"] = merged
        if not str(data.get("ai_summary", "")).strip():
            result["ai_summary"] = _auto_ai_summary(result["overall_summary"], result["verdicts"], metrics, overall_score)

    priorities = data.get("practice_priorities")
    if isinstance(priorities, list):
        cleaned_priorities = []
        for item in priorities:
            if not isinstance(item, dict):
                continue
            skill = str(item.get("skill", "")).strip()
            focus = str(item.get("focus", "")).strip()
            why = str(item.get("why", "")).strip()
            action = str(item.get("action", "")).strip()
            if skill and focus:
                cleaned_priorities.append(
                    {
                        "skill": skill[:32],
                        "focus": focus[:40],
                        "why": (why or "Focus on this skill.")[:120],
                        "action": (action or "Practice deliberately.")[:80],
                    }
                )
        if cleaned_priorities:
            result["practice_priorities"] = cleaned_priorities[:3]

    highlights = data.get("mistake_highlights")
    if isinstance(highlights, list):
        cleaned_highlights = []
        for item in highlights:
            if not isinstance(item, dict):
                continue
            excerpt = str(item.get("excerpt", "")).strip()
            issue = str(item.get("issue", "")).strip()
            fix = str(item.get("fix", "")).strip()
            if issue and fix:
                cleaned_highlights.append(
                    {
                        "excerpt": excerpt[:140],
                        "issue": issue[:60],
                        "fix": fix[:80],
                    }
                )
        if cleaned_highlights:
            result["mistake_highlights"] = cleaned_highlights[:3]

    ai_summary = str(data.get("ai_summary", "")).strip()
    if ai_summary:
        result["ai_summary"] = ai_summary[:500]

    confidence = data.get("confidence", result["confidence"])
    try:
        result["confidence"] = float(np.clip(float(confidence), 0.2, 0.98))
    except Exception:
        pass

    return result


def _openai_text_completion(*, model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    client = _get_openai_client()
    try:
        completion = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"openai_completion_failed: {exc}") from exc

    if not completion.choices:
        raise ValueError("openai_completion_failed: no choices returned")

    message = completion.choices[0].message
    raw = getattr(message, "content", None)

    if isinstance(raw, list):
        parts = []
        for part in raw:
            text_piece = getattr(part, "text", None)
            if text_piece:
                parts.append(text_piece)
        raw = "".join(parts)

    if raw is None:
        raw = ""

    raw = str(raw).strip()
    if not raw:
        model_dump = getattr(message, "model_dump", None)
        if callable(model_dump):
            dump = model_dump()
        else:
            dump = str(message)
        raise ValueError(f"openai_completion_failed: empty response body :: {dump}")

    return raw


def _openai_responses_text(*, model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    client = _get_openai_client()
    prompt = (
        "System Instruction:\n" + system_prompt.strip() + "\n\n" +
        "User Request:\n" + user_prompt.strip()
    )
    try:
        response = client.responses.create(  # type: ignore[attr-defined]
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"openai_response_failed: {exc}") from exc

    raw = getattr(response, "output_text", None)
    if not raw:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_piece = getattr(content, "text", None)
                if text_piece:
                    chunks.append(text_piece)
        raw = "".join(chunks)
    raw = (raw or "").strip()
    if not raw:
        raise RuntimeError("openai_response_failed: empty output")
    return raw


def _legacy_summary_parse(raw: str) -> tuple[str, list[str]]:
    rationale = "Keep refining your delivery."
    tips: list[str] = []

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in lines:
        upper = line.upper()
        if upper.startswith("SUMMARY:"):
            rationale = line.split(":", 1)[1].strip() or rationale
        elif upper.startswith("TIP:"):
            tips.append(line.split(":", 1)[1].strip())
        elif line.startswith("- "):
            tips.append(line[2:].strip())

    if not tips and rationale:
        # heuristically split rationale into two sentences
        sentences = [s.strip() for s in rationale.replace("?", ".").split(".") if s.strip()]
        if sentences:
            tips = [sentences[0]]
            if len(sentences) > 1:
                tips.append(sentences[1])

    tips = [tip for tip in tips if tip]
    return rationale, tips[:3]


def _parse_grammar_text(raw: str) -> Dict[str, Any]:
    grammar = 12.0
    spelling = 3.0
    top_errors: list[str] = []
    cefr_band = "B1"
    priority_feedback: list[str] = []
    confidence = 0.55

    try:
        data = json.loads(raw)
        grammar = float(data.get("grammar_errors_per_100w", grammar))
        spelling = float(data.get("spelling_errors_per_100w", spelling))
        candidate_errors = data.get("top_error_types") or []
        if isinstance(candidate_errors, list):
            top_errors = [str(e).strip() for e in candidate_errors if str(e).strip()]
        cefr_candidate = str(data.get("cefr_band", cefr_band)).strip().upper()
        if cefr_candidate in {"A1", "A2", "B1", "B2", "C1", "C2"}:
            cefr_band = cefr_candidate
        pf_candidate = data.get("priority_feedback") or []
        if isinstance(pf_candidate, list):
            priority_feedback = [str(item).strip() for item in pf_candidate if str(item).strip()]
        try:
            confidence = float(data.get("confidence", confidence))
        except Exception:
            confidence = confidence
    except json.JSONDecodeError:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for line in lines:
            lower = line.lower()
            if "grammar" in lower and "per" in lower:
                grammar = _extract_float(line, grammar)
            elif "spelling" in lower and "per" in lower:
                spelling = _extract_float(line, spelling)
            elif line.startswith("- "):
                top_errors.append(line[2:].strip())

    grammar = min(max(grammar, 3.5), 45.0)
    spelling = min(max(spelling, 1.2), 12.0)
    top_errors = top_errors[:3] if top_errors else ["Articles", "Verb tense"]
    priority_feedback = priority_feedback[:3]
    confidence = float(np.clip(confidence, 0.2, 0.95))

    return {
        "grammar_errors_per_100w": grammar,
        "spelling_errors_per_100w": spelling,
        "top_error_types": top_errors,
        "cefr_band": cefr_band,
        "priority_feedback": priority_feedback,
        "confidence": confidence,
    }


def _extract_float(text: str, default: float) -> float:
    for token in text.replace(",", ".").split():
        try:
            value = float(token)
            if value >= 0:
                return value
        except ValueError:
            continue
    return default


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5173, reload=True)
