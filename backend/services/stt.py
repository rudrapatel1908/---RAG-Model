"""
Speech-to-text via ElevenLabs (settings.stt_provider == "elevenlabs").
Verified against the installed `elevenlabs` SDK's real interface
(elevenlabs==2.64.0): client.speech_to_text.convert(model_id=...,
file=(filename, bytes, content_type)) -> response with a `.text`
attribute. model_id is "scribe_v1" or "scribe_v2".

Verified end-to-end against the real API on 2026-08-18 (both a local
.m4a file and a live browser recording via the /api/query route) -
see README.md.
"""
import sys
import time
import logging
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

logger = logging.getLogger("voice-rag.stt")

# Transport-level failures (DNS, connection reset, timeout) surface as
# raw httpx exceptions from this SDK, not wrapped - retry those always.
_TRANSPORT_ERRORS = (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)

# HTTP status codes worth retrying: rate limit + server-side transient
# errors. NOT 400/401/403/404/422 - those fail identically every time
# (bad audio, bad key, bad request shape) and retrying just burns time.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _TRANSPORT_ERRORS):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in _RETRYABLE_STATUS_CODES


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _convert_with_retry(client, **kwargs):
    try:
        return client.speech_to_text.convert(**kwargs)
    except Exception as e:
        if _is_retryable(e):
            logger.warning("Transient ElevenLabs STT error, will retry: %s: %s", type(e).__name__, e)
        raise


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm", content_type: str = "audio/webm") -> dict:
    """Returns {"text": str, "elapsed_ms": float, "error": str|None}.
    Transient failures (rate limit, 5xx, connection drop) are retried
    up to 3x with exponential backoff before giving up."""
    t0 = time.perf_counter()

    if settings.stt_provider != "elevenlabs":
        return {"text": "", "elapsed_ms": 0.0, "error": f"stt_provider '{settings.stt_provider}' not implemented"}

    if not settings.elevenlabs_api_key:
        return {"text": "", "elapsed_ms": 0.0, "error": "ELEVENLABS_API_KEY not set in .env"}

    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        response = _convert_with_retry(
            client,
            model_id="scribe_v1",
            file=(filename, audio_bytes, content_type),
        )
        text = getattr(response, "text", "") or ""
        elapsed = (time.perf_counter() - t0) * 1000
        return {"text": text.strip(), "elapsed_ms": elapsed, "error": None}
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"text": "", "elapsed_ms": elapsed, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    # Manual smoke test - point at a real short audio file to verify
    # the live API call actually works before relying on it in main.py.
    if len(sys.argv) < 2:
        print("Usage: python services/stt.py <path_to_audio_file>")
        sys.exit(1)
    path = Path(sys.argv[1])
    data = path.read_bytes()
    ext = path.suffix.lstrip(".") or "webm"
    content_types = {"webm": "audio/webm", "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}
    result = transcribe_audio(data, filename=path.name, content_type=content_types.get(ext, "audio/webm"))
    print(result)
