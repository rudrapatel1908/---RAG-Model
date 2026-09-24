"""
HTTP layer. Wires voice/text input -> STT (if audio) -> harness ->
JSON response. Deliberately thin - all real logic (guardrails,
retrieval, grounding, polish) stays in services/, this just adapts
it to HTTP.

Run from repo root:
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import time
import base64
import logging
import traceback
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "services"))

from config import settings  # noqa: E402
from services.harness import process_query, process_query_fast  # noqa: E402
from services.stt import transcribe_audio  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-rag-api")

app = FastAPI(title="Voice RAG API")

# Permissive CORS for demo purposes - this is a hackathon submission
# served from a single known frontend, not a multi-tenant product.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Without this, any unhandled exception falls through to Starlette's
# default plain-text "Internal Server Error" page - which breaks the
# frontend's res.json() parse (that's the exact bug that surfaced:
# "Unexpected token 'I', 'Internal S'... is not valid JSON"). This
# guarantees /api/* always returns JSON, and logs the full traceback
# server-side so the real cause is visible in the uvicorn terminal.
@app.exception_handler(Exception)
async def json_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s", request.method, request.url.path)
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "detail": "See server logs for the full traceback.",
        },
    )


def _json_safe(obj):
    """Recursively convert numpy/pandas scalar types to native Python
    types. FastAPI's default encoder chokes on numpy.int64/float32/bool_
    (common in retrieved-chunk metadata pulled straight from a pandas
    DataFrame, e.g. is_selected_gt), which throws mid-serialization -
    another way to end up with a non-JSON error response."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        # numpy scalar types (int64, float32, bool_, etc.) all expose .item()
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


class QueryRequest(BaseModel):
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    audio_content_type: Optional[str] = "audio/webm"
    mode: str = "fast"  # "fast" (guaranteed <200ms retrieval budget) or "polished" (adds an LLM call)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/query")
def query(req: QueryRequest):
    t_start = time.perf_counter()
    stt_ms = None
    transcript = req.text

    if req.audio_base64:
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid base64 audio: {e}")

        stt_result = transcribe_audio(audio_bytes, content_type=req.audio_content_type or "audio/webm")
        stt_ms = stt_result["elapsed_ms"]

        if stt_result["error"]:
            return {
                "status": "error",
                "transcript": None,
                "message": f"Speech-to-text failed: {stt_result['error']}",
                "latency": {"stt_ms": stt_ms, "total_ms": (time.perf_counter() - t_start) * 1000},
            }

        transcript = stt_result["text"]
        if not transcript:
            return {
                "status": "blocked",
                "transcript": "",
                "guardrail_flag": "invalid_input",
                "message": "Could not transcribe any speech from the audio.",
                "latency": {"stt_ms": stt_ms, "total_ms": (time.perf_counter() - t_start) * 1000},
            }

    if not transcript:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'audio_base64'.")

    if req.mode == "polished":
        result = process_query(transcript)
    else:
        result = process_query_fast(transcript)

    result["transcript"] = transcript
    if stt_ms is not None:
        result["latency"]["stt_ms"] = stt_ms
        result["latency"]["total_ms"] = stt_ms + result["latency"].get(
            "end_to_end_ms", result["latency"]["fast_path_total_ms"]
        )

    return _json_safe(result)


# Serve the minimal frontend, if present, at the root. Falls back
# gracefully (API-only) if the frontend hasn't been built yet.
FRONTEND_DIR = REPO_ROOT / "frontend"
if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("frontend/index.html not found - serving API only, no UI mounted at /")
