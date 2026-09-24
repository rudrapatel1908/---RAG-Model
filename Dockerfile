# Single-container deploy: FastAPI serves both /api/* and the
# frontend (see backend/api/main.py's StaticFiles mount). Prebuilt
# FAISS/BM25 indexes ship inside the image (data/indexes/). The
# embedding model is baked into the image at BUILD time (see the
# RUN step below) rather than downloaded on first request - this
# means the container needs zero outbound network access to
# huggingface.co at runtime, which matters on hosts with restricted
# egress (e.g. Railway's trial network limits).
FROM python:3.11-slim

WORKDIR /app

# faiss-cpu and sentence-transformers/torch pull in some packages that
# want basic build tools even when installing prebuilt wheels on
# certain platforms - keep this minimal but present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch explicitly, BEFORE the rest of requirements.txt.
# Standard `pip install torch` from PyPI bundles CUDA runtime libraries
# by default even though this app never uses a GPU - that unused bulk
# sits in memory the moment torch is imported, before the model even
# loads. This forces the real CPU-only wheel instead; identical
# computation and identical scores, just without the GPU baggage.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
# Pre-download the Hugging Face embedding model during build time
# Pre-download the SentenceTransformer model during build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"


# Pre-download the embedding model during the build (network access
# is available at build time on every platform we've tried, unlike
# at runtime on some). Cached into the image's HF cache dir so
# _lazy_load() finds it locally at startup - no code changes needed
# in hybrid_search.py, it already just calls SentenceTransformer(name),
# which checks the local cache before hitting the network.
ENV HF_HOME=/app/.hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY backend /app/backend
COPY retrieval /app/retrieval
COPY frontend /app/frontend
COPY data /app/data

# HF Spaces' Docker SDK expects the app on port 7860. Falls back to
# $PORT (Render/Railway/Fly convention) if set, so this same image
# works on any of them without edits.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
