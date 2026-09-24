"""
Central config. All tunables live here so nothing is hardcoded
inside services. Reads from .env (see .env.example).
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # --- Paths ---
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    index_dir: Path = Path(__file__).resolve().parent.parent / "data" / "indexes"

    # --- Dataset ---
    hf_dataset_repo: str = "ai4bharat/MSMARCO-XI"
    hf_config: str = "default"     # verified via ingestion/00_check_configs.py — only config that exists
    hf_split: str = "train"
    target_query_rows: int = 20000

    # --- Chunking ---
    fixed_chunk_size: int = 256
    fixed_chunk_overlap: int = 50
    sentence_chunk_target_size: int = 200
    semantic_similarity_threshold: float = 0.55

    # --- Embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = 8

    # --- Retrieval ---
    served_strategy: str = "sentence"
    vector_top_k: int = 10
    vector_top_k: int = 10
    bm25_top_k: int = 10
    fusion_top_k: int = 5
    rrf_k: int = 60  # standard RRF constant
    max_query_chars: int = 512

    # --- Relevance gate ---
    min_fused_score_to_answer: float = 0.35

    # --- Grounding ---
    min_groundedness_score: float = 0.45
    # CONFIRMED via run_guardrail_tests.py on 2026-08-18 (11/11 pass).
    # Real measured range on this dataset/model:
    #   highest false-positive dense score (nonsense_gibberish): 0.8454
    #   lowest genuine dense-only paraphrase (semantic_paraphrase): 0.8626
    # Centered at the midpoint of that gap (~0.854) rather than at 0.86,
    # which left only 0.0026 margin on the paraphrase side. This gives
    # ~0.0086 clearance on both sides. If you add new guardrail test
    # cases and see a false positive score above ~0.85 or a genuine
    # paraphrase below ~0.86, re-run and re-center this value.
    dense_bypass_score: float = 0.854

    # --- STT ---
    stt_provider: str = "elevenlabs"   # or "sarvam"
    elevenlabs_api_key: str = ""
    sarvam_api_key: str = ""

    # --- Generation ---
    anthropic_api_key: str = ""
    generation_model: str = "claude-sonnet-4-6"
    max_answer_tokens: int = 500

    # --- Server ---
    request_timeout_s: float = 8.0
    stage_timeout_s: float = 3.0
    max_retries: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
settings.index_dir.mkdir(parents=True, exist_ok=True)
