import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
"""
Tier 1 retrieval: dense (FAISS) + sparse (BM25) search over the
served chunk strategy, combined via Reciprocal Rank Fusion (RRF).
Thread pools capped above (before numpy/torch/faiss import) to reduce
peak memory on Railway's 1GB trial tier - each thread holds its own
compute buffers, and this is the difference between fitting and an
OOM kill during model load or inference.
"""
import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from tokenizer import tokenize  # noqa: E402

STRATEGY = settings.served_strategy

_model = None
_faiss_index = None
_bm25_data = None
_meta_df = None


def _lazy_load():
    global _model, _faiss_index, _bm25_data, _meta_df
    if _model is None:
        print(f"[retrieval] loading embedding model {settings.embedding_model} ...")
        _model = SentenceTransformer(settings.embedding_model, model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True})
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(settings.index_dir / f"faiss_{STRATEGY}.index"))
    if _bm25_data is None:
        with open(settings.index_dir / f"bm25_{STRATEGY}.pkl", "rb") as f:
            _bm25_data = pickle.load(f)
    if _meta_df is None:
        _meta_df = pd.read_parquet(settings.index_dir / f"meta_{STRATEGY}.parquet")
        _meta_df = _meta_df.reset_index(drop=True)


def dense_search(query: str, k: int) -> tuple[list[int], list[float], float]:
    t0 = time.perf_counter()
    q_emb = _model.encode(["query: " + query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = _faiss_index.search(q_emb, k)
    elapsed = (time.perf_counter() - t0) * 1000
    valid = [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
    rows = [v[0] for v in valid]
    scs = [v[1] for v in valid]
    return rows, scs, elapsed


def sparse_search(query: str, k: int) -> tuple[list[int], list[float], float]:
    t0 = time.perf_counter()
    tokens = tokenize(query)
    query_token_set = set(tokens)
    # Short queries (<=3 content words) need ALL words to co-occur - a
    # common 2-word phrase like "phone number" alone isn't enough
    # evidence of real relevance to a rarer word like "neighbor".
    # Longer queries need at least 3 overlapping words.
    if not query_token_set:
        min_overlap = 0
    elif len(query_token_set) <= 3:
        min_overlap = len(query_token_set)
    else:
        min_overlap = 3

    bm25 = _bm25_data["bm25"]
    tokenized_corpus = _bm25_data["tokenized_corpus"]
    scores = bm25.get_scores(tokens)
    sorted_idx = np.argsort(scores)[::-1]

    top_idx = []
    for i in sorted_idx[: k * 3]:
        if scores[i] <= 0:
            continue
        overlap = len(query_token_set & set(tokenized_corpus[i]))
        if overlap >= min_overlap:
            top_idx.append(i)
        if len(top_idx) >= k:
            break

    elapsed = (time.perf_counter() - t0) * 1000
    rows = [int(i) for i in top_idx]
    scs = [float(scores[i]) for i in top_idx]
    return rows, scs, elapsed


def rrf_fuse(dense_rows, dense_scores, sparse_rows, k=None, top_k=None) -> list[dict]:
    k = k or settings.rrf_k
    top_k = top_k or settings.fusion_top_k

    dense_score_map = {row: score for row, score in zip(dense_rows, dense_scores)}

    fused: dict[int, dict] = {}
    for rank, row in enumerate(dense_rows, start=1):
        fused.setdefault(row, {"dense_rank": None, "sparse_rank": None, "rrf_score": 0.0, "dense_score": None})
        fused[row]["dense_rank"] = rank
        fused[row]["dense_score"] = dense_score_map.get(row)
        fused[row]["rrf_score"] += 1.0 / (k + rank)
    for rank, row in enumerate(sparse_rows, start=1):
        fused.setdefault(row, {"dense_rank": None, "sparse_rank": None, "rrf_score": 0.0, "dense_score": None})
        fused[row]["sparse_rank"] = rank
        fused[row]["rrf_score"] += 1.0 / (k + rank)

    ranked = sorted(fused.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[:top_k]
    return [{"row": row, **info} for row, info in ranked]


def hybrid_retrieve(query: str) -> dict:
    _lazy_load()

    dense_rows, dense_scores, dense_ms = dense_search(query, settings.vector_top_k)
    sparse_rows, sparse_scores, sparse_ms = sparse_search(query, settings.bm25_top_k)

    t0 = time.perf_counter()
    fused = rrf_fuse(dense_rows, dense_scores, sparse_rows)
    fusion_ms = (time.perf_counter() - t0) * 1000

    results = []
    for item in fused:
        row = _meta_df.iloc[item["row"]]
        results.append({
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "text": row["text"],
            "language": row["language"],
            "strategy": row["strategy"],
            "is_selected_gt": row.get("is_selected_gt"),
            "dense_rank": item["dense_rank"],
            "sparse_rank": item["sparse_rank"],
            "rrf_score": item["rrf_score"],
            "dense_score": item["dense_score"],
        })

    return {
        "results": results,
        "latency": {
            "query_embed_and_dense_ms": dense_ms,
            "sparse_ms": sparse_ms,
            "fusion_ms": fusion_ms,
        },
    }


def passes_relevance_gate(result: dict) -> bool:
    """
    Decide whether a fused retrieval result is relevant enough to
    ground an answer on, vs. being abstained on as off-topic/noise.

    Two independent paths to "pass":
      1. Real keyword overlap (sparse_rank is not None) AND dense_score
         clears a modest floor.
      2. Dense score alone clears a high bypass threshold - catches
         true paraphrases sharing no vocabulary with the source.
    """
    dense_score = result.get("dense_score")
    has_sparse_overlap = result.get("sparse_rank") is not None

    if dense_score is None:
        return has_sparse_overlap

    if has_sparse_overlap and dense_score >= settings.sparse_overlap_dense_floor:
        return True

    if dense_score >= settings.dense_bypass_score:
        return True

    return False


def hybrid_retrieve_gated(query: str) -> dict:
    """Same as hybrid_retrieve, but filters to only gate-passing results."""
    out = hybrid_retrieve(query)
    out["results"] = [r for r in out["results"] if passes_relevance_gate(r)]
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        with open(sys.argv[2], "r", encoding="utf-8-sig") as f:
            q = f.read().strip()
    elif len(sys.argv) > 1:
        q = sys.argv[1]
    else:
        q = "मैनहट्टन परियोजना की सफलता का प्रभाव क्या था?"
    out = hybrid_retrieve(q)
    print(f"\nQuery: {q}\n")
    for i, r in enumerate(out["results"]):
        print(f"#{i+1} rrf={r['rrf_score']:.4f} dense_score={r['dense_score']} "
              f"dense_rank={r['dense_rank']} sparse_rank={r['sparse_rank']} gt={r['is_selected_gt']}")
        print(f"    {r['text'][:120]}...")
    print(f"\nLatency: {out['latency']}")
