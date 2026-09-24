"""
Adapter exposing our real embedding model to rag-local-eval-loop (see
that tool's eval/target.py for the required interface). Reuses the
exact production model loader (retrieval/hybrid_search.py's
_lazy_load()) rather than a separate copy, so this suite scores the
same model actually deployed - not a stand-in.

e5 prefix convention: intfloat/multilingual-e5-small requires "query: "
on queries and "passage: " on documents to retrieve correctly (this is
the model's own documented behavior). Confirmed from the eval loop's
real source (not assumed): eval/pipeline.py calls embed_one() only on
query text (ex.query_en / ex.query_hi), and eval/index_build.py calls
embed() only on candidate passage text. So embed_one() applies
"query: " and embed() applies "passage: " below - getting these
backwards wouldn't error, it would just silently degrade retrieval
scores, so this mapping was checked against the real call sites, not
guessed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import hybrid_search  # noqa: E402


def get_model():
    """Side effect only, per the suite's interface contract (its return
    value is unused) - loads and caches the real production model."""
    hybrid_search._lazy_load()
    return hybrid_search._model


def embed_one(text: str) -> np.ndarray:
    """Called on QUERY text (eval/pipeline.py::_search). Must return a
    numpy array supporting .reshape(1, -1) per eval/target.py's
    interface contract."""
    model = get_model()
    vec = model.encode(
        ["query: " + text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype("float32")
    return vec


def embed(texts: list) -> np.ndarray:
    """Called on PASSAGE/candidate text (eval/index_build.py::build_index)
    - these are documents being indexed, not queries."""
    model = get_model()
    prefixed = ["passage: " + t for t in texts]
    vecs = model.encode(
        prefixed,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return vecs
