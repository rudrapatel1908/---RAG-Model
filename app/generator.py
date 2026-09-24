"""
Adapter exposing our real grounding decision to rag-local-eval-loop.

IMPORTANT, stated plainly rather than hidden: our production grounding
gate (backend/services/grounding.py) is two-sided - it passes if EITHER
a real BM25 keyword hit exists OR the dense score clears
settings.dense_bypass_score (0.854, centered against real measured
false-positive/true-paraphrase scores - see backend/config.py's
comments and README.md). This eval loop builds its OWN throwaway,
dense-only index over freshly re-chunked MSMARCO-XI candidates (see
that tool's eval/index_build.py - plain IndexHNSWFlat, no BM25/sparse
component at all). There is no real "has_sparse_hit" signal available
here, because this suite doesn't build one.

Rather than silently pretend that signal exists, this adapter computes
its own lexical-overlap check using our real tokenizer
(retrieval/tokenizer.py) as a genuine stand-in for it, then applies the
same dense_bypass_score constant for the dense-only path - same spirit
as production, not a literal reuse of backend/services/grounding.py
(whose function signature expects inputs - has_sparse_hit specifically
- this suite's search results don't carry).

Because this suite's chunking (fixed 400-char windows with 60-char
overlap, see eval/index_build.py) differs from our production chunking
strategies, and the candidate pool differs too, exact scores will land
somewhat differently than the 0.854 figure measured against our own
indexes. That's expected drift from evaluating against a different
index, not a bug - noted here so it isn't mistaken for one.

Answer text: for a grounded result, this returns the top retrieved
chunk's text directly, not a re-run of retrieval/extractive.py's exact
best-supported-span extraction - that module expects our own retrieval
result shape (dense_score, sparse_rank, etc.), which this suite's
duck-typed context objects (.text / .source only) don't provide. Using
the top chunk directly is a reasonable, honest simplification, not
identical to the production extraction path.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from tokenizer import tokenize  # noqa: E402
from config import settings  # noqa: E402

ABSTAIN_TEXT = "I don't have enough information in the provided corpus to answer that."


@dataclass
class _Answer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def _lexical_overlap(query: str, text: str) -> bool:
    """Stand-in for our production sparse/BM25 hit signal - see this
    module's docstring for why. min_overlap logic mirrors
    retrieval/hybrid_search.py's sparse_search() short-query handling."""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return False
    t_tokens = set(tokenize(text))
    overlap = len(q_tokens & t_tokens)
    min_overlap = min(2, len(q_tokens))
    return overlap >= min_overlap


def generate_answer(query: str, results):
    t0 = time.perf_counter()

    if not results:
        return _Answer(
            text=ABSTAIN_TEXT,
            grounded=False,
            generation_ms=(time.perf_counter() - t0) * 1000,
            model="e5-small-extractive",
        )

    top = results[0]
    # .score is present on this suite's own _Context dataclass but is
    # NOT part of the required duck-typed interface (only .text/.source
    # are guaranteed per eval/target.py's docstring) - accessed
    # defensively, never assumed to exist.
    top_score = getattr(top, "score", None)

    has_lexical_hit = _lexical_overlap(query, top.text)
    dense_clears_bypass = top_score is not None and top_score >= settings.dense_bypass_score
    grounded = has_lexical_hit or dense_clears_bypass

    answer_text = top.text if grounded else ABSTAIN_TEXT

    return _Answer(
        text=answer_text,
        grounded=grounded,
        generation_ms=(time.perf_counter() - t0) * 1000,
        model="e5-small-extractive",
    )
