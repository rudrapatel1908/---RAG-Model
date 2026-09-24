"""
Optional config, read defensively by rag-local-eval-loop's
eval/target.py::optional_config() - see that module's docstring. Nothing
here is required; if this file didn't exist the suite would just fall
back to its own defaults.
"""

# We hold one shared embedding model in-process (see app/embedder.py,
# which reuses retrieval/hybrid_search.py's real _lazy_load()). That's
# the exact "local" convention eval/pipeline.py's docstring describes -
# setting this clamps the suite to --workers 1 automatically, avoiding
# concurrent threads contending for the same in-process model.
GENERATION_BACKEND = "local"

# Our real, measured retrieval budget - see README.md's "Verified
# results" table (offline P50/P70/P100 = 43/47/66ms, 49/49 under this).
LATENCY_BUDGET_MS = 200

# Cosmetic label only, shown in the suite's report - not read anywhere
# in our own code.
GENERATION_MODEL = "e5-small-extractive"
