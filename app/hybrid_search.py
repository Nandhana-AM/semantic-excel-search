"""
hybrid_search.py
Combines structured filtering + semantic search.

Strategy:
1. Apply structured filters first (narrows the dataset)
2. Run semantic search on the filtered subset
3. If structured gives too few results, fall back to semantic on full dataset
4. Merge and deduplicate results
"""

import pandas as pd
from typing import Dict, Any, List

from app.structured_search import structured_search
from app.semantic_search import semantic_search

MIN_STRUCTURED_RESULTS = 3  # If structured returns fewer, expand with semantic


def hybrid_search(df: pd.DataFrame, query: str, parsed: Dict[str, Any]) -> List[Dict]:
    """
    Hybrid search: structured filter → semantic re-rank.
    """
    if df.empty:
        return []

    # ── Step 1: Structured filter (with known skills only) ────────────────────
    # In hybrid mode, known skills are matched as hard structured filters,
    # while dynamic/unknown skills are skipped here and matched semantically.
    parsed_structured = parsed.copy()
    parsed_structured["skills"] = parsed.get("known_skills", [])

    structured_results = structured_search(df, parsed_structured)

    # ── Step 2: Semantic on filtered subset ───────────────────────────────────
    if structured_results:
        filtered_df = pd.DataFrame(structured_results)
        semantic_results = semantic_search(filtered_df, query, parsed=parsed)
    else:
        semantic_results = []

    # ── Step 3: Merge + deduplicate ───────────────────────────────────────────
    # If dynamic skills are present, candidates must match them semantically.
    # We exclude structured-only results that failed the dynamic skill semantic match.
    if parsed and parsed.get("dynamic_skills"):
        merged = semantic_results
    else:
        merged = _merge_results(structured_results, semantic_results)

    return merged


def _merge_results(structured: List[Dict], semantic: List[Dict]) -> List[Dict]:
    """
    Merge structured and semantic results.
    Semantic results ranked by similarity score come first.
    Structured-only results appended at the end (without score).
    Deduplication by Name + Role combination.
    """
    seen = set()
    final = []

    def _key(record: Dict) -> str:
        return f"{record.get('Name', '')}_{record.get('Role', '')}_{record.get('Location', '')}"

    # Semantic results first (already scored)
    for record in semantic:
        k = _key(record)
        if k not in seen:
            seen.add(k)
            final.append(record)

    # Add structured-only results (not already in semantic)
    for record in structured:
        k = _key(record)
        if k not in seen:
            seen.add(k)
            record["_similarity_score"] = None  # No semantic score
            final.append(record)

    return final
