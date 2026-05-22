"""
response_formatter.py
Formats the final API response as a clean, validated JSON structure.
"""

from typing import List, Dict, Any, Optional
import math


def format_response(
    results: List[Dict],
    query: str,
    mode: str,
    parsed_filters: Dict[str, Any],
    total_rows: int
) -> Dict[str, Any]:
    """
    Format search results into a standardized API response.

    Returns:
    {
        "query": "...",
        "mode": "...",
        "filters_applied": {...},
        "total_in_file": N,
        "results_count": M,
        "results": [...]
    }
    """
    # Clean results: remove internal keys, handle NaN
    cleaned = [_clean_record(r) for r in results]

    # Build filters summary
    filters_applied = _summarize_filters(parsed_filters)

    return {
        "query": query,
        "mode": mode,
        "filters_applied": filters_applied,
        "total_in_file": total_rows,
        "results_count": len(cleaned),
        "results": cleaned
    }


def _clean_record(record: Dict) -> Dict:
    """Remove internal keys and sanitize values for JSON serialization."""
    cleaned = {}
    for k, v in record.items():
        if k.startswith("_") and k != "_similarity_score":
            continue  # skip internal fields except score
        cleaned[k] = _sanitize_value(v)
    return cleaned


def _sanitize_value(v: Any) -> Any:
    """Convert NaN, Inf, and other non-JSON-safe values."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    if isinstance(v, (int, str, bool, list, dict)):
        return v
    return str(v)  # fallback


def _summarize_filters(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the applied (non-None) filter fields for display."""
    filter_keys = ["name", "role", "location", "experience_min", "experience_max", "experience_level", "skills"]
    summary = {}
    for k in filter_keys:
        v = parsed.get(k)
        if v is not None and v != []:  # Also skip empty skills list
            summary[k] = v
    return summary
