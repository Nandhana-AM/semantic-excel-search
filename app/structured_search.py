"""
structured_search.py
Filter-based search using Pandas.
Handles role, location, experience (numeric + level) filters.
"""

import pandas as pd
from typing import Dict, Any, List

EXPERIENCE_LEVEL_MAP = {
    "fresher":     (0, 1),
    "entry-level": (0, 2),
    "entrylevel":  (0, 2),
    "junior":      (1, 3),
    "mid-level":   (3, 6),
    "midlevel":    (3, 6),
    "senior":      (6, 15),
    "lead":        (8, 20),
    "principal":   (10, 30),
}


def structured_search(df: pd.DataFrame, parsed: Dict[str, Any]) -> List[Dict]:
    """
    Apply structured filters based on parsed query.

    Filters applied (all are AND conditions):
    - Role (case-insensitive contains)
    - Location (case-insensitive contains)
    - Experience (numeric range or keyword level)
    """
    filtered = df.copy()

    # ── Role filter ───────────────────────────────────────────────────────────
    if parsed.get("role"):
        role_query = parsed["role"].lower()
        filtered = filtered[
            filtered["Role"].str.lower().str.contains(role_query, na=False)
        ]

    # ── Location filter ───────────────────────────────────────────────────────
    if parsed.get("location"):
        loc_query = parsed["location"].lower()
        filtered = filtered[
            filtered["Location"].str.lower().str.contains(loc_query, na=False)
        ]

    # ── Experience filter ─────────────────────────────────────────────────────
    # First, parse Experience column to numeric
    filtered = _parse_experience_column(filtered)

    if parsed.get("experience_level"):
        level = parsed["experience_level"].replace(" ", "").lower()
        level_key = None
        for k in EXPERIENCE_LEVEL_MAP:
            if level in k or k in level:
                level_key = k
                break
        if level_key:
            min_exp, max_exp = EXPERIENCE_LEVEL_MAP[level_key]
            filtered = filtered[
                (filtered["Experience_Numeric"] >= min_exp) &
                (filtered["Experience_Numeric"] <= max_exp)
            ]

    elif parsed.get("experience_min") is not None:
        min_exp = parsed["experience_min"]
        filtered = filtered[filtered["Experience_Numeric"] >= min_exp]

        if parsed.get("experience_max") is not None:
            max_exp = parsed["experience_max"]
            filtered = filtered[filtered["Experience_Numeric"] <= max_exp]

    # ── Cleanup temp column ───────────────────────────────────────────────────
    if "Experience_Numeric" in filtered.columns:
        filtered.drop(columns=["Experience_Numeric"], inplace=True)

    return filtered.to_dict(orient="records")


def _parse_experience_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Experience column into a numeric value for filtering.
    Handles formats like: '5', '5 years', '3-5 years', '5+', 'Senior'
    """
    import re

    def extract_years(val):
        if pd.isna(val):
            return 0
        val = str(val).lower().strip()

        # Range: "3-5" → take lower bound
        range_match = re.search(r"(\d+)\s*-\s*(\d+)", val)
        if range_match:
            return int(range_match.group(1))

        # Single number: "5", "5+", "5 years"
        num_match = re.search(r"(\d+)", val)
        if num_match:
            return int(num_match.group(1))

        return 0

    df = df.copy()
    df["Experience_Numeric"] = df["Experience"].apply(extract_years)
    return df
