"""
validator.py
Validates that the uploaded Excel DataFrame has all required columns.
"""

import pandas as pd
from typing import Optional

REQUIRED_COLUMNS = ["Name", "Role", "Location", "Experience", "Skills"]


def validate_schema(df: pd.DataFrame) -> Optional[str]:
    """
    Validate that the DataFrame has all required columns (case-insensitive).

    Returns:
        None if valid.
        Error message string if invalid.
    """
    if df is None or df.empty:
        return "Uploaded Excel file is empty or could not be read."

    # Normalize uploaded column names for comparison
    uploaded_cols_normalized = {col.strip().lower(): col for col in df.columns}
    required_normalized = {col.lower(): col for col in REQUIRED_COLUMNS}

    missing = []
    for req_lower, req_original in required_normalized.items():
        if req_lower not in uploaded_cols_normalized:
            missing.append(req_original)

    if missing:
        return f"Missing required columns: {', '.join(missing)}. Required: {', '.join(REQUIRED_COLUMNS)}"

    # Rename columns to standard casing (important for downstream processing)
    rename_map = {}
    for req_lower, req_original in required_normalized.items():
        actual_col = uploaded_cols_normalized[req_lower]
        if actual_col != req_original:
            rename_map[actual_col] = req_original

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    return None  # Valid
