"""
loader.py
Loads Excel file bytes into a Pandas DataFrame.
"""

import io
import pandas as pd
from typing import Tuple, Optional


def load_excel(file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load Excel file bytes into a Pandas DataFrame.

    Returns:
        (DataFrame, None) on success.
        (None, error_message) on failure.
    """
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

        # Drop fully empty rows
        df.dropna(how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Strip whitespace from string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        return df, None

    except Exception as e:
        return None, f"Failed to read Excel file: {str(e)}"
