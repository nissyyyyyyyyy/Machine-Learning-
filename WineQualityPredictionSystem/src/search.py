"""
search.py
---------
Lets the user filter the dataset by a chosen quality value/label
(menu option 9). Kept tiny and separate since it's a distinct, reusable
piece of behaviour.
"""

import pandas as pd


def search_by_quality(df: pd.DataFrame, target_column: str, value: str, max_rows: int = 20) -> pd.DataFrame:
    """
    Returns rows whose target column matches `value`. Comparison is done
    as strings so it works whether the target is numeric (e.g. 6) or text
    (e.g. "high").
    """
    mask = df[target_column].astype(str).str.strip() == str(value).strip()
    matches = df[mask]
    return matches.head(max_rows)
