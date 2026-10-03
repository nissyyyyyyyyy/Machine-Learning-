"""
statistics_module.py
----------------------
Descriptive statistics about the raw dataset: shape, per-feature summary
stats (mean/std/min/max/quartiles), and class balance. Named
"statistics_module" (not "statistics") to avoid shadowing Python's own
built-in `statistics` standard-library module.
"""

import pandas as pd


def basic_info(df: pd.DataFrame, target_column: str) -> dict:
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": target_column,
        "unique_classes": df[target_column].nunique(),
    }


def numeric_summary(df: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """Returns a DataFrame: rows=features, columns=mean/std/min/25%/50%/75%/max."""
    return df[feature_columns].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]


def class_balance(df: pd.DataFrame, target_column: str) -> pd.Series:
    return df[target_column].value_counts().sort_index()


def correlation_with_target(df: pd.DataFrame, feature_columns, target_column: str) -> pd.Series:
    """
    Correlation of each numeric feature with the target. If the target is
    non-numeric (text labels), it's label-encoded on the fly just for this
    correlation check so the calculation still works.
    """
    working = df.copy()
    if not pd.api.types.is_numeric_dtype(working[target_column]):
        working[target_column] = working[target_column].astype("category").cat.codes

    corr = working[feature_columns + [target_column]].corr()[target_column]
    corr = corr.drop(labels=[target_column]).sort_values(ascending=False)
    return corr
