"""
data_loader.py
---------------
Everything related to FINDING and READING the dataset lives here.
No cleaning, no modeling - just: locate the CSV, load it into a DataFrame,
and figure out which column is the prediction target.

This is deliberately defensive/flexible: Kaggle datasets vary slightly in
column naming (e.g. "quality" vs "quality_label"), so instead of hard-coding
a column name we search for the most likely candidate. That way, if you
swap in a different wine-quality CSV, the rest of the program still works.
"""

import glob
import os

import pandas as pd

from src import config


class DatasetNotFoundError(Exception):
    """Raised when no usable CSV file can be found in the data/ folder."""


def find_dataset_path() -> str:
    """
    Looks for the dataset in the data/ folder.

    Order of preference:
      1. The exact file name configured in config.DEFAULT_DATASET_FILENAME
      2. If that's missing, the first .csv file found in data/
    """
    preferred = os.path.join(config.DATA_DIR, config.DEFAULT_DATASET_FILENAME)
    if os.path.isfile(preferred):
        return preferred

    csv_files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.csv")))
    if csv_files:
        return csv_files[0]

    raise DatasetNotFoundError(
        f"No CSV file found in '{config.DATA_DIR}'.\n"
        f"Download the dataset from Kaggle "
        f"(taweilo/wine-quality-dataset-balanced-classification), "
        f"and place the .csv file inside the 'data' folder.\n"
        f"Either name it '{config.DEFAULT_DATASET_FILENAME}' or leave it as "
        f"the only .csv file in that folder."
    )


def load_raw_dataframe(path: str = None) -> pd.DataFrame:
    """Loads the CSV as-is (no cleaning) and returns a pandas DataFrame."""
    if path is None:
        path = find_dataset_path()
    df = pd.read_csv(path)
    # Normalize column names: strip whitespace so "quality " == "quality"
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Figures out which column holds the wine quality label/score.

    Strategy: check config.TARGET_COLUMN_CANDIDATES (case-insensitive) in
    order. If none match, fall back to any column whose name contains
    "quality". If that also fails, use the last column and warn the caller
    via the returned column name being clearly wrong is avoided by raising.
    """
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in config.TARGET_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]

    for lower_name, original_name in lower_map.items():
        if "quality" in lower_name:
            return original_name

    raise ValueError(
        "Could not automatically detect the target column (e.g. 'quality'). "
        f"Available columns are: {list(df.columns)}. "
        "Rename your label column to 'quality' or add it to "
        "config.TARGET_COLUMN_CANDIDATES."
    )


def get_feature_columns(df: pd.DataFrame, target_column: str) -> list:
    """
    Returns the list of columns that should be used as model features:
    every numeric column except the target and any obvious ID column.
    """
    feature_cols = []
    for col in df.columns:
        if col == target_column:
            continue
        if col.strip().lower() in config.ID_COLUMN_NAMES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols
