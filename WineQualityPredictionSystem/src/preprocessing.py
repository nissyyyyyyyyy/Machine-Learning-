"""
preprocessing.py
------------------
Turns a raw DataFrame into clean, model-ready arrays.

Responsibilities:
  - drop duplicate rows / handle missing values
  - encode the target column (works whether "quality" is numeric like 3-8,
    or text like "low"/"medium"/"high")
  - split into train/test sets (stratified, so class balance is preserved)
  - scale numeric features (RandomForest doesn't strictly need this, but
    scaling is kept because it makes the pipeline reusable if you swap in a
    distance-based model like KNN or Logistic Regression later)
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config, data_loader


@dataclass
class PreparedData:
    """Bundles everything downstream modules need, so we pass ONE object
    around instead of five loose variables."""
    df: pd.DataFrame
    target_column: str
    feature_columns: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    label_encoder: LabelEncoder
    class_names: List[str] = field(default_factory=list)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic, dataset-agnostic cleaning."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    # Drop rows where every value is missing, then fill any remaining
    # missing numeric values with the column median (robust to outliers).
    df = df.dropna(how="all")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if removed:
        print(f"[preprocessing] Removed {removed} duplicate row(s).")
    return df


def prepare_dataset(path: str = None) -> PreparedData:
    """
    Full pipeline: load -> clean -> detect target -> encode -> split -> scale.
    Returns a PreparedData object used by the model, predictor, and stats
    modules.
    """
    raw_df = data_loader.load_raw_dataframe(path)
    df = clean_dataframe(raw_df)

    target_column = data_loader.detect_target_column(df)
    feature_columns = data_loader.get_feature_columns(df, target_column)

    if not feature_columns:
        raise ValueError(
            "No numeric feature columns were found. Check that your CSV "
            "contains the physicochemical measurement columns."
        )

    X = df[feature_columns].values
    y_raw = df[target_column]

    # LabelEncoder works for both numeric quality scores (3,4,5...) and
    # text labels ("low","medium","high") - it just maps each unique value
    # to an integer class 0..N-1, which is what sklearn classifiers expect.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = [str(c) for c in label_encoder.classes_]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return PreparedData(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        label_encoder=label_encoder,
        class_names=class_names,
    )
