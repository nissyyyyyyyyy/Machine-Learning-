"""
predictor.py
-------------
Everything involved in turning a set of wine features into a predicted
quality: manual entry, picking a row from the dataset, picking a random
row, and batch-testing several random rows at once. Kept separate from
menu.py so the prediction logic can be unit-tested without any print()/
input() calls getting in the way (each function returns data; menu.py
decides how to display it).
"""

import random
from typing import List, Tuple

import numpy as np
import pandas as pd


def predict_features(model, scaler, feature_values: List[float], label_encoder):
    """
    Runs a single feature vector through the trained model.
    Returns (predicted_label, confidence, class_probabilities_dict).
    """
    X = np.array(feature_values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    probabilities = model.predict_proba(X_scaled)[0]
    class_names = label_encoder.inverse_transform(model.classes_)
    prob_dict = {str(c): float(p) for c, p in zip(class_names, probabilities)}
    confidence = float(max(probabilities))

    return predicted_label, confidence, prob_dict


def prompt_manual_features(feature_columns: List[str], df: pd.DataFrame) -> List[float]:
    """
    Interactively asks the user for a value for each feature column.
    Shows the dataset's min/max for that column as a helpful hint and
    keeps re-asking until a valid number is entered.
    """
    values = []
    for col in feature_columns:
        col_min, col_max = df[col].min(), df[col].max()
        while True:
            raw = input(f"  {col} (typical range {col_min:.3f} - {col_max:.3f}): ").strip()
            try:
                values.append(float(raw))
                break
            except ValueError:
                print("    Please enter a numeric value.")
    return values


def get_row_by_index(df: pd.DataFrame, feature_columns: List[str], target_column: str, index: int):
    """
    Returns (feature_values, actual_label) for a specific dataset row index.

    NOTE: we deliberately do NOT do `df.iloc[index]` and then pull values
    out of that one-row Series. A pandas row Series must have a single
    dtype, so if the DataFrame mixes int and float columns (e.g. an int
    "quality" column alongside float physicochemical columns), pulling a
    whole row silently upcasts everything to float - turning quality 5
    into 5.0 and breaking exact-match comparisons later. Reading each
    column directly avoids that trap.
    """
    feature_values = [float(df[c].iloc[index]) for c in feature_columns]
    actual_label = df[target_column].iloc[index]
    return feature_values, actual_label


def get_random_row(df: pd.DataFrame, feature_columns: List[str], target_column: str):
    """Returns (row_index, feature_values, actual_label) for a random dataset row."""
    idx = random.randint(0, len(df) - 1)
    feature_values, actual_label = get_row_by_index(df, feature_columns, target_column, idx)
    return idx, feature_values, actual_label


def batch_test_random_samples(model, scaler, df: pd.DataFrame, feature_columns: List[str],
                               target_column: str, label_encoder, n: int) -> Tuple[List[dict], float]:
    """
    Picks n random rows, predicts each, and reports whether the prediction
    matched the actual label. Returns (list_of_result_dicts, accuracy_pct).
    """
    n = min(n, len(df))
    indices = random.sample(range(len(df)), n)
    results = []
    correct = 0

    for idx in indices:
        feature_values, actual_label = get_row_by_index(df, feature_columns, target_column, idx)
        predicted_label, confidence, _ = predict_features(model, scaler, feature_values, label_encoder)
        is_correct = str(predicted_label) == str(actual_label)
        correct += int(is_correct)
        results.append({
            "index": idx,
            "actual": actual_label,
            "predicted": predicted_label,
            "confidence": confidence,
            "correct": is_correct,
        })

    accuracy_pct = (correct / n * 100) if n else 0.0
    return results, accuracy_pct
