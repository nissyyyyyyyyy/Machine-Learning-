"""
model.py
--------
Owns the machine learning model itself: building it, training it,
evaluating it, and persisting it to disk so you don't have to retrain
every single time you run the program.

Model choice: RandomForestClassifier.
  - Handles multi-class targets (wine quality scores) well out of the box.
  - Gives us feature_importances_ for free (needed for menu option 7).
  - Robust to unscaled/mixed-range features and doesn't need much tuning
    to get a solid baseline - a good fit for a course project.
"""

import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config
from src.preprocessing import PreparedData


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    report: str
    class_names: List[str]


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def train_model(prepared: PreparedData) -> RandomForestClassifier:
    model = build_model()
    model.fit(prepared.X_train, prepared.y_train)
    return model


def evaluate_model(model: RandomForestClassifier, prepared: PreparedData) -> EvaluationResult:
    y_pred = model.predict(prepared.X_test)

    return EvaluationResult(
        accuracy=accuracy_score(prepared.y_test, y_pred),
        precision=precision_score(prepared.y_test, y_pred, average="weighted", zero_division=0),
        recall=recall_score(prepared.y_test, y_pred, average="weighted", zero_division=0),
        f1=f1_score(prepared.y_test, y_pred, average="weighted", zero_division=0),
        confusion=confusion_matrix(prepared.y_test, y_pred),
        report=classification_report(
            prepared.y_test, y_pred,
            target_names=prepared.class_names,
            zero_division=0,
        ),
        class_names=prepared.class_names,
    )


def get_feature_importance(model: RandomForestClassifier, feature_names: List[str]):
    """Returns a list of (feature_name, importance) sorted high to low."""
    importances = model.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Persistence: save/load the trained model + the preprocessing objects it
# depends on (scaler, label encoder, feature/target column names). Bundling
# all of these together avoids a common bug where a saved model is loaded
# back but paired with a mismatched, freshly-fit scaler.
# ---------------------------------------------------------------------------

def save_bundle(model, prepared: PreparedData, evaluation: EvaluationResult, path: str = None) -> str:
    path = path or config.MODEL_PATH
    bundle = {
        "model": model,
        "scaler": prepared.scaler,
        "label_encoder": prepared.label_encoder,
        "feature_columns": prepared.feature_columns,
        "target_column": prepared.target_column,
        "class_names": prepared.class_names,
        "evaluation": evaluation,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


def load_bundle(path: str = None) -> dict:
    path = path or config.MODEL_PATH
    with open(path, "rb") as f:
        return pickle.load(f)
