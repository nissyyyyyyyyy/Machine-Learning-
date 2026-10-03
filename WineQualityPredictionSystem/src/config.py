"""
config.py
---------
Central place for every path and tunable constant used across the project.
Keeping these in one module means you only edit ONE file if you rename your
dataset, move folders, or want to tune the model.
"""

import os

# ---------------------------------------------------------------------------
# Folder layout (all paths are computed relative to this file, so the project
# works no matter where you clone/copy it on disk).
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Where the trained model + preprocessing objects are cached between runs.
MODEL_PATH = os.path.join(MODELS_DIR, "wine_quality_model.pkl")

# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------
# Put your Kaggle CSV in the data/ folder. If you don't rename it, this is
# the file name the loader looks for first. If it's not found, the loader
# will automatically use whichever single .csv file it finds in data/.
DEFAULT_DATASET_FILENAME = "wine_quality_classification.csv"

# Candidate names for the target/label column, checked in this order.
TARGET_COLUMN_CANDIDATES = ["quality_label", "quality", "label", "class", "target"]

# Column names that are identifiers, not features, and should be dropped.
ID_COLUMN_NAMES = {"id", "index", "unnamed: 0"}

# ---------------------------------------------------------------------------
# Model / training settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 300
MAX_DEPTH = None  # let trees grow fully; RandomForest handles overfitting via bagging

# ---------------------------------------------------------------------------
# CLI display settings
# ---------------------------------------------------------------------------
APP_TITLE = "WINE QUALITY PREDICTION SYSTEM"
DIVIDER_WIDTH = 60
