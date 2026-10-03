# Wine Quality Prediction System

A menu-driven Python CLI that trains a Random Forest classifier on the
[Wine Quality (Balanced Classification)](https://www.kaggle.com/datasets/taweilo/wine-quality-dataset-balanced-classification)
Kaggle dataset and lets you explore, predict, and evaluate it interactively.

```
============================================================
          WINE QUALITY PREDICTION SYSTEM
============================================================

1. Enter Wine Values Manually
2. Select a Sample from Dataset
3. Random Dataset Sample
4. View Wine Quality Distribution
5. Test Model with Dataset Samples
6. Model Performance
7. Feature Importance
8. Dataset Statistics
9. Search Dataset by Quality
10. Exit
```

## Project structure

```
WineQualityPredictionSystem/
├── bin/
│   └── run.py              <- entry point: run THIS file
├── src/
│   ├── config.py            <- all paths & tunable constants in one place
│   ├── ui.py                 <- colored terminal output / table printer
│   ├── data_loader.py        <- finds & loads the CSV, detects target column
│   ├── preprocessing.py      <- cleaning, encoding, train/test split, scaling
│   ├── model.py               <- builds/trains/evaluates/saves the RandomForest
│   ├── predictor.py           <- manual / sample / random / batch predictions
│   ├── visualization.py       <- every matplotlib chart (saved to plots/)
│   ├── statistics_module.py   <- dataset summary statistics
│   ├── search.py               <- search dataset rows by quality value
│   └── menu.py                 <- the interactive menu, wires everything together
├── data/                     <- put the Kaggle CSV here (see data/PUT_DATASET_HERE.md)
├── models/                   <- trained model is auto-saved here each run (.pkl)
├── plots/                    <- charts are auto-saved here as .png
├── requirements.txt
└── README.md
```

**Why this structure?** Each module has exactly one job (single-responsibility
principle): `data_loader` never touches the model, `model` never prints
anything, `menu` never computes anything itself — it just calls the other
modules and displays results. This is worth pointing out in your project
write-up/viva; it's a big part of what separates a "script" from a
"system".

## Setup (VSCode)

1. **Get the dataset.** Download the CSV from the Kaggle link above and
   place it inside the `data/` folder (see `data/PUT_DATASET_HERE.md`).
   No renaming needed — just make sure it's the only `.csv` in that folder.

2. **Open the project folder in VSCode** (`File > Open Folder...`).

3. **Create a virtual environment (recommended)** — open a terminal in
   VSCode (`` Ctrl+` ``) and run:

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run it:**

   ```bash
   python bin/run.py
   ```

   or, in VSCode, open `bin/run.py` and click the ▶ **Run** button
   (top-right), or press `F5`.

On startup the program loads the CSV, cleans it, trains the Random Forest,
prints the test accuracy, and saves the trained model to `models/` —
*then* it shows you the menu above.

## What each menu option does

| # | Option | What happens |
|---|--------|--------------|
| 1 | Enter Wine Values Manually | Prompts you for every feature (shows the dataset's typical range as a hint), then predicts the quality with a confidence score and full probability breakdown. |
| 2 | Select a Sample from Dataset | You type a row index; it shows that row's features, the actual quality, and the model's prediction side by side. |
| 3 | Random Dataset Sample | Same as above but picks a random row for you. |
| 4 | View Wine Quality Distribution | Prints class counts and saves/opens a bar chart (`plots/quality_distribution.png`). |
| 5 | Test Model with Dataset Samples | Predicts N random rows at once (you choose N) and reports a table plus a batch accuracy percentage. |
| 6 | Model Performance | Accuracy / precision / recall / F1 (weighted) plus the full `sklearn` classification report, with an optional confusion-matrix chart. |
| 7 | Feature Importance | Ranks features by the Random Forest's built-in importance score, as a table and a bar chart. |
| 8 | Dataset Statistics | Row/column counts, missing values, per-feature mean/std/min/max/quartiles, and each feature's correlation with quality, with an optional correlation heatmap. |
| 9 | Search Dataset by Quality | Lists the available quality values, then shows every row matching the one you pick. |
| 10 | Exit | Quits cleanly. |

## Notes for your report / presentation

- **Model:** `RandomForestClassifier` (scikit-learn), 300 trees. Chosen
  because it handles a multi-class target well, needs little tuning, and
  gives feature importances for free.
- **Preprocessing:** duplicate rows dropped, missing numeric values filled
  with the column median, features standardized with `StandardScaler`,
  target label-encoded, 80/20 stratified train/test split (`config.py`
  lets you change these in one place).
- **Extensibility:** the dataset loader auto-detects the target column
  name and feature columns, so if you download a slightly different
  version of the dataset (or swap in the red/white UCI version) the code
  still runs without edits.
- **Every chart is also saved as a PNG** in `plots/`, so you can drop them
  straight into your slides/report even if you only ran the program once.
