"""
menu.py
-------
The CLI itself. This module is intentionally "thin": it prints things and
reads input, but delegates every real calculation to the other src/
modules (data_loader, preprocessing, model, predictor, visualization,
statistics_module, search). That separation is what makes the project easy
to extend, debug, and grade - each module can be understood on its own.

Run this indirectly via bin/run.py, e.g.:
    python bin/run.py
"""

import sys

from src import config, data_loader, ui
from src import model as model_module
from src import predictor, preprocessing, search, statistics_module, visualization

MENU_TEXT = """
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
"""


class WineQualityApp:
    """Holds the trained model + prepared data for the life of the session,
    so we only load the CSV and train the model ONCE at startup, and every
    menu option after that is instant."""

    def __init__(self):
        self.prepared: preprocessing.PreparedData = None
        self.model = None
        self.evaluation: model_module.EvaluationResult = None

    # -- startup ------------------------------------------------------
    def bootstrap(self) -> bool:
        ui.banner()
        ui.section("Startup")
        try:
            dataset_path = data_loader.find_dataset_path()
            ui.info(f"Using dataset: {dataset_path}")

            ui.info("Cleaning data, encoding labels, splitting train/test...")
            self.prepared = preprocessing.prepare_dataset(dataset_path)

            ui.info(f"Training RandomForestClassifier "
                     f"({config.N_ESTIMATORS} trees) ...")
            self.model = model_module.train_model(self.prepared)

            self.evaluation = model_module.evaluate_model(self.model, self.prepared)
            saved_path = model_module.save_bundle(self.model, self.prepared, self.evaluation)

            ui.success(f"Model trained. Test accuracy: {self.evaluation.accuracy * 100:.2f}%")
            ui.info(f"Model saved to: {saved_path}")
            ui.key_value("Rows loaded:", len(self.prepared.df))
            ui.key_value("Features used:", len(self.prepared.feature_columns))
            ui.key_value("Target column:", self.prepared.target_column)
            ui.key_value("Classes:", ", ".join(self.prepared.class_names))
            return True

        except data_loader.DatasetNotFoundError as e:
            ui.error(str(e))
            return False
        except Exception as e:  # noqa: BLE001 - top-level safety net at startup
            ui.error(f"Failed to start the application: {e}")
            return False

    # -- main loop ------------------------------------------------------
    def run(self) -> None:
        if not self.bootstrap():
            sys.exit(1)

        actions = {
            "1": self.enter_manual_values,
            "2": self.select_sample,
            "3": self.random_sample,
            "4": self.view_distribution,
            "5": self.test_with_samples,
            "6": self.model_performance,
            "7": self.feature_importance,
            "8": self.dataset_statistics,
            "9": self.search_by_quality,
        }

        while True:
            ui.banner()
            print(MENU_TEXT)
            choice = input("Enter your choice: ").strip()

            if choice == "10":
                ui.success("Thank you for using the Wine Quality Prediction System. Goodbye!")
                break

            action = actions.get(choice)
            if action is None:
                ui.error("Invalid choice. Please enter a number from 1 to 10.")
                ui.pause()
                continue

            try:
                action()
            except Exception as e:  # noqa: BLE001 - keep the menu alive on any error
                ui.error(f"Something went wrong: {e}")
                ui.pause()

    # -- option 1 ---------------------------------------------------------
    def enter_manual_values(self) -> None:
        ui.section("Enter Wine Values Manually")
        p = self.prepared
        ui.info("Enter a value for each feature below.\n")

        feature_values = predictor.prompt_manual_features(p.feature_columns, p.df)
        predicted_label, confidence, probabilities = predictor.predict_features(
            self.model, p.scaler, feature_values, p.label_encoder
        )

        print()
        ui.highlight(f"Predicted Wine Quality: {predicted_label}  "
                     f"(confidence: {confidence * 100:.1f}%)")
        ui.print_table(
            ["Class", "Probability"],
            [[c, f"{v * 100:.1f}%"] for c, v in sorted(probabilities.items(), key=lambda x: -x[1])],
        )
        ui.pause()

    # -- option 2 ---------------------------------------------------------
    def select_sample(self) -> None:
        ui.section("Select a Sample from Dataset")
        p = self.prepared
        n_rows = len(p.df)
        ui.info(f"Dataset has {n_rows} rows (valid indices: 0 to {n_rows - 1}).")

        raw = input("Enter a row index: ").strip()
        if not raw.isdigit() or not (0 <= int(raw) < n_rows):
            ui.error("Invalid index.")
            ui.pause()
            return

        index = int(raw)
        self._predict_and_show_row(index)
        ui.pause()

    # -- option 3 ---------------------------------------------------------
    def random_sample(self) -> None:
        ui.section("Random Dataset Sample")
        p = self.prepared
        index, feature_values, actual_label = predictor.get_random_row(
            p.df, p.feature_columns, p.target_column
        )
        ui.info(f"Randomly selected row index: {index}")
        self._predict_and_show_row(index)
        ui.pause()

    def _predict_and_show_row(self, index: int) -> None:
        p = self.prepared
        feature_values, actual_label = predictor.get_row_by_index(
            p.df, p.feature_columns, p.target_column, index
        )
        predicted_label, confidence, _ = predictor.predict_features(
            self.model, p.scaler, feature_values, p.label_encoder
        )

        ui.print_table(
            ["Feature", "Value"],
            list(zip(p.feature_columns, [f"{v:.4f}" for v in feature_values])),
        )
        print()
        ui.key_value("Actual quality:", actual_label)
        ui.key_value("Predicted quality:", f"{predicted_label} ({confidence * 100:.1f}% confidence)")
        if str(actual_label) == str(predicted_label):
            ui.success("Prediction matches the actual label.")
        else:
            ui.warn("Prediction differs from the actual label.")

    # -- option 4 ---------------------------------------------------------
    def view_distribution(self) -> None:
        ui.section("Wine Quality Distribution")
        p = self.prepared
        counts = p.df[p.target_column].value_counts().sort_index()
        ui.print_table(["Quality", "Count"], list(counts.items()))

        path = visualization.plot_quality_distribution(p.df, p.target_column)
        ui.success(f"Chart saved to: {path}")
        ui.pause()

    # -- option 5 ---------------------------------------------------------
    def test_with_samples(self) -> None:
        ui.section("Test Model with Dataset Samples")
        p = self.prepared

        raw = input("How many random samples to test? [default 10]: ").strip()
        n = int(raw) if raw.isdigit() and int(raw) > 0 else 10

        results, accuracy_pct = predictor.batch_test_random_samples(
            self.model, p.scaler, p.df, p.feature_columns, p.target_column, p.label_encoder, n
        )

        rows = []
        for r in results:
            mark = "MATCH" if r["correct"] else "MISS"
            rows.append([r["index"], r["actual"], r["predicted"],
                         f"{r['confidence'] * 100:.1f}%", mark])
        ui.print_table(["Index", "Actual", "Predicted", "Confidence", "Result"], rows)

        print()
        ui.highlight(f"Batch accuracy: {accuracy_pct:.2f}% ({len(results)} samples)")
        ui.pause()

    # -- option 6 ---------------------------------------------------------
    def model_performance(self) -> None:
        ui.section("Model Performance")
        ev = self.evaluation

        ui.key_value("Accuracy:", f"{ev.accuracy * 100:.2f}%")
        ui.key_value("Precision (weighted):", f"{ev.precision * 100:.2f}%")
        ui.key_value("Recall (weighted):", f"{ev.recall * 100:.2f}%")
        ui.key_value("F1 Score (weighted):", f"{ev.f1 * 100:.2f}%")
        print()
        ui.info("Classification Report:")
        print(ev.report)

        show = input("Show confusion matrix chart? (y/n): ").strip().lower()
        if show == "y":
            path = visualization.plot_confusion_matrix(ev.confusion, ev.class_names)
            ui.success(f"Chart saved to: {path}")
        ui.pause()

    # -- option 7 ---------------------------------------------------------
    def feature_importance(self) -> None:
        ui.section("Feature Importance")
        p = self.prepared
        pairs = model_module.get_feature_importance(self.model, p.feature_columns)

        ui.print_table(
            ["Feature", "Importance"],
            [[name, f"{value:.4f}"] for name, value in pairs],
        )

        path = visualization.plot_feature_importance(pairs)
        ui.success(f"Chart saved to: {path}")
        ui.pause()

    # -- option 8 ---------------------------------------------------------
    def dataset_statistics(self) -> None:
        ui.section("Dataset Statistics")
        p = self.prepared

        info = statistics_module.basic_info(p.df, p.target_column)
        for key, value in info.items():
            ui.key_value(f"{key}:", value)

        print()
        ui.info("Feature summary statistics:")
        summary = statistics_module.numeric_summary(p.df, p.feature_columns)
        rows = [[idx] + [f"{v:.3f}" for v in row] for idx, row in summary.iterrows()]
        ui.print_table(["Feature"] + list(summary.columns), rows)

        print()
        ui.info("Correlation of each feature with quality:")
        corr = statistics_module.correlation_with_target(p.df, p.feature_columns, p.target_column)
        ui.print_table(["Feature", "Correlation"], [[k, f"{v:.3f}"] for k, v in corr.items()])

        show = input("\nShow correlation heatmap chart? (y/n): ").strip().lower()
        if show == "y":
            path = visualization.plot_correlation_heatmap(p.df, p.feature_columns)
            ui.success(f"Chart saved to: {path}")
        ui.pause()

    # -- option 9 ---------------------------------------------------------
    def search_by_quality(self) -> None:
        ui.section("Search Dataset by Quality")
        p = self.prepared

        available = sorted(p.df[p.target_column].astype(str).unique())
        ui.info(f"Available quality values: {', '.join(available)}")

        value = input("Enter a quality value to search for: ").strip()
        matches = search.search_by_quality(p.df, p.target_column, value)

        if matches.empty:
            ui.warn(f"No rows found with quality = '{value}'.")
        else:
            ui.success(f"Found {matches.shape[0]} matching row(s) (showing up to 20):")
            # Pull each column as its own Series (not via .iterrows() /
            # .values, which force a single dtype across the whole row and
            # would upcast an integer quality column to "5.0" alongside
            # the float feature columns) so quality displays as "5", not
            # "5.0".
            display_columns = p.feature_columns + [p.target_column]
            feature_value_lists = [matches[c].tolist() for c in p.feature_columns]
            target_values = matches[p.target_column].tolist()

            rows = []
            for i in range(len(matches)):
                formatted = [f"{feature_value_lists[j][i]:.3f}" for j in range(len(p.feature_columns))]
                formatted.append(target_values[i])
                rows.append(formatted)
            ui.print_table(display_columns, rows)
        ui.pause()
