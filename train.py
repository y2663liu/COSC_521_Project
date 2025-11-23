"""Train gesture classifiers using graph metrics and basic gesture statistics.

The script expects a CSV file with one row per trial (e.g., output of
compute_graph_metrics.R) that includes a label column (default: ``gesture``).
It performs basic feature cleaning, trains a few compact models, and reports
accuracy, macro-F1, and confusion matrices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC


SUPPORTED_GESTURES = [
    "Swipe",
    "Poke",
    "Tickle",
    "Tap",
    "Pat",
    "Press",
    "Pinch5",
    "Pinch2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        type=Path,
        default=Path("gesture_features_rewritten.csv"),
        nargs="?",
        help="Path to the aggregated feature CSV (default: train_data/gesture_features_rewritten.csv).",
    )
    parser.add_argument(
        "--label-col",
        default="gesture",
        help="Name of the label column in the feature table (default: gesture).",
    )
    parser.add_argument(
        "--id-cols",
        nargs="*",
        default=["participant"],
        help=(
            "Identifier columns to exclude from the feature set (default: config participant trial_id). "
            "Include interval_id if present."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size fraction (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def load_dataset(path: Path, label_col: str, id_cols: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Some CSV exports may accidentally include a copy of the header row as the
    # first data row (e.g., when "titles" are pasted into the sheet). Detect a
    # close match between the column names and the first row and drop it so the
    # remaining rows are valid samples.
    if not df.empty:
        cols_norm = [str(c).strip().lower() for c in df.columns]
        first_norm = [str(v).strip().lower() for v in df.iloc[0]]
        match_ratio = sum(a == b for a, b in zip(cols_norm, first_norm)) / len(cols_norm)
        if match_ratio >= 0.6:
            print(
                "Dropping first row that appears to contain column titles instead of data.",
                file=sys.stderr,
            )
            df = df.iloc[1:].reset_index(drop=True)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}")

    # Keep only known gestures if the column exists.
    labels = df[label_col].astype(str)
    filtered = df.loc[labels.isin(SUPPORTED_GESTURES)].copy()
    if filtered.empty:
        raise ValueError(
            "No samples remain after filtering to supported gestures. "
            "Check that the gesture labels in the CSV match the expected names or disable filtering."
        )

    df = filtered
    labels = labels.loc[df.index]

    drop_cols = set(id_cols) | {label_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns left after dropping identifiers; check --id-cols.")

    features = df[feature_cols]
    return features, labels


def build_pipeline(model) -> Pipeline:
    # A compact preprocessing stack: impute missing values, drop near-constant features,
    # standardize scales, then fit the model.
    preprocessing = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("var_thresh", VarianceThreshold(threshold=1e-6)),
            ("scale", StandardScaler()),
        ]
    )

    return Pipeline(steps=[("pre", preprocessing), ("model", model)])


def train_and_eval(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Dict[str, Dict[str, object]]:
    if X.empty or y.empty:
        raise ValueError(
            "No samples available for training. Ensure the CSV has data and the label column matches --label-col."
        )

    class_counts = y.value_counts()
    if (class_counts < 2).any():
        raise ValueError(
            "Each gesture class needs at least two samples for a stratified train/test split. "
            f"Observed counts: {class_counts.to_dict()}"
        )

    if isinstance(test_size, float):
        test_n = int(np.ceil(len(y) * test_size))
    else:
        test_n = int(test_size)

    train_n = len(y) - test_n
    if train_n <= 0 or test_n <= 0:
        raise ValueError(
            "Requested test/train sizes leave an empty split. "
            f"Adjust --test-size (current {test_size}) or provide more samples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = {
        "logreg": LogisticRegression(max_iter=1000, multi_class="auto"),
        "linear_svm": LinearSVC(),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=random_state),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            max_iter=400,
            alpha=1e-3,
            random_state=random_state,
        ),
    }

    results: Dict[str, Dict[str, object]] = {}
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=np.unique(y)),
            "labels": np.unique(y),
            "classification_report": classification_report(y_test, preds, zero_division=0),
            "pipeline": pipe,
        }
    return results


def main() -> None:
    args = parse_args()
    try:
        features, labels = load_dataset(args.data, args.label_col, tuple(args.id_cols))
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Failed to load dataset: {exc}")

    results = train_and_eval(features, labels, args.test_size, args.random_state)

    for name, info in results.items():
        print("\n===", name, "===")
        print(f"Accuracy:   {info['accuracy']:.3f}")
        print(f"Macro F1:   {info['macro_f1']:.3f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(info["confusion_matrix"], index=info["labels"], columns=info["labels"]))
        print("\nClassification report:")
        print(info["classification_report"])


if __name__ == "__main__":
    main()