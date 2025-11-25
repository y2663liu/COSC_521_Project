"""Train gesture classifiers using graph metrics.

This script performs robust Cross-Validation:
1. In-Subject: 10-Fold Cross-Validation (Single Run).
   Tests theoretical model capacity on mixed data.
2. Cross-Subject: Leave-One-Group-Out (LOGO) Cross-Validation.
   Iterates over EVERY participant to test generalization to new users.

It generates a Feature Importance Report based on the full dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedKFold, 
    LeaveOneGroupOut, 
    cross_val_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC

# Force scikit-learn to keep pandas column names through the pipeline
sklearn.set_config(transform_output="pandas")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        type=Path,
        default=Path("train_data/raw_iou0.20_move0.90_dist10_simple_features_extended_rewritten.csv"),
        nargs="?",
        help="Path to the feature CSV.",
    )
    parser.add_argument(
        "--label-col",
        default="gesture",
        help="Name of the label column (default: gesture).",
    )
    parser.add_argument(
        "--group-col",
        default="participant",
        help="Name of the participant/subject column for cross-subject validation (default: participant).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def load_dataset(path: Path, label_col: str, group_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Loads data and separates Features (X), Labels (y), and Groups (participants)."""
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # 1. Header Cleaning
    if not df.empty:
        cols_norm = [str(c).strip().lower() for c in df.columns]
        first_norm = [str(v).strip().lower() for v in df.iloc[0]]
        match_ratio = sum(a == b for a, b in zip(cols_norm, first_norm)) / len(cols_norm)
        if match_ratio >= 0.6:
            print("Note: Dropping first row (appears to be repeated header).", file=sys.stderr)
            df = df.iloc[1:].reset_index(drop=True)

    # 2. Validation
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found. Required for cross-subject validation.")

    # 3. Extraction
    labels = df[label_col].astype(str)
    groups = df[group_col].astype(str)
    
    cols_to_drop = {label_col, group_col, "interval"}
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    features = df[feature_cols]
    
    # Coerce to numeric
    features = features.apply(pd.to_numeric, errors='coerce')

    print(f"Loaded {len(df)} samples, {len(feature_cols)} features.")
    print(f"Classes found ({len(labels.unique())}): {sorted(labels.unique())}")
    print(f"Participants found ({len(groups.unique())}): {sorted(groups.unique())}")
    
    return features, labels, groups


def build_pipeline(model) -> Pipeline:
    """Creates a robust preprocessing and training pipeline."""
    preprocessing = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")), 
            ("var_thresh", VarianceThreshold(threshold=0)), 
            ("scale", StandardScaler()),
        ]
    )
    return Pipeline(steps=[("pre", preprocessing), ("model", model)])


def report_feature_importance(X: pd.DataFrame, y: pd.Series):
    """Trains a single RF on full data to report Feature Importance."""
    print("\n" + "="*60)
    print("GLOBAL FEATURE IMPORTANCE ANALYSIS (Gini)")
    print("="*60)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    pipe = build_pipeline(model)
    
    try:
        pipe.fit(X, y)
        
        # Get names out of preprocessor
        pre = pipe.named_steps["pre"]
        feat_names = pre.get_feature_names_out()
        
        # Get importances
        rf = pipe.named_steps["model"]
        imps = rf.feature_importances_
        
        df_imp = pd.DataFrame({"Feature": feat_names, "Importance": imps})
        df_imp = df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        print("TOP 15 KEY METRICS:")
        print(df_imp.head(15))
        print("\nBOTTOM 10 CANDIDATES FOR REMOVAL:")
        print(df_imp.tail(10))
        
    except Exception as e:
        print(f"Skipping importance report: {e}")


def get_models(random_state: int) -> Dict:
    """Returns dictionary of models to train."""
    return {
        "LogReg": LogisticRegression(
            max_iter=2000, solver='lbfgs', class_weight="balanced"
        ),
        "LinearSVM": LinearSVC(
            dual="auto", max_iter=2000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=random_state, class_weight="balanced"
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", max_iter=2000, alpha=1e-3, random_state=random_state,
        ),
    }


def run_insubject_cv(X: pd.DataFrame, y: pd.Series, random_state: int):
    """10-Fold Cross Validation (Average Accuracy)."""
    print("\n\n" + "="*60)
    print("EXPERIMENT 1: In-Subject Validation (10-Fold CV)")
    print("Tests robustness on mixed data.")
    print("="*60)
    
    # Single run of 10-fold CV
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    models = get_models(random_state)
    
    for name, model in models.items():
        pipe = build_pipeline(model)
        # Returns array of 10 scores
        scores = cross_val_score(pipe, X, y, cv=kf, n_jobs=-1)
        
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        
        print(f"Model: {name:15s} | Avg Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")


def run_cross_subject_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int):
    """Leave-One-Group-Out Cross Validation."""
    print("\n\n" + "="*60)
    print("EXPERIMENT 2: Cross-Subject Validation (Leave-One-Group-Out)")
    print(f"Iterating over {len(groups.unique())} participants. Tests new user generalization.")
    print("="*60)
    
    logo = LeaveOneGroupOut()
    models = get_models(random_state)
    
    for name, model in models.items():
        pipe = build_pipeline(model)
        
        # cross_val_score automatically handles the groups iteration
        scores = cross_val_score(pipe, X, y, groups=groups, cv=logo, n_jobs=-1)
        
        mean_acc = np.mean(scores)
        std_acc = np.std(scores) # High std dev means it works great for some users, fails for others
        
        print(f"Model: {name:15s} | Avg Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")


def main() -> None:
    args = parse_args()
    
    try:
        X, y, groups = load_dataset(args.data, args.label_col, args.group_col)
    except Exception as exc:
        sys.exit(f"Data Load Error: {exc}")

    if len(X) < 50:
        print("WARNING: Dataset is extremely small. Cross-validation results may be unstable.")

    # 1. Feature Analysis (Global)
    report_feature_importance(X, y)

    # 2. In-Subject (10-fold)
    run_insubject_cv(X, y, args.random_state)

    # 3. Cross-Subject (Iterate all participants)
    if len(groups.unique()) > 1:
        run_cross_subject_cv(X, y, groups, args.random_state)
    else:
        print("\nSkipping Cross-Subject: Only 1 participant found in data.")


if __name__ == "__main__":
    main()