"""Train gesture classifiers using graph metrics.

This script performs two types of validation:
1. In-Subject (Random Split): Randomly shuffles all data. Tests if the model 
   understands the gesture physics generally, but may overfit to specific users.
   Default test size: 0.1 (10%).

2. Cross-Subject (Group Split): Holds out specific participants entirely. 
   Tests if the model generalizes to new users it has never seen before.
   Default hold-out: 1 participant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        type=Path,
        default=Path("train_data/raw_iou0.20_move0.90_dist10_simple_features_extended.csv"),
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
        "--test-size-random",
        type=float,
        default=0.1,
        help="Test split ratio for In-Subject validation (default: 0.1).",
    )
    parser.add_argument(
        "--test-groups-count",
        type=int,
        default=1,
        help="Number of participants to hold out for Cross-Subject validation (default: 1).",
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

    # 1. Header Cleaning: Check if first row looks like a duplicate header
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
    # Automatically detect classes from data (No filtered list)
    labels = df[label_col].astype(str)
    groups = df[group_col].astype(str)
    
    # Drop labels and groups from features. 
    cols_to_drop = {label_col, group_col, "interval"}
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    features = df[feature_cols]
    
    # Coerce to numeric
    features = features.apply(pd.to_numeric, errors='coerce')

    print(f"Loaded {len(df)} samples, {len(feature_cols)} features.")
    print(f"Classes found ({len(labels.unique())}): {sorted(labels.unique())}")
    
    return features, labels, groups


def build_pipeline(model) -> Pipeline:
    """Creates a robust preprocessing and training pipeline."""
    preprocessing = Pipeline(
        steps=[
            # Impute missing values (e.g. if a graph metric is NaN)
            ("impute", SimpleImputer(strategy="median")), 
            # Remove constant features (variance == 0)
            ("var_thresh", VarianceThreshold(threshold=0)), 
            # Standardize (Mean=0, Var=1) - CRITICAL for Normalization
            ("scale", StandardScaler()),
        ]
    )
    return Pipeline(steps=[("pre", preprocessing), ("model", model)])


def run_evaluation(
    title: str,
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    random_state: int
) -> None:
    """Trains models and prints a consolidated report for a specific split."""
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {title}")
    print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
    print(f"{'='*60}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Split resulted in empty train or test set. Skipping.")
        return

    # UPDATED: Added class_weight="balanced" to automatically reweight samples
    models = {
        "LogReg": LogisticRegression(
            max_iter=2000, 
            multi_class="auto", 
            solver='lbfgs',
            class_weight="balanced"  # <--- Balances gesture counts
        ),
        "LinearSVM": LinearSVC(
            dual="auto", 
            max_iter=2000,
            class_weight="balanced"  # <--- Balances gesture counts
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=None, 
            random_state=random_state,
            class_weight="balanced"  # <--- Balances gesture counts
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            max_iter=300,
            alpha=1e-3,
            random_state=random_state,
            # Note: MLPClassifier does not support class_weight="balanced" natively in sklearn
        ),
    }

    for name, model in models.items():
        pipe = build_pipeline(model)
        
        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            
            print(f"\n--- Model: {name} ---")
            print(f"Accuracy: {acc:.2%}")
            print(f"Macro F1: {f1:.3f}")
            
            labels = np.unique(np.concatenate((y_test, preds)))
            cm = confusion_matrix(y_test, preds, labels=labels)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            
            if name == "RandomForest":
                print("\nDetailed Report (Random Forest):")
                print(classification_report(y_test, preds, zero_division=0))
            else:
                print("Confusion Matrix:")
                print(cm_df)
                
        except Exception as e:
            print(f"Failed to train {name}: {e}")


def main() -> None:
    args = parse_args()
    
    try:
        X, y, groups = load_dataset(args.data, args.label_col, args.group_col)
    except Exception as exc:
        sys.exit(f"Data Load Error: {exc}")

    # --- VALIDATION 1: In-Subject ---
    print("\n\n>>> STARTING EXPERIMENT 1: In-Subject Validation (Mixed Data) <<<")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size_random, random_state=args.random_state)
    
    train_idx, test_idx = next(sss.split(X, y))
    run_evaluation(
        f"In-Subject (Random Split {args.test_size_random*100}%)",
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        args.random_state
    )

    # --- VALIDATION 2: Cross-Subject ---
    print("\n\n>>> STARTING EXPERIMENT 2: Cross-Subject Validation (Unseen Users) <<<")
    
    unique_groups = groups.unique()
    n_groups = len(unique_groups)
    
    if n_groups <= args.test_groups_count:
        print(f"WARNING: Total participants ({n_groups}) <= hold-out count ({args.test_groups_count}). Skipping.")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_groups_count, random_state=args.random_state)
        
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        test_participants = groups.iloc[test_idx].unique()
        print(f"Total Participants: {n_groups}")
        print(f"Held-out Participants ({len(test_participants)}): {test_participants}")
        
        run_evaluation(
            f"Cross-Subject (Held out {args.test_groups_count} users)",
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            args.random_state
        )


if __name__ == "__main__":
    main()