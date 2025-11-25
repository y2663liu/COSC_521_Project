"""Train gesture classifiers using graph metrics.

This script performs two types of validation:
1. In-Subject (Random Split): Tests physics generalization.
2. Cross-Subject (Group Split): Tests new user generalization.

It generates a Feature Importance Report AFTER training to help identify 
key metrics vs. noise.
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
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
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


def report_feature_importance(pipeline: Pipeline, model_name: str):
    """Extracts and prints feature importance from trained models."""
    
    # We need to get the feature names coming OUT of the preprocessor
    # (Because VarianceThreshold might have dropped some columns)
    try:
        preprocessor = pipeline.named_steps["pre"]
        feature_names = preprocessor.get_feature_names_out()
        
        model = pipeline.named_steps["model"]
        
        importances = None
        imp_type = ""

        # Strategy 1: Tree-based Importance (Random Forest)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            imp_type = "Gini Importance"
            
        # Strategy 2: Linear Model Coefficients (Logistic Regression / SVM)
        elif hasattr(model, "coef_"):
            # coef_ is shape (n_classes, n_features). We take the mean absolute value across classes.
            importances = np.mean(np.abs(model.coef_), axis=0)
            imp_type = "Mean Abs Coefficient"

        if importances is not None:
            df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            df_imp = df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
            
            print(f"\n--- {model_name} Feature Analysis ({imp_type}) ---")
            print("TOP 15 KEY METRICS:")
            print(df_imp.head(15))
            
            print("\nBOTTOM 10 UNIMPORTANT METRICS (Candidates for filtering):")
            print(df_imp.tail(10))
            print("-" * 40)
            
    except Exception as e:
        print(f"Could not extract feature importance for {model_name}: {e}")


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

    models = {
        "LogReg": LogisticRegression(
            max_iter=2000, 
            multi_class="auto", 
            solver='lbfgs',
            class_weight="balanced"
        ),
        "LinearSVM": LinearSVC(
            dual="auto", 
            max_iter=2000,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=None, 
            random_state=random_state,
            class_weight="balanced"
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            max_iter=2000,
            alpha=1e-3,
            random_state=random_state,
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
            
            # --- Feature Importance Reporting ---
            # We report this mainly for Random Forest (Non-linear) and LogReg (Linear)
            if name in ["RandomForest", "LogReg"]:
                report_feature_importance(pipe, name)

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