"""Train gesture classifiers using graph metrics.

This script performs robust Cross-Validation:
1. In-Subject: 10-Fold Cross-Validation (Single Run).
2. Cross-Subject: Leave-One-Group-Out (LOGO) Cross-Validation.

It implements a Recursive Feature Elimination (RFE) loop:
- Runs the training/evaluation 5 times.
- Keeps track of the best performing feature sets.
- At the end, saves ONLY the models corresponding to the best In-Subject 
  and best Cross-Subject performance.
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
import joblib  # For saving models
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
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save the trained models.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def load_dataset(path: Path, label_col: str, group_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if not df.empty:
        cols_norm = [str(c).strip().lower() for c in df.columns]
        first_norm = [str(v).strip().lower() for v in df.iloc[0]]
        match_ratio = sum(a == b for a, b in zip(cols_norm, first_norm)) / len(cols_norm)
        if match_ratio >= 0.6:
            print("Note: Dropping first row (appears to be repeated header).", file=sys.stderr)
            df = df.iloc[1:].reset_index(drop=True)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found.")

    labels = df[label_col].astype(str)
    groups = df[group_col].astype(str)
    
    cols_to_drop = {label_col, group_col, "interval"}
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    features = df[feature_cols]
    features = features.apply(pd.to_numeric, errors='coerce')

    print(f"Loaded {len(df)} samples, {len(feature_cols)} features.")
    print(f"Classes found ({len(labels.unique())}): {sorted(labels.unique())}")
    
    return features, labels, groups


def build_pipeline(model) -> Pipeline:
    preprocessing = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")), 
            ("var_thresh", VarianceThreshold(threshold=0)), 
            ("scale", StandardScaler()),
        ]
    )
    return Pipeline(steps=[("pre", preprocessing), ("model", model)])


def get_least_important_features(X: pd.DataFrame, y: pd.Series, n_remove: int = 2) -> List[str]:
    """Trains a RF to find the least important features to prune."""
    print(f"\n>>> Analyzing Feature Importance to drop bottom {n_remove}...")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    pipe = build_pipeline(model)
    
    try:
        pipe.fit(X, y)
        pre = pipe.named_steps["pre"]
        feat_names = pre.get_feature_names_out()
        rf = pipe.named_steps["model"]
        imps = rf.feature_importances_
        
        df_imp = pd.DataFrame({"Feature": feat_names, "Importance": imps})
        df_imp = df_imp.sort_values(by="Importance", ascending=True) 
        
        to_drop = df_imp["Feature"].head(n_remove).tolist()
        
        print(f"Dropping features: {to_drop}")
        return to_drop
        
    except Exception as e:
        print(f"Error calculating importance: {e}")
        return []


def get_models(random_state: int) -> Dict:
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


def run_insubject_cv(X: pd.DataFrame, y: pd.Series, random_state: int) -> float:
    """Returns the BEST average accuracy achieved by any model in this suite."""
    print("\n--- EXPERIMENT 1: In-Subject Validation (10-Fold CV) ---")
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    models = get_models(random_state)
    
    best_acc = 0.0
    
    for name, model in models.items():
        pipe = build_pipeline(model)
        scores = cross_val_score(pipe, X, y, cv=kf, n_jobs=-1)
        mean_acc = np.mean(scores)
        
        print(f"Model: {name:15s} | Avg Accuracy: {mean_acc:.2%} (+/- {np.std(scores):.2%})")
        
        if mean_acc > best_acc:
            best_acc = mean_acc
            
    return best_acc


def run_cross_subject_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int) -> float:
    """Returns the BEST average accuracy achieved by any model in this suite."""
    print("\n--- EXPERIMENT 2: Cross-Subject Validation (Leave-One-Group-Out) ---")
    
    logo = LeaveOneGroupOut()
    models = get_models(random_state)
    
    best_acc = 0.0
    
    for name, model in models.items():
        pipe = build_pipeline(model)
        scores = cross_val_score(pipe, X, y, groups=groups, cv=logo, n_jobs=-1)
        mean_acc = np.mean(scores)
        
        print(f"Model: {name:15s} | Avg Accuracy: {mean_acc:.2%} (+/- {np.std(scores):.2%})")
        
        if mean_acc > best_acc:
            best_acc = mean_acc
            
    return best_acc


def save_final_models(X: pd.DataFrame, y: pd.Series, save_dir: Path, random_state: int, suffix: str = ""):
    """Trains models on ALL data and saves them to disk."""
    if not save_dir.exists():
        os.makedirs(save_dir)
        
    models = get_models(random_state)
    
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X, y)
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = save_dir / f"{safe_name}{suffix}.joblib"
        
        joblib.dump(pipe, filename)
        print(f"Saved {name} to: {filename}")


def main() -> None:
    args = parse_args()
    
    try:
        X, y, groups = load_dataset(args.data, args.label_col, args.group_col)
    except Exception as exc:
        sys.exit(f"Data Load Error: {exc}")

    if len(X) < 50:
        print("WARNING: Dataset is extremely small. Cross-validation results may be unstable.")

    # RFE Loop Configuration
    total_rounds = 25
    current_X = X.copy()
    
    # Track best performance
    # Structure: {"round": int, "features": [cols], "score": float}
    best_insubject = {"round": 0, "features": [], "score": -1.0}
    best_cross = {"round": 0, "features": [], "score": -1.0}
    
    print(f"\n>>> Starting Recursive Feature Elimination ({total_rounds} Rounds)")
    
    for round_idx in range(1, total_rounds + 1):
        print("\n" + "="*60)
        print(f"ROUND {round_idx} / {total_rounds}")
        print(f"Features Count: {current_X.shape[1]}")
        print("="*60)
        
        # 1. In-Subject Validation
        score_in = run_insubject_cv(current_X, y, args.random_state)
        if score_in > best_insubject["score"]:
            best_insubject = {
                "round": round_idx, 
                "features": current_X.columns.tolist(), 
                "score": score_in
            }

        # 2. Cross-Subject Validation
        if len(groups.unique()) > 1:
            score_cross = run_cross_subject_cv(current_X, y, groups, args.random_state)
            if score_cross > best_cross["score"]:
                best_cross = {
                    "round": round_idx, 
                    "features": current_X.columns.tolist(), 
                    "score": score_cross
                }
        else:
            score_cross = 0.0

        # 3. Pruning (Prepare for next round)
        if round_idx < total_rounds:
            drop_candidates = get_least_important_features(current_X, y, n_remove=1)
            
            if not drop_candidates:
                print("No features returned to drop. Stopping RFE early.")
                break
            if current_X.shape[1] - len(drop_candidates) < 2:
                print("Warning: Too few features remaining. Stopping RFE.")
                break
                
            current_X = current_X.drop(columns=drop_candidates)

    # --- FINAL SUMMARY & SAVING ---
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print(f"BEST IN-SUBJECT:    Round {best_insubject['round']} (Acc: {best_insubject['score']:.2%})")
    print(f"BEST CROSS-SUBJECT: Round {best_cross['round']} (Acc: {best_cross['score']:.2%})")
    
    if args.save_dir:
        print("\n>>> Saving Top Performing Models...")
        
        # Save Best In-Subject
        cols_in = best_insubject["features"]
        save_final_models(X[cols_in], y, args.save_dir, args.random_state, suffix="_BestInSubject")
        
        # Save Best Cross-Subject (Check if it's different to avoid redundant work, though saving twice is fine)
        cols_cross = best_cross["features"]
        save_final_models(X[cols_cross], y, args.save_dir, args.random_state, suffix="_BestCrossSubject")
        
        print(f"\nModels saved to {args.save_dir}")


if __name__ == "__main__":
    main()