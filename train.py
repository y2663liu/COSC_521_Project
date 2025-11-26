"""Train gesture classifiers using graph metrics.

This script performs robust Cross-Validation:
1. In-Subject: 10-Fold Cross-Validation (Single Run).
2. Cross-Subject: Leave-One-Group-Out (LOGO) Cross-Validation.

RFE Loop:
- Runs 25 rounds of Recursive Feature Elimination.
- Tracks the Best In-Subject and Best Cross-Subject performance.
- Saves the WINNING models (Pipeline + Feature List) to disk.
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
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

sklearn.set_config(transform_output="pandas")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Path to the feature CSV.")
    parser.add_argument("--label-col", default="gesture", help="Name of the label column.")
    parser.add_argument("--group-col", default="participant", help="Name of the participant column.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save the trained models.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_dataset(path: Path, label_col: str, group_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not path.exists(): raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)

    # Header check
    if not df.empty:
        cols_norm = [str(c).strip().lower() for c in df.columns]
        first_norm = [str(v).strip().lower() for v in df.iloc[0]]
        if sum(a == b for a, b in zip(cols_norm, first_norm)) / len(cols_norm) >= 0.6:
            print("Note: Dropping first row (repeated header).", file=sys.stderr)
            df = df.iloc[1:].reset_index(drop=True)

    if label_col not in df.columns: raise ValueError(f"Label column '{label_col}' not found.")
    if group_col not in df.columns: raise ValueError(f"Group column '{group_col}' not found.")

    labels = df[label_col].astype(str)
    groups = df[group_col].astype(str)
    
    cols_to_drop = {label_col, group_col, "interval"}
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    features = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    print(f"Loaded {len(df)} samples, {len(feature_cols)} features.")
    return features, labels, groups


def build_pipeline(model) -> Pipeline:
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")), 
        ("var_thresh", VarianceThreshold(threshold=0)), 
        ("scale", StandardScaler()),
        ("model", model)
    ])


def get_least_important_features(X: pd.DataFrame, y: pd.Series, n_remove: int = 2) -> List[str]:
    print(f"\n>>> Feature Importance Analysis (Drop bottom {n_remove})...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    pipe = build_pipeline(model)
    try:
        pipe.fit(X, y)
        feat_names = pipe.named_steps["var_thresh"].get_feature_names_out() # Get names after variance filter
        imps = pipe.named_steps["model"].feature_importances_
        
        df_imp = pd.DataFrame({"Feature": feat_names, "Importance": imps})
        df_imp = df_imp.sort_values(by="Importance", ascending=True)
        return df_imp["Feature"].head(n_remove).tolist()
    except Exception as e:
        print(f"Error calculating importance: {e}")
        return []


def get_models(random_state: int) -> Dict:
    return {
        "LogReg": LogisticRegression(max_iter=2000, solver='lbfgs', class_weight="balanced"),
        "LinearSVM": LinearSVC(dual="auto", max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state, class_weight="balanced"),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=2000, alpha=1e-3, random_state=random_state),
    }


def run_insubject_cv(X: pd.DataFrame, y: pd.Series, random_state: int) -> float:
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    models = get_models(random_state)
    best_acc = 0.0
    print("\n[In-Subject CV]")
    for name, model in models.items():
        pipe = build_pipeline(model)
        scores = cross_val_score(pipe, X, y, cv=kf, n_jobs=-1)
        mean_acc = np.mean(scores)
        print(f"{name:15s}: {mean_acc:.2%} (+/- {np.std(scores):.2%})")
        if mean_acc > best_acc: best_acc = mean_acc
    return best_acc


def run_cross_subject_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int) -> float:
    logo = LeaveOneGroupOut()
    models = get_models(random_state)
    best_acc = 0.0
    print("\n[Cross-Subject CV]")
    for name, model in models.items():
        pipe = build_pipeline(model)
        scores = cross_val_score(pipe, X, y, groups=groups, cv=logo, n_jobs=-1)
        mean_acc = np.mean(scores)
        print(f"{name:15s}: {mean_acc:.2%} (+/- {np.std(scores):.2%})")
        if mean_acc > best_acc: best_acc = mean_acc
    return best_acc


def save_final_models(X: pd.DataFrame, y: pd.Series, save_dir: Path, random_state: int, suffix: str = ""):
    if not save_dir.exists(): os.makedirs(save_dir)
    models = get_models(random_state)
    
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X, y)
        
        # SAVE METADATA: Save features list alongside the model
        # This ensures visualization scripts know which columns to pick from the raw CSV
        artifact = {
            "pipeline": pipe,
            "features": X.columns.tolist()
        }
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = save_dir / f"{safe_name}{suffix}.joblib"
        joblib.dump(artifact, filename)
        print(f"Saved {filename}")


def main() -> None:
    args = parse_args()
    try:
        X, y, groups = load_dataset(args.data, args.label_col, args.group_col)
    except Exception as e:
        sys.exit(f"Data Load Error: {e}")

    total_rounds = 25
    current_X = X.copy()
    
    best_insubject = {"round": 0, "features": [], "score": -1.0}
    best_cross = {"round": 0, "features": [], "score": -1.0}
    
    print(f"\n>>> Starting RFE ({total_rounds} Rounds)")
    
    for round_idx in range(1, total_rounds + 1):
        print(f"\n=== ROUND {round_idx} ({current_X.shape[1]} Features) ===")
        
        # 1. Evaluate
        score_in = run_insubject_cv(current_X, y, args.random_state)
        if score_in > best_insubject["score"]:
            best_insubject = {"round": round_idx, "features": current_X.columns.tolist(), "score": score_in}

        if len(groups.unique()) > 1:
            score_cross = run_cross_subject_cv(current_X, y, groups, args.random_state)
            if score_cross > best_cross["score"]:
                best_cross = {"round": round_idx, "features": current_X.columns.tolist(), "score": score_cross}
        
        # 2. Prune
        if round_idx < total_rounds:
            drop = get_least_important_features(current_X, y, n_remove=1)
            if not drop or current_X.shape[1] - len(drop) < 2: break
            current_X = current_X.drop(columns=drop)

    # 3. Save Best Models
    print("\n" + "="*60)
    print(f"BEST IN-SUBJECT:    Round {best_insubject['round']} (Acc: {best_insubject['score']:.2%})")
    print(f"BEST CROSS-SUBJECT: Round {best_cross['round']} (Acc: {best_cross['score']:.2%})")
    
    if args.save_dir:
        print("\n>>> Saving Winning Models...")
        save_final_models(X[best_insubject["features"]], y, args.save_dir, args.random_state, "_BestInSubject")
        save_final_models(X[best_cross["features"]], y, args.save_dir, args.random_state, "_BestCrossSubject")

if __name__ == "__main__":
    main()