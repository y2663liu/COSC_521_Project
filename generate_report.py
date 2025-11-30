import argparse
import joblib
import sys
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# --- RECONSTRUCTION HELPERS ---
def get_model_instance(name, random_state=42):
    """Recreates a fresh model instance based on the name string."""
    if "RandomForest" in name:
        return RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state, class_weight="balanced")
    elif "LogReg" in name:
        return LogisticRegression(max_iter=2000, solver='lbfgs', class_weight="balanced")
    elif "LinearSVM" in name:
        return LinearSVC(dual="auto", max_iter=2000, class_weight="balanced")
    elif "MLP" in name:
        return MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=2000, alpha=1e-3, random_state=random_state)
    else:
        # Default fallback if name parsing fails
        return RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")

def build_clean_pipeline(model_instance):
    """Builds a fresh pipeline structure matching train.py."""
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")), 
        ("var_thresh", VarianceThreshold(threshold=0)), 
        ("scale", StandardScaler()),
        ("model", model_instance)
    ])

# --- DATA LOADING ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Visual Report from Config")
    parser.add_argument("--config", default="pipeline_config.json", help="Path to configuration file")
    return parser.parse_args()

def load_config_and_paths(config_path):
    if not os.path.exists(config_path):
        sys.exit(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    pool_lbl = "pool" if config.get("use_pooling", False) else "raw"
    rm_dup_lbl = "simple" if config.get("remove_duplicates", True) else "complex"
    
    subdir_name = (
        f"{pool_lbl}_iou{config['iou_threshold']:.2f}_"
        f"move{config['movement_threshold']:.2f}_"
        f"dist{config['max_match_dist']:.0f}_{rm_dup_lbl}"
    )
    
    feature_dir = Path(config.get("feature_dir", "train_data"))
    data_path = feature_dir / f"{subdir_name}_features_extended_rewritten.csv"
    
    # Directory where models are stored
    model_dir = Path("result") / subdir_name
    
    return data_path, model_dir, "gesture", "participant"

def load_full_dataset(data_path, label_col, group_col):
    if not data_path.exists(): sys.exit(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    
    # Clean headers
    cols_norm = [str(c).strip().lower() for c in df.columns]
    first_norm = [str(v).strip().lower() for v in df.iloc[0]]
    if sum(a == b for a, b in zip(cols_norm, first_norm)) / len(cols_norm) >= 0.6:
        df = df.iloc[1:].reset_index(drop=True)
    
    y = df[label_col].astype(str)
    groups = df[group_col].astype(str)
    return df, y, groups

def reconstruct_model_from_file(model_path):
    """Loads joblib artifact and rebuilds a fresh pipeline."""
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "features" not in artifact:
        print(f"Skipping {model_path}: Invalid format.")
        return None, None

    feature_names = artifact["features"]
    model_type_str = model_path.name
    fresh_model = get_model_instance(model_type_str)
    pipeline = build_clean_pipeline(fresh_model)
    
    return pipeline, feature_names

# --- PLOTTING ---
def plot_confusion_matrix_percent(y_true, y_pred, title, filename):
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Row-normalized percentages
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{title}\nConfusion Matrix (Row Normalized)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved plot: {filename}")
    plt.close()

def plot_tsne(X, y, filename):
    print("Generating t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette="tab10", s=60, alpha=0.8)
    plt.title("t-SNE Feature Visualization (Best Model Features)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved plot: {filename}")
    plt.close()

# --- EVALUATION LOGIC ---
def evaluate_model_metrics(pipeline, X, y, groups):
    """Calculates Acc/F1 for In-Subject and Cross-Subject without plotting."""
    
    # 1. In-Subject (Stratified 10-Fold)
    cv_in = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model_in = clone(pipeline)
    y_pred_in = cross_val_predict(model_in, X, y, cv=cv_in, n_jobs=-1)
    
    acc_in = accuracy_score(y, y_pred_in)
    f1_in = f1_score(y, y_pred_in, average="macro")
    
    # 2. Cross-Subject (LOGO)
    acc_cross, f1_cross = 0.0, 0.0
    if len(groups.unique()) > 1:
        logo = LeaveOneGroupOut()
        model_cross = clone(pipeline)
        y_pred_cross = cross_val_predict(model_cross, X, y, groups=groups, cv=logo, n_jobs=-1)
        acc_cross = accuracy_score(y, y_pred_cross)
        f1_cross = f1_score(y, y_pred_cross, average="macro")
    
    return acc_in, f1_in, acc_cross, f1_cross

def main():
    args = parse_args()
    data_path, model_dir, label_col, group_col = load_config_and_paths(args.config)
    
    print(f"Processing models in: {model_dir}")
    
    # 1. Load Data Frame
    df_full, y_full, groups_full = load_full_dataset(data_path, label_col, group_col)
    
    # 2. Find all Cross-Subject models
    if not model_dir.exists():
        sys.exit(f"Model directory not found: {model_dir}")
        
    model_files = list(model_dir.glob("*_BestCrossSubject.joblib"))
    if not model_files:
        sys.exit("No models ending with _BestCrossSubject.joblib found.")
        
    print(f"Found {len(model_files)} models. Evaluating...")
    
    results = []
    
    # 3. Iterate and Evaluate
    for m_path in model_files:
        print(f"Evaluating: {m_path.name}")
        pipeline, feature_names = reconstruct_model_from_file(m_path)
        
        if pipeline is None: continue
        
        # Subset features for this specific model
        try:
            X = df_full[feature_names].apply(pd.to_numeric, errors='coerce')
        except KeyError as e:
            print(f"  Error: Missing features in CSV for this model. {e}")
            continue
            
        acc_in, f1_in, acc_cross, f1_cross = evaluate_model_metrics(pipeline, X, y_full, groups_full)
        
        results.append({
            "Model_File": m_path.name,
            "Num_Features": len(feature_names),
            "InSubject_Acc": acc_in,
            "InSubject_F1": f1_in,
            "CrossSubject_Acc": acc_cross,
            "CrossSubject_F1": f1_cross,
            "Path": m_path,
            "Pipeline": pipeline,
            "Features": feature_names
        })

    # 4. Save CSV Report
    if not results:
        sys.exit("No results generated.")
        
    df_results = pd.DataFrame(results).drop(columns=["Path", "Pipeline", "Features"])
    csv_path = model_dir / "model_comparison_report.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nComparison report saved to: {csv_path}")
    print(df_results)

    # 5. Identify Best Model (by Cross-Subject F1)
    best_run = max(results, key=lambda x: x["CrossSubject_F1"])
    print(f"\nBest Model: {best_run['Model_File']} (Cross-F1: {best_run['CrossSubject_F1']:.4f})")
    
    # 6. Generate Visuals ONLY for the Best Model
    print("\nGenerating visuals for the best model...")
    
    best_pipeline = best_run["Pipeline"]
    best_features = best_run["Features"]
    X_best = df_full[best_features].apply(pd.to_numeric, errors='coerce')
    model_name = Path(best_run["Model_File"]).stem
    
    # A. t-SNE
    if hasattr(best_pipeline, "steps"):
        pre = best_pipeline[:-1]
        pre.fit(X_best)
        X_pre = pre.transform(X_best)
    else:
        X_pre = X_best
    plot_tsne(X_pre, y_full, model_dir / f"{model_name}_tsne.png")
    
    # B. Confusion Matrices (Recalculate predictions to plot)
    # In-Subject
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model_clone = clone(best_pipeline)
    y_pred_in = cross_val_predict(model_clone, X_best, y_full, cv=cv, n_jobs=-1)
    plot_confusion_matrix_percent(y_full, y_pred_in, f"In-Subject ({model_name})", model_dir / f"{model_name}_eval_insubject.png")
    
    # Cross-Subject
    if len(groups_full.unique()) > 1:
        logo = LeaveOneGroupOut()
        model_clone = clone(best_pipeline)
        y_pred_cross = cross_val_predict(model_clone, X_best, y_full, groups=groups_full, cv=logo, n_jobs=-1)
        plot_confusion_matrix_percent(y_full, y_pred_cross, f"Cross-Subject ({model_name})", model_dir / f"{model_name}_eval_crosssubject.png")

if __name__ == "__main__":
    main()