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
        raise ValueError(f"Unknown model type in filename: {name}")

def build_clean_pipeline(model_instance):
    """Builds a fresh pipeline structure matching train.py."""
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")), 
        ("var_thresh", VarianceThreshold(threshold=0)), 
        ("scale", StandardScaler()),
        ("model", model_instance)
    ])

# --- MAIN LOGIC ---

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
    
    model_dir = Path("result") / subdir_name
    model_path = None
    
    if model_dir.exists():
        candidates = list(model_dir.glob("best_model.joblib"))
        if not candidates:
            candidates = list(model_dir.glob("MLP_BestCrossSubject.joblib"))
        if candidates:
            model_path = candidates[0]
            print(f"Found model file: {model_path}")
    
    return data_path, model_path, "gesture", "participant"

def load_data_and_reconstruct_model(data_path, model_path, label_col, group_col):
    # 1. Load Data
    if not data_path.exists(): sys.exit(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    
    cols_norm = [str(c).strip().lower() for c in df.columns]
    first_norm = [str(v).strip().lower() for v in df.iloc[0]]
    if sum(a == b for a, b in zip(cols_norm, first_norm)) / len(cols_norm) >= 0.6:
        df = df.iloc[1:].reset_index(drop=True)
        
    # 2. Load Artifact
    if not model_path or not model_path.exists(): 
        sys.exit(f"Model file not found at: {model_path}")
    
    # We load the artifact, but we might discard the pickled pipeline object if it's broken
    artifact = joblib.load(model_path)
    
    if not isinstance(artifact, dict) or "features" not in artifact:
        sys.exit("Invalid model file format.")
        
    feature_names = artifact["features"]
    print(f"\n>>> Features used ({len(feature_names)}): {', '.join(feature_names)}")
    
    # 3. Reconstruct Fresh Pipeline
    # We determine the model type from the filename (e.g., "MLP_BestCrossSubject.joblib")
    print(">>> Reconstructing fresh pipeline to avoid version mismatches...")
    model_type_str = model_path.name
    fresh_model = get_model_instance(model_type_str)
    pipeline = build_clean_pipeline(fresh_model)
    
    # 4. Prepare Data
    try:
        X = df[feature_names].apply(pd.to_numeric, errors='coerce')
        y = df[label_col].astype(str)
        groups = df[group_col].astype(str)
    except KeyError as e:
        sys.exit(f"Feature missing in CSV: {e}")
        
    return X, y, groups, pipeline, model_path.stem

def plot_confusion_matrix(y_true, y_pred, title, filename):
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    
    print(f"\n--- {title} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{title}\nConfusion Matrix")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved plot: {filename}")
    plt.close()

def plot_tsne(X, y, filename):
    print("\nGenerating t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette="tab10", s=60, alpha=0.8)
    plt.title("t-SNE Feature Visualization")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved plot: {filename}")
    plt.close()

def main():
    args = parse_args()
    data_path, model_path, label_col, group_col = load_config_and_paths(args.config)
    
    # Load data and build a FRESH pipeline structure
    X, y, groups, pipeline, model_name = load_data_and_reconstruct_model(data_path, model_path, label_col, group_col)
    
    output_dir = model_path.parent
    print(f"Generating report in: {output_dir}")

    # 1. t-SNE
    # We need to fit the preprocessing steps on the data first since it's a fresh pipeline
    # We do this on the whole dataset just for visualization purposes
    preprocessor = pipeline[:-1]
    preprocessor.fit(X) 
    X_pre = preprocessor.transform(X)
    plot_tsne(X_pre, y, output_dir / f"{model_name}_tsne.png")

    # 2. In-Subject CV
    print("\nRunning In-Subject Validation (10-Fold)...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model_clone = clone(pipeline) 
    y_pred_in = cross_val_predict(model_clone, X, y, cv=cv, n_jobs=-1)
    
    plot_confusion_matrix(y, y_pred_in, "In-Subject Validation", output_dir / f"{model_name}_eval_insubject.png")

    # 3. Cross-Subject CV
    if len(groups.unique()) > 1:
        print("\nRunning Cross-Subject Validation (LOGO)...")
        logo = LeaveOneGroupOut()
        model_clone = clone(pipeline)
        y_pred_cross = cross_val_predict(model_clone, X, y, groups=groups, cv=logo, n_jobs=-1)
        
        plot_confusion_matrix(y, y_pred_cross, "Cross-Subject Validation", output_dir / f"{model_name}_eval_crosssubject.png")
    else:
        print("Skipping Cross-Subject (Only 1 group found)")

if __name__ == "__main__":
    main()