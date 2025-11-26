import subprocess
import json
import os
import sys
from pathlib import Path

def run_command(cmd, log_file=None):
    print(f"\n>>> RUNNING: {cmd}")
    
    if log_file:
        # Run and redirect stdout/stderr to file AND print to console
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, "w") as f:
            # We use Popen to stream output to console and file simultaneously
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            
            # Stream output
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
            
            process.wait()
            if process.returncode != 0:
                print(f"Error running command: {cmd}")
                sys.exit(process.returncode)
    else:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Error running command: {cmd}")
            sys.exit(result.returncode)

def main():
    print("=== Starting Gesture Recognition Pipeline ===")

    # 1. Read Config
    with open("pipeline_config.json", "r") as f:
        config = json.load(f)

    pool_lbl = "pool" if config["use_pooling"] else "raw"
    rm_dup_lbl = "simple" if config["remove_duplicates"] else "complex"
    
    # Construct folder/file name from hyperparams
    subdir_name = (
        f"{pool_lbl}_iou{config['iou_threshold']:.2f}_"
        f"move{config['movement_threshold']:.2f}_"
        f"dist{config['max_match_dist']:.0f}_{rm_dup_lbl}"
    )
    
    # Input file for training
    final_train_file = f"{config['feature_dir']}/{subdir_name}_features_extended_rewritten.csv"
    
    # Output directory for this run's artifacts (logs + models)
    # Example: result/raw_iou0.20.../
    output_dir = Path("result") / subdir_name
    log_file = output_dir / "performance.txt"

    print(f"Artifacts will be saved to: {output_dir}")

    # 2. Execute Pipeline Steps
    
    # Step A: Build Networks (R)
    run_command("Rscript build_network.R")
    
    # Step B: Extract Features (R)
    run_command("Rscript extract_features.R")
    
    # Step C: Merge & Clean (Python)
    run_command("python merge_col.py")
    
    # Step D: Train Model (Python)
    # We pass the data file AND the output directory for saving models/logs
    train_cmd = f"python train.py {final_train_file} --save-dir {output_dir}"
    
    print(f"\n>>> Training Model & Saving Results to {log_file} ...")
    run_command(train_cmd, log_file=log_file)

    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()