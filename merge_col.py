import pandas as pd
import json
import os

# 1. Load Configuration
with open("pipeline_config.json", "r") as f:
    config = json.load(f)

# 2. Reconstruct the filename expected from extract_features.R
pool_lbl = "pool" if config["use_pooling"] else "raw"
rm_dup_lbl = "simple" if config["remove_duplicates"] else "complex"

subdir_name = (
    f"{pool_lbl}_iou{config['iou_threshold']:.2f}_"
    f"move{config['movement_threshold']:.2f}_"
    f"dist{config['max_match_dist']:.0f}_{rm_dup_lbl}"
)

folder = config["feature_dir"]
input_file = f"{folder}/{subdir_name}_features_extended.csv"
output_file = f"{folder}/{subdir_name}_features_extended_rewritten.csv"

print(f"Reading: {input_file}")

if not os.path.exists(input_file):
    print(f"Error: Input file not found! Run extract_features.R first.")
    exit(1)

# 3. Load Data
df = pd.read_csv(input_file)

# 4. Clean Data (Custom Logic)
df = df[df['participant'] != 'Iverson_Right']

# Replace gesture labels
df['gesture'] = df['gesture'].replace({
    'SwipeH': 'Swipe',
    'SwipeV': 'Swipe',
    'Pat': 'Pat',
    'Slap': 'Pat'
})

# 5. Save
df.to_csv(output_file, index=False)
print(f"✅ Gesture labels rewritten. Saved to: {output_file}")