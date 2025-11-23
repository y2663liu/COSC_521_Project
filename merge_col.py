import pandas as pd

# Load the CSV file
folder = "train_data"
file = "raw_iou0.20_move0.50_dist20_simple"
df = pd.read_csv(f"{folder}/{file}.csv")

# Replace gesture labels
df['gesture'] = df['gesture'].replace({
    'SwipeH': 'Swipe',
    'SwipeV': 'Swipe',
    'Pat': 'Pat',
    'Slap': 'Pat'
})

# Save to a new file
df.to_csv(f"{folder}/{file}_rewritten.csv", index=False)

print("✅ Gesture labels rewritten")
