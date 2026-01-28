import os
import sys
import numpy as np
import pandas as pd
import argparse

sys.path.append('..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=DATA_PATH)
args = parser.parse_args()

# Define probe sizes matching accuracy-sweep.py
probe_sizes = ['roberta-edu', 'edu-61M', 'ModernBERT-large', '13M', '29M', '61M', '113M', '224M']

thresholds = []

for probe_size in probe_sizes:
    print(f"Processing {probe_size}...")
    
    # Determine the correct path based on probe type
    if probe_size == 'roberta-edu':
        data_path = os.path.join(args.data_path, 'filtered-roberta')
    elif probe_size == 'ModernBERT-large':
        data_path = os.path.join(args.data_path, 'filtered-bert')
    elif probe_size == 'edu-61M':
        data_path = os.path.join(args.data_path, 'filtered-edu')
    else:
        data_path = os.path.join(args.data_path, f'filtered-{probe_size}')
    
    if probe_size == 'ModernBERT-large':
        val_filter_path = os.path.join(data_path, 'val_filter_bert.bin')
    else:
        val_filter_path = os.path.join(data_path, 'val_filter.bin')

    
    if not os.path.exists(val_filter_path):
        print(f"Warning: {val_filter_path} not found, skipping {probe_size}")
        continue
    
    # Load the predictions
    arr = np.memmap(val_filter_path, dtype=np.float16, mode='r')
    predictions = arr.astype(np.float32)
    
    # Compute 75th percentile (so 25% of tokens are classified as 1)
    threshold = np.quantile(predictions, 0.75)
    
    print(f"  Total tokens: {len(predictions)}")
    print(f"  Threshold (75th percentile): {threshold:.4f}")
    print(f"  Tokens above threshold: {(predictions > threshold).sum()} ({(predictions > threshold).sum() / len(predictions) * 100:.2f}%)")
    
    thresholds.append({
        'model': probe_size,
        'threshold': threshold
    })

# Save to CSV
df = pd.DataFrame(thresholds)
df.to_csv('../config/accuracy-sweep-thresholds.csv', index=False)
print(f"\nSaved thresholds to ../config/accuracy-sweep-thresholds.csv")
print(df)

