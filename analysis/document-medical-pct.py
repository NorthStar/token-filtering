"""
Analyze the percentage of tokens per document classified as medical in the validation set.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_PATH
from colors import MASK_COLORS, THEME_COLORS

from plotnine import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='path to data directory')
parser.add_argument('--filter_dir', type=str, default='filtered-224M', help='directory containing filtered data')
parser.add_argument('--split', type=str, default='val', help='which split to analyze')
parser.add_argument('--threshold', type=float, default=0.18597136437892914, help='threshold for classifying as medical')
parser.add_argument('--rerun', action='store_true', help='recompute values even if CSV exists')
args = parser.parse_args()

csv_path = f'results/document-medical-pct-{args.filter_dir}.csv'

if os.path.exists(csv_path) and not args.rerun:
    print(f"loading cached results from {csv_path}")
    df = pd.read_csv(csv_path)
    medical_percentages = df['medical_pct'].values
else:
    # Load validation data
    filter_path = os.path.join(args.data_path, args.filter_dir)

    tokens_file = os.path.join(filter_path, f'{args.split}.bin')
    filter_file = os.path.join(filter_path, f'{args.split}_filter.bin')
    lens_file = os.path.join(filter_path, f'{args.split}_lens.bin')

    print(f"loading data from {filter_path}")
    print(f"  tokens: {tokens_file}")
    print(f"  filter: {filter_file}")
    print(f"  lengths: {lens_file}")

    # Load the data
    tokens = np.memmap(tokens_file, dtype=np.uint32, mode='r')
    filter_probs = np.memmap(filter_file, dtype=np.float16, mode='r')
    doc_lengths = np.memmap(lens_file, dtype=np.uint32, mode='r')

    print(f"\ndata shapes:")
    print(f"  tokens: {len(tokens)}")
    print(f"  filter_probs: {len(filter_probs)}")
    print(f"  doc_lengths: {len(doc_lengths)} documents")
    print(f"  sum of doc_lengths: {np.sum(doc_lengths)}")

    # Compute per-document percentage
    medical_percentages = []
    start_idx = 0

    for doc_len in tqdm(doc_lengths, desc="computing per-document stats"):
        end_idx = start_idx + doc_len

        if doc_len > 0:
            doc_probs = filter_probs[start_idx:end_idx]
            medical_count = np.sum(doc_probs > args.threshold)
            pct_medical = 100.0 * medical_count / doc_len
            medical_percentages.append(pct_medical)

        start_idx = end_idx

    medical_percentages = np.array(medical_percentages)

    # Save to CSV
    df = pd.DataFrame({'medical_pct': medical_percentages})
    df.to_csv(csv_path, index=False)
    print(f"\nsaved results to {csv_path}")

print(f"\nmedical percentage statistics:")
print(f"  mean: {np.mean(medical_percentages):.2f}%")
print(f"  median: {np.median(medical_percentages):.2f}%")
print(f"  std: {np.std(medical_percentages):.2f}%")
print(f"  min: {np.min(medical_percentages):.2f}%")
print(f"  max: {np.max(medical_percentages):.2f}%")

print(f"\n% of documents exceeding threshold:")
for threshold in [5, 10, 25, 50, 75]:
    pct_above = 100.0 * np.mean(medical_percentages > threshold)
    print(f"  >{threshold}% medical: {pct_above:.2f}% of documents")

# Create dataframe for plotting
df = pd.DataFrame({'medical_pct': medical_percentages})

# Plot style - use mask color from Ohchi palette and standard theme
color = MASK_COLORS['remove']
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']

# Histogram
p_hist = (
    ggplot(df, aes(x='medical_pct', y=after_stat('count/sum(count)')))
    + geom_histogram(fill=color, alpha=0.7, color='white', bins=50)
    + theme_bw(base_family='Helvetica Neue')
    + theme(
        figure_size=(3.375, 2.5),
        panel_grid_major=element_line(size=0.3, color=grid_color),
        panel_grid_minor=element_blank(),
        axis_title_x=element_text(size=9, color=text_color),
        axis_title_y=element_text(size=9, color=text_color),
        axis_text_x=element_text(size=7, color=text_color),
        axis_text_y=element_text(size=7, color=text_color),
        plot_background=element_rect(fill=bg_color),
        panel_background=element_rect(fill=bg_color),
        panel_border=element_rect(color=line_color, size=0.5),
        axis_ticks=element_line(size=0.5),
        axis_ticks_minor=element_blank(),
    )
    + labs(x='% Medical Tokens Per Document', y='% of Documents')
    + coord_cartesian(xlim=(0, 100))
)

output_path_hist = f'plots/document-medical-pct-hist-{args.filter_dir}.png'
p_hist.save(output_path_hist, dpi=300, width=3.375, height=2.5)
p_hist.save(output_path_hist.replace('.png', '.svg'), dpi=300, width=3.375, height=2.5)
p_hist.save(output_path_hist.replace('.png', '.pdf'), dpi=300, width=3.375, height=2.5)
print(f"saved histogram to {output_path_hist}")

