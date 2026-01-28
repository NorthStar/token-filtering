"""
Plot refusal evaluation results from eval-refusals.py

Creates a bar plot showing refusal rates for each mask type,
with dodge positioning for alpaca vs. healthsearchqa datasets.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from plotnine import *
from colors import MASK_COLORS, MASK_LABELS, get_mask_color_list, THEME_COLORS
import matplotlib.pyplot as plt

# Configure matplotlib to use Helvetica Neue for mathtext
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica Neue:bold'
plt.rcParams['mathtext.sf'] = 'Helvetica Neue'

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='results/refusal-graded')
parser.add_argument('--refusal_token', action='store_true', help='use refusal-graded-token directory')
args = parser.parse_args()

# Update input_dir if refusal_token is set and input_dir is default
if args.refusal_token and args.input_dir == 'results/refusal-graded':
    args.input_dir = 'results/refusal-graded-token'

# Mask type labels (imported from colors.py)
# MASK_LABELS imported at top of file

# Dataset labels with arrows
DATASET_LABELS = {
    'alpaca': r'Alpaca ($\downarrow$)',
    'healthsearchqa': r'HealthSearchQA ($\uparrow$)',
}

def load_graded_results(input_dir):
    """Load all graded JSONL files and combine into a dataframe."""
    all_data = []
    
    # Find all JSONL files in the directory
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    for filename in jsonl_files:
        # Extract mask type and dataset from filename (e.g., 'mask-alpaca.jsonl')
        parts = filename.replace('.jsonl', '').split('-')
        mask_type = parts[0]
        dataset = parts[1] if len(parts) > 1 else 'unknown'
        
        filepath = os.path.join(input_dir, filename)
        print(f"Loading {filepath}...")
        
        with open(filepath, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['mask'] = mask_type
                # Use dataset from filename if not in item
                if 'dataset' not in item:
                    item['dataset'] = dataset
                all_data.append(item)
    
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} total items from {len(jsonl_files)} files")
    
    return df

def calculate_refusal_rates(df):
    """Calculate refusal rate for each mask type and dataset.
    
    Also calculates standard error: SE = sqrt(p * (1-p) / n)
    """
    results = []
    
    # Group by mask type and dataset
    for mask in df['mask'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['mask'] == mask) & (df['dataset'] == dataset)]
            
            if len(subset) == 0:
                continue
            
            # Calculate refusal rate
            refusal_rate = subset['refusal'].mean()
            n = len(subset)
            # Standard error of proportion
            se = np.sqrt(refusal_rate * (1 - refusal_rate) / n) if n > 0 else 0
            
            results.append({
                'mask': mask,
                'dataset': dataset,
                'refusal_rate': refusal_rate,
                'se': se,
                'se_lower': refusal_rate - se,
                'se_upper': refusal_rate + se,
                'n_samples': n
            })
    
    return pd.DataFrame(results)

# Load and process data
print(f"Loading data from {args.input_dir}...")
df = load_graded_results(args.input_dir)

print("\nCalculating refusal rates...")
df_plot = calculate_refusal_rates(df)

# Convert refusal rate to percentage
df_plot['refusal_rate'] = df_plot['refusal_rate'] * 100
df_plot['se_lower'] = df_plot['se_lower'] * 100
df_plot['se_upper'] = df_plot['se_upper'] * 100

# Filter to only the 4 mask types we want
mask_order = ['nomask', 'document', 'mask', 'remove']
df_plot = df_plot[df_plot['mask'].isin(mask_order)].copy()

# Apply labels
df_plot['mask'] = df_plot['mask'].replace(MASK_LABELS)
df_plot['mask'] = pd.Categorical(
    df_plot['mask'],
    categories=[MASK_LABELS[m] for m in mask_order],
    ordered=True
)
df_plot['dataset'] = pd.Categorical(
    df_plot['dataset'].replace(DATASET_LABELS),
    categories=list(DATASET_LABELS.values()),
    ordered=True
)

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']
base_theme = theme_bw

# Prepare colors - use centralized mask colors
hex_colors = get_mask_color_list(mask_order)

# Print summary statistics
print("\nSummary by mask type and dataset:")
print(df_plot[['mask', 'dataset', 'refusal_rate', 'n_samples']])

# Create plot
print("\nCreating plot...")
os.makedirs('plots', exist_ok=True)

dodge_pos = position_dodge(width=0.8)
p = (
    ggplot(df_plot, aes(x='dataset', y='refusal_rate', fill='mask'))
        + geom_col(position=dodge_pos, width=0.7, color=line_color, size=0.3)
        + scale_y_continuous(name='Refusal Rate (%)')
        + scale_x_discrete(name='')
        + scale_fill_manual(values=hex_colors)
        + guides(fill=guide_legend(nrow=1))
        + base_theme(base_family='Helvetica Neue')
        + theme(
            figure_size=(3.375, 2.5),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            axis_title_x=element_text(size=9, color=text_color),
            axis_text_x=element_text(size=7, rotation=0, color=text_color),
            axis_title_y=element_text(size=9, color=text_color),
            axis_text_y=element_text(size=7, color=text_color),
            legend_text=element_text(size=7, color=text_color),
            legend_key_size=7,
            strip_text=element_text(size=9, color=text_color),
            legend_position='top',
            legend_direction='horizontal',
            plot_background=element_rect(fill=bg_color),
            panel_background=element_rect(fill=bg_color),
            panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
            legend_background=element_rect(fill=bg_color)
        )
)

output_suffix = '-token' if args.refusal_token else ''
p.save(f'plots/refusal-rates{output_suffix}.png', dpi=300, width=3.375, height=2.5)
p.save(f'plots/refusal-rates{output_suffix}.svg', dpi=300, width=3.375, height=2.5)
p.save(f'plots/refusal-rates{output_suffix}.pdf', dpi=300, width=3.375, height=2.5)

print(f"\nPlot saved as 'plots/refusal-rates{output_suffix}.png', 'plots/refusal-rates{output_suffix}.svg', and 'plots/refusal-rates{output_suffix}.pdf'")
print(p)
