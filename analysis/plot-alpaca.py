"""
Plot evaluation results from eval-alpaca.py

Creates a faceted bar plot showing the proportion of YES responses for each
evaluation criterion, grouped by mask type.
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
parser.add_argument('--input_dir', type=str, default='results/alpaca-graded-filtered')
args = parser.parse_args()

# Evaluation criteria from eval-alpaca.py
CRITERIA = [
    "correctness",
    "coherence",
    "relevance",
]

# Pretty labels for criteria
CRITERIA_LABELS = {
    "correctness": "Correctness",
    "coherence": "Coherence",
    "relevance": "Relevance"
}

# Mask type labels (imported from colors.py)
# MASK_LABELS imported at top of file

def load_graded_results(input_dir):
    """Load all graded JSONL files and combine into a dataframe."""
    all_data = []

    # Find all JSONL files in the directory
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]

    for filename in jsonl_files:
        # Extract mask type from filename (e.g., 'mask.jsonl' -> 'mask')
        mask_type = filename.replace('.jsonl', '')

        filepath = os.path.join(input_dir, filename)
        print(f"Loading {filepath}...")

        with open(filepath, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['mask'] = mask_type
                all_data.append(item)

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} total items from {len(jsonl_files)} files")

    return df

def calculate_proportions(df):
    """Calculate proportion of YES (1) for each criterion by mask type and dataset.

    Also calculates standard error: SE = sqrt(p * (1-p) / n)
    """
    results = []

    # Group by mask type and dataset
    for mask in df['mask'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['mask'] == mask) & (df['dataset'] == dataset)]

            if len(subset) == 0:
                continue

            # Calculate proportion for each criterion
            for criterion in CRITERIA:
                if criterion in subset.columns:
                    proportion = subset[criterion].mean()
                    n = len(subset)
                    # Standard error of proportion
                    se = np.sqrt(proportion * (1 - proportion) / n) if n > 0 else 0

                    results.append({
                        'mask': mask,
                        'dataset': dataset,
                        'criterion': criterion,
                        'proportion': proportion,
                        'se': se,
                        'se_lower': proportion - se,
                        'se_upper': proportion + se,
                        'n_samples': n
                    })

    return pd.DataFrame(results)

# Load and process data
print(f"Loading data from {args.input_dir}...")
df = load_graded_results(args.input_dir)

print("\nCalculating proportions across all data...")
all_results = []
for mask in df['mask'].unique():
    subset = df[df['mask'] == mask]

    for criterion in CRITERIA:
        if criterion in subset.columns:
            proportion = subset[criterion].mean()
            n = len(subset)
            # Standard error of proportion
            se = np.sqrt(proportion * (1 - proportion) / n) if n > 0 else 0

            all_results.append({
                'mask': mask,
                'criterion': criterion,
                'proportion': proportion,
                'se': se,
                'se_lower': proportion - se,
                'se_upper': proportion + se,
                'n_samples': n
            })

df_plot = pd.DataFrame(all_results)

# Convert proportion to percentage
df_plot['proportion'] = df_plot['proportion'] * 100
df_plot['se_lower'] = df_plot['se_lower'] * 100
df_plot['se_upper'] = df_plot['se_upper'] * 100

# Filter to only the 4 mask types we want
mask_order = ['nomask', 'document', 'mask', 'remove']
df_plot = df_plot[df_plot['mask'].isin(mask_order)].copy()

# Apply labels
df_plot['criterion'] = pd.Categorical(
    df_plot['criterion'].replace(CRITERIA_LABELS),
    categories=[CRITERIA_LABELS[c] for c in CRITERIA],
    ordered=True
)
df_plot['mask'] = df_plot['mask'].replace(MASK_LABELS)
df_plot['mask'] = pd.Categorical(
    df_plot['mask'],
    categories=[MASK_LABELS[m] for m in mask_order],
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
print("\nSummary by mask type:")
print(df_plot.groupby(['mask'])['proportion'].mean())

# Create plot
print("\nCreating plot...")
os.makedirs('plots', exist_ok=True)

# ============================================================================
# Plot 1: Absolute proportions
# ============================================================================
dodge_pos = position_dodge(width=0.8)
p1 = (
    ggplot(df_plot, aes(x='criterion', y='proportion', fill='mask'))
        + geom_col(position=dodge_pos, width=0.7, color=line_color, size=0.3)
        + scale_y_continuous(name='Proportion of Generations (%)')
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

p1.save('plots/alpaca-evaluation.png', dpi=300, width=3.375, height=2.5)
p1.save('plots/alpaca-evaluation.svg', dpi=300, width=3.375, height=2.5)
p1.save('plots/alpaca-evaluation.pdf', dpi=300, width=3.375, height=2.5)

print(f"\nPlot 1 saved as 'plots/alpaca-evaluation.png', 'plots/alpaca-evaluation.svg', and 'plots/alpaca-evaluation.pdf'")
print(p1)

# ============================================================================
# Plot 2: Relative to nomask baseline
# ============================================================================
print("\nCalculating relative performance (vs. nomask)...")

# Get nomask baseline values
nomask_baseline = df_plot[df_plot['mask'] == 'No Filtering'][['criterion', 'proportion', 'se']].copy()
nomask_baseline.columns = ['criterion', 'baseline_prop', 'baseline_se']

# Filter to only document and mask
df_relative = df_plot[df_plot['mask'].isin(['Document-level Removal', 'Token-level Loss Masking'])].copy()

# Merge with baseline
df_relative = df_relative.merge(nomask_baseline, on='criterion')

# Calculate relative performance (ratio)
df_relative['relative'] = df_relative['proportion'] / df_relative['baseline_prop']

# Calculate SE for ratio using delta method: SE(p1/p2) ≈ (p1/p2) * sqrt((SE(p1)/p1)^2 + (SE(p2)/p2)^2)
# Avoid division by zero
def safe_relative_se(row):
    if row['proportion'] == 0 or row['baseline_prop'] == 0:
        return 0
    se_ratio_sq = (row['se'] / row['proportion'])**2 + (row['baseline_se'] / row['baseline_prop'])**2
    return row['relative'] * np.sqrt(se_ratio_sq)

df_relative['relative_se'] = df_relative.apply(safe_relative_se, axis=1)
df_relative['relative_se_lower'] = df_relative['relative'] - df_relative['relative_se']
df_relative['relative_se_upper'] = df_relative['relative'] + df_relative['relative_se']

dodge_pos = position_dodge(width=0.8)
p2 = (
    ggplot(df_relative, aes(x='criterion', y='relative', fill='mask'))
        + geom_col(position=dodge_pos, width=0.7, color=line_color, size=0.3)
        + geom_hline(yintercept=1.0, color=line_color, linetype='dashed', size=0.5)
        + scale_y_continuous(name='Relative Performance (vs. No Filtering)')
        + scale_x_discrete(name='')
        + scale_fill_manual(values=[hex_colors[1], hex_colors[2]])  # Only document and mask colors
        + guides(fill=guide_legend(nrow=1))
        + base_theme(base_family='Helvetica Neue')
        + theme(
            figure_size=(7, 2.625),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            axis_title_x=element_text(size=9, color=text_color),
            axis_text_x=element_text(size=7, rotation=0, color=text_color),
            axis_title_y=element_text(size=9, color=text_color),
            axis_text_y=element_text(size=7, color=text_color),
            legend_text=element_text(size=7, color=text_color),
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

p2.save('plots/alpaca-evaluation-relative.png', dpi=300, width=7, height=2.625)
p2.save('plots/alpaca-evaluation-relative.svg', dpi=300, width=7, height=2.625)
p2.save('plots/alpaca-evaluation-relative.pdf', dpi=300, width=7, height=2.625)

print(f"\nPlot 2 saved as 'plots/alpaca-evaluation-relative.png', 'plots/alpaca-evaluation-relative.svg', and 'plots/alpaca-evaluation-relative.pdf'")
print(p2)
