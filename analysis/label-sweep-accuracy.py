"""
Plot token-level accuracy vs label granularity for the largest classifier.
"""

import pandas as pd
from plotnine import *
from colors import MASK_COLORS, THEME_COLORS

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']

# Load probe label scaling data
df = pd.read_csv('results/probe-label-scaling.csv')

# Filter to largest classifier (224M)
df = df[df['size'] == '224M'].copy()
print(f"Filtered to 224M classifier")

# Filter to token-level evaluation and F1 metric
df = df[(df['evaluated_on'] == 'token') & (df['metric'] == 'f1')].copy()

# Rename for clarity
df = df.rename(columns={'probe_trained_on': 'granularity', 'score': 'f1'})

# Map granularity to display names
granularity_names = {
    'token': 'Token',
    'sentence': 'Sentence',
    'document': 'Document'
}
df['granularity_label'] = df['granularity'].map(granularity_names)

# Set order for x-axis
granularity_order = ['Token', 'Sentence', 'Document']
df['granularity_label'] = pd.Categorical(
    df['granularity_label'],
    categories=granularity_order,
    ordered=True
)

# Get colors
colors = [MASK_COLORS['mask'], '#CC978E', '#B5CBB7']

print(df[['granularity_label', 'f1']])

# Create plot
p = (ggplot(df, aes(x='granularity_label', y='f1', fill='granularity_label'))
    + geom_bar(stat='identity', width=0.7, color=line_color, size=0.3)
    + scale_fill_manual(values=colors)
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
        legend_position='none'
    )
    + labs(x='Classifier Training Granularity', y='F1 on Token Ground Truth Labels')
    + coord_cartesian(ylim=(0, 1))
)

p.save('plots/label-sweep-accuracy.png', dpi=300, width=3.375, height=2.5)
p.save('plots/label-sweep-accuracy.svg', dpi=300, width=3.375, height=2.5)
p.save('plots/label-sweep-accuracy.pdf', dpi=300, width=3.375, height=2.5)
print("Saved plots/label-sweep-accuracy.{png,svg,pdf}")
