import os
import numpy as np
import pandas as pd
import argparse
from plotnine import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../../../workspace/med/data/filtered-224M')
parser.add_argument('--n_components', type=int, default=6, help='number of equal-sized components to split the distribution into')
args = parser.parse_args()

arr = np.memmap(os.path.join(args.data_path, 'val_filter.bin'), dtype=np.float16, mode='r')
df = pd.DataFrame({'predictions': arr.astype(np.float32)})

print(len(df[df['predictions'] > 0.5]), len(df), len(df[df['predictions'] > 0.5]) / len(df))

# compute thresholds for N log-spaced components starting at 0.5
n = args.n_components
# Create log intervals: 0.5, 0.25, 0.125, 0.0625, etc.
log_positions = 1 - 0.5 * (0.5 ** np.arange(n-1))  # [0.5, 0.25, 0.125, ...]
thresholds = np.quantile(df['predictions'], log_positions)

threshold_df = pd.DataFrame({'threshold': thresholds})
threshold_df.to_csv('../config/thresholds.csv', index=False)

# plot = (
#     ggplot(df, aes(x='predictions'))
#      + geom_histogram(aes(y=after_stat('density')), bins=100)
#      + geom_vline(data=threshold_df, mapping=aes(xintercept='threshold'), 
#                   color='black', linetype='dashed', size=0.8, alpha=0.7)
#      + scale_x_continuous(name='probe p(medical)')
#      + scale_y_continuous(name='density')
#      + theme_bw(base_family='Palatino')
#      + theme(
#         figure_size=(12, 8),
#         panel_grid_major=element_line(size=0.3, color="#dddddd"),
#         panel_grid_minor=element_blank(),
#         legend_title=element_blank())
# )

# plot.save('plots/train_stats.png', dpi=300, width=12, height=8)