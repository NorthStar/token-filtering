"""
saves best lr, beta2, w_decay for each n_layer
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from plotnine import *
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'adamw-sweep'))
args = parser.parse_args()

model_path = args.model_path

models = {
    'file' : [],
    'n_layer' : [],
    'lr' : [],
    'w_decay' : [],
    'loss' : []
}

def get_loss(model_file):
    checkpoint = torch.load(model_file)
    return checkpoint['best_val_loss']

for file in os.listdir("../config/adamw-sweep"):
    
    # gpt-l-<n_layer>-lr-<lr>-wdecay-<w_decay>.yaml
    _, _, n_layer, _, lr, _, w_decay = file.split('-')
    w_decay = float('.'.join(w_decay.split('.')[:-1]))
    model_file = '.'.join(('-'.join(file.split('-')[1:])).split('.')[:-1]) + '.pt'
    
    try:
        models['loss'].append(get_loss(os.path.join(model_path, model_file)).item())
    except Exception as e:
        print(f"error loading model {model_file}: {e}")
        continue
    
    models['file'].append(model_file)
    models['n_layer'].append(int(n_layer))
    models['lr'].append(float(lr))
    models['w_decay'].append(w_decay)

best_params = {
    'lr' : [],
    'w_decay' : []
}

df = pd.DataFrame(models)

best_results = []
for n_layer in df['n_layer'].unique():
    subset = df[df['n_layer'] == n_layer]
    
    if len(subset) > 0:

        best_row = subset.loc[subset['loss'].idxmin()]

        best_results.append({
            'n_layer': n_layer,
            'best_lr': best_row['lr'],
            'best_w_decay': best_row['w_decay'],
            'best_loss': best_row['loss'],
            'best_file': best_row['file']
        })

best_df = pd.DataFrame(best_results)
best_df = best_df.sort_values('n_layer')

print("best hyperparameters:")
print("-" * 20)
print(best_df.to_string(index=False))
best_df.to_csv('../config/adamw-hparams.csv', index=False)

p = (ggplot(df, aes(x='lr', y='loss', color='w_decay', group='w_decay')) +
     geom_point() +
     geom_line() +
     facet_wrap('n_layer') +
     scale_x_log10(name='adamw learning rate') +
     scale_color_cmap(name='adamw weight decay') +
     scale_y_log10(name='loss') +
     theme_bw(base_family='Palatino') +
     theme(
        figure_size=(7, 4),
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_position='right',
        strip_text=element_text(size=12),
        strip_background=element_blank()
    )
)

p.save('plots/adamw-sweep-lr.png', dpi=300, bbox_inches='tight')

p = (ggplot(df, aes(x='w_decay', y='loss', color='lr', group='lr')) +
     geom_point() +
     geom_line() +
     facet_wrap('n_layer') +
     scale_x_continuous(name='adamw weight decay') +
     xlim(-0.1, 0.11) +
     scale_y_log10(name='loss') +
     scale_color_cmap(trans='log10', name='adamw learning rate') +
     theme_bw(base_family='Palatino') +
     theme(
        figure_size=(7, 4),
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_position='right',
        strip_text=element_text(size=12),
        strip_background=element_blank()
    )
)
p.save('plots/adamw-sweep-wdecay.png', dpi=300, bbox_inches='tight')