""""
evaluate label sweep on test sets and create scaling law plot
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
from colors import MASK_COLORS, THEME_COLORS
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use Helvetica Neue for mathtext
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica Neue:bold'
plt.rcParams['mathtext.sf'] = 'Helvetica Neue'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=str, default='cuda')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'))
parser.add_argument('--label_sweep_path', type=str, default=os.path.join(MODEL_PATH, 'label-sweep'))
parser.add_argument('--data_path',  type=str, default=os.path.join(DATA_PATH, 'test'))
parser.add_argument('--eval_iters', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--block_size', type=int, default=2048)
parser.add_argument('--rerun', action='store_true', help='rerun results')
args = parser.parse_args()

device = args.device
data_dir = args.data_path
model_path = args.model_path
label_sweep_path = args.label_sweep_path
eval_iters = args.eval_iters
batch_size = args.batch_size
block_size = args.block_size

# determine the device type
device_type = 'cuda' if 'cuda' in device else 'cpu'

# collect models from label-sweep directory
models = []
for file in os.listdir("../config/label-sweep"):
    if file.endswith('.yaml'):
        # e.g., gpt-113M-document.yaml -> document-113M.pt
        parts = file.replace('.yaml', '').split('-')
        size = parts[1]
        label_type = parts[2]
        models.append({
            'file': f"{label_type}-{size}.pt",
            'path': label_sweep_path,
            'mask_type': label_type
        })

# collect mask and nomask models from scaling directory
for file in os.listdir("../config/scaling"):
    if '1816M' in file or '1030M' in file:
        continue
    if file.endswith('.yaml') and ('-mask.yaml' in file or '-nomask.yaml' in file):
        # e.g., gpt-113M-mask.yaml -> mask-113M.pt
        parts = file.replace('.yaml', '').split('-')
        size = parts[1]
        mask_type = parts[2]
        # rename 'mask' to 'token'
        display_type = 'token' if mask_type == 'mask' else mask_type
        models.append({
            'file': f"{mask_type}-{size}.pt",
            'path': model_path,
            'mask_type': display_type
        })

def load_model(model_file):
    checkpoint = torch.load(model_file, map_location=device)
    model_args = checkpoint['model_args']

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    return model

loader_dtype = np.uint32
def get_test_batch(batch_size=32):

    test_target = np.memmap(os.path.join(data_dir, 'test_target.bin'), dtype=loader_dtype, mode='r')
    test_ood = np.memmap(os.path.join(data_dir, 'test_ood.bin'), dtype=loader_dtype, mode='r')
    test_parallel = np.memmap(os.path.join(data_dir, 'test_parallel.bin'), dtype=loader_dtype, mode='r')
    test_parallel_hard = np.memmap(os.path.join(data_dir, 'test_parallel_hard.bin'), dtype=loader_dtype, mode='r')

    ix = torch.randint(len(test_target) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((test_target[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((test_target[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_ood) - block_size, (batch_size,))
    x_ood = torch.stack([torch.from_numpy((test_ood[i:i+block_size]).astype(np.int64)) for i in ix])
    y_ood = torch.stack([torch.from_numpy((test_ood[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_parallel) - block_size, (batch_size,))
    x_parallel = torch.stack([torch.from_numpy((test_parallel[i:i+block_size]).astype(np.int64)) for i in ix])
    y_parallel = torch.stack([torch.from_numpy((test_parallel[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_parallel_hard) - block_size, (batch_size,))
    x_parallel_hard = torch.stack([torch.from_numpy((test_parallel_hard[i:i+block_size]).astype(np.int64)) for i in ix])
    y_parallel_hard = torch.stack([torch.from_numpy((test_parallel_hard[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        x_ood, y_ood = x_ood.pin_memory().to(device, non_blocking=True), y_ood.pin_memory().to(device, non_blocking=True)
        x_parallel, y_parallel = x_parallel.pin_memory().to(device, non_blocking=True), y_parallel.pin_memory().to(device, non_blocking=True)
        x_parallel_hard, y_parallel_hard = x_parallel_hard.pin_memory().to(device, non_blocking=True), y_parallel_hard.pin_memory().to(device, non_blocking=True)

    else:
        x, y = x.to(device), y.to(device)
        x_ood, y_ood = x_ood.to(device), y_ood.to(device)
        x_parallel, y_parallel = x_parallel.to(device), y_parallel.to(device)
        x_parallel_hard, y_parallel_hard = x_parallel_hard.to(device), y_parallel_hard.to(device)

    return x, x_ood, y, y_ood, x_parallel, y_parallel, x_parallel_hard, y_parallel_hard

@torch.no_grad()
def estimate_test_loss(model, batch_size=32, eval_iters=10):

    out = {'target' : torch.zeros(eval_iters), 'ood' : torch.zeros(eval_iters), 'parallel' : torch.zeros(eval_iters), 'parallel_hard' : torch.zeros(eval_iters)}
    model.eval()

    for k in range(eval_iters):

        X, X_ood, Y, Y_ood, X_parallel, Y_parallel, X_parallel_hard, Y_parallel_hard = get_test_batch(batch_size=batch_size)

        with torch.no_grad():

            _, loss = model(X, idx_filter=None, targets=Y, targets_filter=None)
            out['target'][k] = loss.item()
        
            _, loss = model(X_ood, idx_filter=None, targets=Y_ood, targets_filter=None)
            out['ood'][k] = loss.item()

            _, loss = model(X_parallel, idx_filter=None, targets=Y_parallel, targets_filter=None)
            out['parallel'][k] = loss.item()

            _, loss = model(X_parallel_hard, idx_filter=None, targets=Y_parallel_hard, targets_filter=None)
            out['parallel_hard'][k] = loss.item()
    
    return {'target' : out['target'].mean(), 'ood' : out['ood'].mean(), 'parallel' : out['parallel'].mean(), 'parallel_hard' : out['parallel_hard'].mean()}

if 'label-sweep-scaling.csv' not in os.listdir('results') or args.rerun:

    df = pd.DataFrame(columns=['params', 'mask', 'target', 'ood', 'parallel', 'parallel_hard'])
    for model_info in models:

        model_file = os.path.join(model_info['path'], model_info['file'])
        
        try:
            model = load_model(model_file)
            print(f"loaded model {model_info['file']}")
        
        except Exception as e:
            print(f"error loading model {model_info['file']}: {e}")
            continue
        
        model.eval()
        model.to(device)
        
        if '1030M' in model_info['file'] or '1816M' in model_info['file']:
            loss = estimate_test_loss(model, batch_size=8, eval_iters=40)
        else:
            loss = estimate_test_loss(model, batch_size=16, eval_iters=20)
        
        # extract params from filename
        if 'token' in model_info['mask_type'] or 'nomask' in model_info['mask_type']:
            # scaling format: mask-113M.pt
            params = int(model_info['file'].split('-')[1].split('.')[0][:-1]) * 1e6
        else:
            # label-sweep format: document-113M.pt
            params = int(model_info['file'].split('-')[1].split('.')[0][:-1]) * 1e6
        
        df.loc[len(df)] = {
            'params'   : params,
            'mask'     : model_info['mask_type'],
            'target'   : loss['target'].item(),
            'ood'      : loss['ood'].item(),
            'parallel' : loss['parallel'].item(),
            'parallel_hard' : loss['parallel_hard'].item()
        }

        print(f"model {model_info['file']} ({model_info['mask_type']}) done | target {loss['target']:.4f} | ood {loss['ood']:.4f}")
        
        # free GPU memory
        del model
        torch.cuda.empty_cache()
        
    df.to_csv('results/label-sweep-scaling.csv', index=False)
else:
    df = pd.read_csv('results/label-sweep-scaling.csv')

# rename mask to label_type
df = df.rename(columns={'mask': 'label_type'})

# Filter out nomask
df = df[df['label_type'] != 'nomask'].copy()

df['params'] = pd.to_numeric(df['params'])

# Compute average_parallel (biology)
df['average_parallel'] = (df['parallel'] + df['parallel_hard']) / 2

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']

# Label type display names (raw label_type -> display name)
label_type_names = {
    'token': 'Token',
    'sentence': 'Sentence',
    'token-with-sent-labels': 'Sentence',
    'document': 'Document',
    'token-with-doc-labels': 'Document'
}

# Map display names to colors
label_display_colors = {
    'Token': MASK_COLORS['mask'],
    'Sentence': '#CC978E',
    'Document': '#B5CBB7'
}

# Loss frontiers plot: x = non-medical/bio, y = medical
frontier_data = []
for _, row in df.iterrows():
    for loss_type in ['target', 'average_parallel']:
        frontier_data.append({
            'params': row['params'],
            'label_type': row['label_type'],
            'ood_loss': row['ood'],
            'x_loss': row[loss_type],
            'loss_type': loss_type
        })

frontier_df = pd.DataFrame(frontier_data)

# Map label types to display names
frontier_df['label_display'] = frontier_df['label_type'].map(label_type_names)

# Set order: Token, Sentence, Document
label_order = ['Token', 'Sentence', 'Document']
frontier_df['label_display'] = pd.Categorical(frontier_df['label_display'], categories=label_order, ordered=True)

# Facet labels with down arrows (latex)
x_loss_labels = {
    'target': r'Non-Medical Loss ($\downarrow$)',
    'average_parallel': r'Biology Loss ($\downarrow$)'
}
frontier_df['loss_type_label'] = frontier_df['loss_type'].map(x_loss_labels)

facet_order = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']
frontier_df['loss_type_label'] = pd.Categorical(frontier_df['loss_type_label'], categories=facet_order, ordered=True)

# Get colors in display order
label_colors = [label_display_colors[l] for l in label_order]

# Facet labels to use as x-axis titles (in facet order)
facet_x_titles = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']

p_frontier = (ggplot(frontier_df, aes(x='x_loss', y='ood_loss', color='label_display'))
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    + facet_wrap('~loss_type_label', ncol=2, scales='free_x')
    + scale_x_log10(name='')
    + scale_y_log10(name=r'Medical Loss ($\uparrow$)')
    + scale_color_manual(values=label_colors, name='Label Granularity')
    + guides(color=guide_legend(nrow=1))
    + theme_bw(base_family='Helvetica Neue')
    + theme(figure_size=(3.375, 2.34),
            strip_text=element_blank(),
            strip_background=element_blank(),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_text(size=7, color=text_color),
            legend_position='top',
            legend_direction='horizontal',
            axis_title_x=element_blank(),
            axis_text_x=element_text(size=7, color=text_color),
            axis_title_y=element_text(size=9, color=text_color),
            axis_text_y=element_text(size=7, color=text_color),
            legend_text=element_text(size=7, color=text_color),
            legend_key_size=7,
            plot_background=element_rect(fill=bg_color),
            panel_background=element_rect(fill=bg_color),
            panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
            legend_background=element_rect(fill=bg_color))
)

# Draw the plot and add per-facet x-axis labels using matplotlib
fig = p_frontier.draw()
for ax, title in zip(fig.axes[:2], facet_x_titles):
    ax.set_xlabel(title, fontsize=9, color=text_color, fontfamily='Helvetica Neue')
fig.savefig('plots/label-sweep-loss-frontiers.png', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/label-sweep-loss-frontiers.svg', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/label-sweep-loss-frontiers.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print("Saved plots/label-sweep-loss-frontiers.{png,svg,pdf}")

