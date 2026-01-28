"""
Evaluate delayed-filter sweep on test sets and create scaling law plots.
Models are trained with filtering starting at different proportions of training.
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
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
from colors import get_delayed_colors, MASK_COLORS, THEME_COLORS

# ============================================================================
# Setup and Configuration
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=str, default='cuda')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'delayed-filter'))
parser.add_argument('--baseline_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'))
parser.add_argument('--data_path',  type=str, default=os.path.join(DATA_PATH, 'filtered-224M'))
parser.add_argument('--eval_iters', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--block_size', type=int, default=1024)
parser.add_argument('--rerun', action='store_true', help='rerun model evaluation')
args = parser.parse_args()

device = args.device
data_dir = args.data_path
model_path = args.model_path
eval_iters = args.eval_iters
batch_size = args.batch_size
block_size = args.block_size

device_type = 'cuda' if 'cuda' in device else 'cpu'

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']
base_theme = theme_bw

# Model sizes to exclude from plots
excluded_params = [13e6, 28e6, 1030e6, 1816e6]

# ============================================================================
# Model Discovery
# ============================================================================

# begin_filter_steps from config/delayed-filter-sweep.py
begin_filter_props = [0, 0.2, 0.4, 0.6, 0.8]

models = []
for file in os.listdir("../config/delayed-filter"):
    if file.endswith('.yaml'):
        # Files are named like: gpt-{params}M-{begin_filter_step}.yaml
        # Model files will be: mask-{params}M-{begin_filter_step}.pt
        parts = file.replace('.yaml', '').split('-')
        params = parts[1]  # e.g., '224M'
        begin_step = parts[2]  # e.g., '0', '20', '40', etc.
        models.append(f'mask-{params}-{begin_step}.pt')

baseline_models = []
for file in os.listdir("../config/scaling"):
    if '1030M' in file:
        continue
    if file.endswith('.yaml') and 'nomask' in file:
        baseline_models.append(file.split('-')[2].split('.')[0] + '-' + file.split('-')[1] + '.pt')

# ============================================================================
# Model Loading and Evaluation Functions
# ============================================================================

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
def get_test_batch():

    test_target = np.memmap(os.path.join(data_dir, 'test_target_true.bin'), dtype=loader_dtype, mode='r')
    test_ood = np.memmap(os.path.join(data_dir, 'test_ood_true.bin'), dtype=loader_dtype, mode='r')
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
def estimate_test_loss(model):

    out = {'target' : torch.zeros(eval_iters), 'ood' : torch.zeros(eval_iters), 'parallel' : torch.zeros(eval_iters), 'parallel_hard' : torch.zeros(eval_iters)}
    model.eval()

    for k in range(eval_iters):

        X, X_ood, Y, Y_ood, X_parallel, Y_parallel, X_parallel_hard, Y_parallel_hard = get_test_batch()

        with torch.no_grad():

            _, loss = model(X, idx_filter=None, targets=Y, targets_filter=None)
            out['target'][k] = loss.item()

            _, loss = model(X_ood, idx_filter=None, targets=Y_ood, targets_filter=None)
            out['ood'][k] = loss.item()

            _, loss = model(X_parallel, idx_filter=None, targets=Y_parallel, targets_filter=None)
            out['parallel'][k] = loss.item()

            _, loss = model(X_parallel_hard, idx_filter=None, targets=Y_parallel_hard, targets_filter=None)
            out['parallel_hard'][k] = loss.item()

    return {k: v.mean() for k, v in out.items()}

# ============================================================================
# Evaluate Models and Save Results
# ============================================================================

# Create mapping from begin_filter_step (0, 20, 40, 60, 80) to proportion (0, 0.2, 0.4, 0.6, 0.8)
step_to_prop = {int(p * 100): p for p in begin_filter_props}

if 'delayed-scaling.csv' not in os.listdir('results') or args.rerun:

    df = pd.DataFrame(columns=['params', 'begin_filter_prop', 'target', 'ood', 'parallel', 'parallel_hard'])

    for model_file in models:

        try:
            model = load_model(os.path.join(model_path, model_file))

        except Exception as e:
            print(f"error loading model {model_file}: {e}")
            continue

        print(f"evaluating {model_file}...")
        model.eval()
        model.to(device)

        loss = estimate_test_loss(model)

        # Parse model file: mask-{params}M-{begin_filter_step}.pt
        parts = model_file.replace('.pt', '').split('-')
        params_str = parts[1]  # e.g., '224M'
        begin_step = int(parts[2])  # e.g., 0, 20, 40, 60, 80

        df.loc[len(df)] = {
            'params'   : int(params_str[:-1]) * 1e6,
            'begin_filter_prop' : step_to_prop.get(begin_step, begin_step / 100),
            'target'   : loss['target'].item(),
            'ood'      : loss['ood'].item(),
            'parallel' : loss['parallel'].item(),
            'parallel_hard' : loss['parallel_hard'].item()
        }

    for model_file in baseline_models:

        try:
            model = load_model(os.path.join(args.baseline_path, model_file))

        except Exception as e:
            print(f"error loading baseline model {model_file}: {e}")
            continue

        model.eval()
        model.to(device)

        loss = estimate_test_loss(model)

        df.loc[len(df)] = {
            'params'   : int(model_file.split('-')[1].split('.')[0][:-1]) * 1e6,
            'begin_filter_prop' : 1.0,  # Baseline = no filtering (filter starts after 100% of training)
            'target'   : loss['target'].item(),
            'ood'      : loss['ood'].item(),
            'parallel' : loss['parallel'].item(),
            'parallel_hard' : loss['parallel_hard'].item()
        }

    df.to_csv('results/delayed-scaling.csv', index=False)
else:
    df = pd.read_csv('results/delayed-scaling.csv')

print(f"\nLoaded {len(df)} model results")

# ============================================================================
# Data Preparation
# ============================================================================

df['params'] = pd.to_numeric(df['params'])
df['compute'] = 6 * df['params'] * 20 * df['params']

# Compute average_parallel as average of parallel and parallel_hard
df['average_parallel'] = (df['parallel'] + df['parallel_hard']) / 2

# Filter out 13M and 28M models
df = df[~df['params'].isin(excluded_params)].copy()

# Create labels for begin_filter_prop
begin_filter_labels = {p: f'{int(p * 100)}%' for p in begin_filter_props}
begin_filter_labels[1.0] = 'Baseline'

# Order delay levels for consistent coloring (0% first, then increasing)
unique_props = sorted(df['begin_filter_prop'].unique())
delay_labels = [begin_filter_labels.get(p, f'{int(p*100)}%') for p in unique_props]

# Setup color palette - use GnBu for delay levels
# Exclude baseline from GnBu colors, use nomask color for it
delay_labels_no_baseline = [l for l in delay_labels if l != 'Baseline']
gnbu_colors = get_delayed_colors(len(delay_labels_no_baseline))

# Create color mapping, overriding "Baseline" with nomask color
hex_colors = {'Baseline': MASK_COLORS['nomask']}
for i, label in enumerate(delay_labels_no_baseline):
    hex_colors[label] = gnbu_colors[i]

# ============================================================================
# Loss Frontiers Plot
# ============================================================================

frontier_data = []
for _, row in df.iterrows():
    for loss_type in ['target', 'average_parallel']:
        frontier_data.append({
            'params': row['params'],
            'compute': row['compute'],
            'begin_filter_prop': row['begin_filter_prop'],
            'ood_loss': row['ood'],
            'x_loss': row[loss_type],
            'loss_type': loss_type
        })

frontier_df = pd.DataFrame(frontier_data)

# Apply labels for loss types
x_loss_labels = {
    'target': r'Non-Medical Loss ($\downarrow$)',
    'average_parallel': r'Biology Loss ($\downarrow$)'
}
frontier_df['loss_type_label'] = frontier_df['loss_type'].map(x_loss_labels)

# Force the order of facets by loss type
x_facet_order = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']
frontier_df['loss_type_label'] = pd.Categorical(frontier_df['loss_type_label'], categories=x_facet_order, ordered=True)

# Fix ordering for frontier plot
frontier_unique_props = sorted(frontier_df['begin_filter_prop'].unique())
frontier_df['begin_filter_prop'] = pd.Categorical(
    frontier_df['begin_filter_prop'],
    categories=frontier_unique_props,
    ordered=True
)
frontier_df['begin_filter_label'] = frontier_df['begin_filter_prop'].map(begin_filter_labels)

# Sort for proper line connections
frontier_df = frontier_df.sort_values(['begin_filter_label', 'loss_type', 'params'])

# Facet labels to use as x-axis titles (in facet order)
facet_x_titles = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']

p_frontier = (ggplot(frontier_df, aes(x='x_loss', y='ood_loss', color='begin_filter_label', group='begin_filter_label'))
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    + facet_wrap('~loss_type_label', ncol=2, scales='free_x')
    + scale_x_log10(name='')
    + scale_y_log10(name=r'Medical Loss ($\uparrow$)')
    + scale_color_manual(values=hex_colors, name='Start Filtering')
    + guides(color=guide_legend(nrow=1))
    + base_theme(base_family='Helvetica Neue')
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
            legend_entry_spacing_x=3,
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
fig.savefig('plots/delayed-loss-frontiers.png', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/delayed-loss-frontiers.svg', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/delayed-loss-frontiers.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print("Saved plots/delayed-loss-frontiers.png")

# ============================================================================
# Delay vs Compute Ratio Plot (relative to unfiltered baseline)
# ============================================================================

# Get unfiltered baseline data for fitting scaling law
baseline_data = df[df['begin_filter_prop'] == 1.0].copy()

delay_vs_compute_data = []

for loss_type in ['ood']:  # Focus on medical (target) loss only
    if len(baseline_data) < 2:
        continue

    baseline_data['log_compute'] = np.log(baseline_data['compute'])
    baseline_data['log_loss'] = np.log(baseline_data[loss_type])

    X = baseline_data['log_compute'].values.reshape(-1, 1)
    y = baseline_data['log_loss'].values
    reg = LinearRegression().fit(X, y)

    # Calculate compute ratios for all filtered models (excluding baseline)
    filtered_df = df[df['begin_filter_prop'] < 1.0].copy()

    for _, row in filtered_df.iterrows():
        log_loss = np.log(row[loss_type])
        log_baseline_compute_needed = (log_loss - reg.intercept_) / reg.coef_[0]
        baseline_compute_needed = np.exp(log_baseline_compute_needed)
        compute_ratio = baseline_compute_needed / row['compute']

        delay_vs_compute_data.append({
            'params': row['params'],
            'params_label': f"{int(row['params'] // 1e6)}M",
            'begin_filter_prop': row['begin_filter_prop'],
            'delay_pct': row['begin_filter_prop'] * 100,  # Convert to percentage
            'compute_ratio': compute_ratio,
            'loss_type': loss_type
        })

if len(delay_vs_compute_data) > 0:
    delay_compute_df = pd.DataFrame(delay_vs_compute_data)

    # Order by params for consistent colors
    param_order = sorted(delay_compute_df['params'].unique())
    param_labels_ordered = [f'{int(p // 1e6)}M' for p in param_order]
    delay_compute_df['params_label'] = pd.Categorical(
        delay_compute_df['params_label'],
        categories=param_labels_ordered,
        ordered=True
    )

    # Get colors for params - use GnBu discrete
    from colors import get_model_size_color_list
    hex_param_colors = get_model_size_color_list(param_labels_ordered)

    # Sort for proper line connections
    delay_compute_df = delay_compute_df.sort_values(['params_label', 'delay_pct'])

    p_delay_compute = (ggplot(delay_compute_df, aes(x='delay_pct', y='compute_ratio', color='params_label', group='params_label'))
        + geom_line(size=1)
        + geom_point(size=2, stroke=0, alpha=0.9)
        + geom_point(fill="none", stroke=0.5, size=2, color="#4f4f4f")
        + scale_x_continuous(name='% Training Before Filtering', labels=lambda l: [f'{int(x)}%' for x in l])
        + scale_y_log10(name='Compute Ratio (vs Unfiltered)')
        + scale_color_manual(values=hex_param_colors, name='Model Size')
        + base_theme(base_family='Helvetica Neue')
        + theme(figure_size=(3.375, 2.5),
                panel_grid_major=element_line(size=0.3, color=grid_color),
                panel_grid_minor=element_blank(),
                legend_title=element_text(size=7, color=text_color),
                legend_position='top',
                legend_direction='horizontal',
                axis_title_x=element_text(size=9, color=text_color),
                axis_title_y=element_text(size=9, color=text_color),
                axis_text_x=element_text(size=7, color=text_color),
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

    p_delay_compute.save('plots/delayed-compute-ratio.png', dpi=300, width=3.375, height=2.5)
    p_delay_compute.save('plots/delayed-compute-ratio.svg', dpi=300, width=3.375, height=2.5)
    p_delay_compute.save('plots/delayed-compute-ratio.pdf', dpi=300, width=3.375, height=2.5)
    print("Saved plots/delayed-compute-ratio.png")
