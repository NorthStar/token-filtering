"""
evaluate sweep on test sets and create scaling law plot
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
from colors import MASK_COLORS, MASK_LABELS, THEME_COLORS, get_mask_color_list
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
parser.add_argument('--data_path',  type=str, default=os.path.join(DATA_PATH, 'test'))
parser.add_argument('--eval_iters', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--block_size', type=int, default=2048)
parser.add_argument('--rerun', action='store_true', help='rerun results')
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

# Use centralized mask colors and labels
mask_order = ['nomask', 'document', 'mask', 'remove']
mask_colors = get_mask_color_list(mask_order)
legend_labels = {m: MASK_LABELS[m] for m in mask_order}

# Model sizes to exclude from plots
excluded_params = [13e6, 28e6]

models = []
for file in os.listdir("../config/adamw-scaling"):
    
    if file.endswith('.yaml'):
        
        if '1816M' in file:
            continue
        
        models.append(file.split('-')[2].split('.')[0] + '-' + file.split('-')[1] + '.pt')

def load_model(model_file):
    checkpoint = torch.load(model_file, map_location=device)
    model_args = checkpoint['model_args']
    # model_args['mup_width_multiplier'] = 1

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

if 'scaling.csv' not in os.listdir('results') or args.rerun:

    df = pd.DataFrame(columns=['params', 'mask', 'target', 'ood', 'parallel', 'parallel_hard'])
    for model_file in models:

        try:
            model = load_model(os.path.join(model_path, model_file))
        
        except Exception as e:
            print(f"error loading model {model_file}: {e}")
            continue
        
        model.eval()
        model.to(device)
        
        if '1030M' in model_file or '1816M' in model_file:
            loss = estimate_test_loss(model, batch_size=8, eval_iters=40)
        else:
            loss = estimate_test_loss(model, batch_size=16, eval_iters=20)
        
        df.loc[len(df)] = {
            'params'   : int(model_file.split('-')[1].split('.')[0][:-1]) * 1e6,
            'mask'     : model_file.split('-')[0],
            'target'   : loss['target'].item(),
            'ood'      : loss['ood'].item(),
            'parallel' : loss['parallel'].item(),
            'parallel_hard' : loss['parallel_hard'].item()
        }

        print(f"model {model_file} done | target {loss['target']:.4f} | ood {loss['ood']:.4f} | parallel {loss['parallel']:.4f}")
        
    df.to_csv('results/scaling.csv', index=False)
else:
    df = pd.read_csv('results/scaling.csv')

df['params'] = pd.to_numeric(df['params'])

# Filter out 13M and 28M models from plots
df = df[~df['params'].isin(excluded_params)].copy()

df['flops'] = 6 * df['params'] * 20 * df['params']

# Compute average_parallel as average of parallel and parallel_hard
df['average_parallel'] = (df['parallel'] + df['parallel_hard']) / 2

# Apply legend labels to mask column
df['mask'] = df['mask'].replace(legend_labels)
df['mask'] = pd.Categorical(df['mask'], categories=[legend_labels[m] for m in mask_order], ordered=True)

loss_type_labels = {
    'target': r'Non-Medical',
    'ood': r'Medical', 
    'average_parallel': r'Biology'
}

# plotnine needs long dataframe - only use target, ood, average_parallel
df_long = pd.melt(df, 
                  id_vars=['params', 'mask', 'flops'], 
                  value_vars=['target', 'ood', 'average_parallel'],
                  var_name='loss_type', 
                  value_name='loss')
df_long['loss_type_label'] = df_long['loss_type'].map(loss_type_labels)

# force the order of facets by converting to categorical with specific order
facet_order = [r'Medical', r'Non-Medical', r'Biology']
df_long['loss_type_label'] = pd.Categorical(df_long['loss_type_label'], categories=facet_order, ordered=True)

# standard exponent breaks for FLOPs
flops_breaks = [1e18, 1e19, 1e20, 1e21]
flops_labels = [r'$10^{18}$', r'$10^{19}$', r'$10^{20}$', r'$10^{21}$']

# Create facet label annotations (inside top-left of each facet)
max_loss = df_long['loss'].max()
min_loss = df_long['loss'].min()
min_flops = df_long['flops'].min()
facet_labels_df = pd.DataFrame({
    'loss_type_label': pd.Categorical(facet_order, categories=facet_order, ordered=True),
    'label': facet_order,
    'flops': [min_flops * 1.3] * 3,
    'loss': [max_loss * 1.25] * 3  # higher to avoid overlap
})

# Plot 1: Normal loss scaling
p1 = (ggplot(df_long, aes(x='flops', y='loss', color='mask'))
     + geom_smooth(method='lm', se=False, size=1.2)
     + geom_point(size=2, stroke=0, alpha=0.9)
     + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
     + facet_wrap('~loss_type_label', ncol=3, scales='free_y')
     + scale_x_log10(name='Pretraining Compute (FLOPs)', breaks=flops_breaks, labels=flops_labels)
     + scale_y_log10(name='Loss')
     + scale_color_manual(values=mask_colors)
     + base_theme(base_family='Helvetica Neue')
     + theme(figure_size=(7, 3.25),
             strip_text=element_text(size=9, color=text_color),
             strip_background=element_blank(),
             panel_grid_major=element_line(size=0.3, color=grid_color),
             panel_grid_minor=element_blank(),
             legend_title=element_blank(),
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

p1.save('plots/scaling.png', dpi=300, width=7, height=3.25)
p1.save('plots/scaling.svg', dpi=300, width=7, height=3.25)
p1.save('plots/scaling.pdf', dpi=300, width=7, height=3.25)
print("Saved plots/scaling.png")

# Fit power law: loss = coefficient * flops^exponent
equation_data = []
for loss_type in ['target', 'ood', 'average_parallel']:
    for mask_name in mask_order:
        mask_label = legend_labels[mask_name]
        subset = df_long[(df_long['loss_type'] == loss_type) & (df_long['mask'] == mask_label)].copy()
        if len(subset) < 2:
            continue

        log_flops = np.log10(subset['flops'].values)
        log_loss = np.log10(subset['loss'].values)

        # Linear regression in log space
        exponent, log_coeff = np.polyfit(log_flops, log_loss, 1)
        coefficient = 10 ** log_coeff

        equation_data.append({
            'loss_type': loss_type,
            'mask': mask_name,
            'coefficient': coefficient,
            'exponent': exponent
        })

equation_df = pd.DataFrame(equation_data)
equation_df.to_csv('results/scaling-equations.csv', index=False)
print("Saved results/scaling-equations.csv")

# Plot 2: Compute ratio scaling
# Need to use original (non-label-replaced) mask names for filtering
df_compute = pd.read_csv('results/scaling.csv')
df_compute['params'] = pd.to_numeric(df_compute['params'])
df_compute = df_compute[~df_compute['params'].isin(excluded_params)].copy()
df_compute['flops'] = 6 * df_compute['params'] * 20 * df_compute['params']
df_compute['average_parallel'] = (df_compute['parallel'] + df_compute['parallel_hard']) / 2

def compute_efficiency_analysis(baseline_mask, comparison_mask, loss_type='ood'):
    # fit scaling law on baseline
    baseline_data = df_compute[df_compute['mask'] == baseline_mask].copy()
    baseline_data['log_flops'] = np.log(baseline_data['flops'])
    baseline_data['log_loss'] = np.log(baseline_data[loss_type])
    
    X = baseline_data['log_flops'].values.reshape(-1, 1)
    y = baseline_data['log_loss'].values
    reg = LinearRegression().fit(X, y)
    
    # calculate compute ratios
    comparison_data = df_compute[df_compute['mask'] == comparison_mask].copy()
    results = []
    
    for _, row in comparison_data.iterrows():
        log_loss = np.log(row[loss_type])
        log_baseline_flops_needed = (log_loss - reg.intercept_) / reg.coef_[0]
        baseline_flops_needed = np.exp(log_baseline_flops_needed)
        compute_ratio = baseline_flops_needed / row['flops']
        
        results.append({
            'params': row['params'],
            'flops': row['flops'],
            'compute_ratio': compute_ratio,
            'mask': comparison_mask,
            'loss_type': loss_type
        })
    
    return results

compute_data = []
for loss_type in ['ood', 'average_parallel']:
    for mask in ['mask', 'document', 'remove']:
        compute_data.extend(compute_efficiency_analysis('nomask', mask, loss_type=loss_type))

compute_df = pd.DataFrame(compute_data)

# Map loss types to simple facet keys
compute_loss_labels = {
    'ood': 'forget',
    'average_parallel': 'retain'
}
compute_df['loss_type_label'] = compute_df['loss_type'].map(compute_loss_labels)
compute_df['loss_type_label'] = pd.Categorical(
    compute_df['loss_type_label'],
    categories=['forget', 'retain'],
    ordered=True
)

# Apply legend labels to mask column
compute_df['mask'] = compute_df['mask'].replace(legend_labels)
compute_df['mask'] = pd.Categorical(compute_df['mask'], categories=[legend_labels[m] for m in ['document', 'mask', 'remove']], ordered=True)

# Create facet label annotations for compute ratio plot (inside top-left of each facet)
max_ratio = compute_df['compute_ratio'].max()
min_ratio = compute_df['compute_ratio'].min()
min_flops_compute = compute_df['flops'].min()
compute_facet_labels = pd.DataFrame({
    'loss_type_label': pd.Categorical(['forget', 'retain'],
                                       categories=['forget', 'retain'], ordered=True),
    'label': [r'$\mathbf{Forget}$ (Medical)', r'$\mathbf{Retain}$ (Biology)'],
    'flops': [min_flops_compute * 1.1] * 2,
    'compute_ratio': [2.3] * 2  # fixed position near top
})

# Plot 2a: Compute ratio - Forget/Retain faceted (3.375 x 3)
p2a = (ggplot(compute_df, aes(x='flops', y='compute_ratio', color='mask'))
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    + geom_text(data=compute_facet_labels,
                mapping=aes(x='flops', y='compute_ratio', label='label'),
                ha='left', va='top', size=8, family='Helvetica Neue',
                color=text_color, inherit_aes=False)
    + facet_wrap('~loss_type_label', ncol=2)
    + scale_x_log10(name='Pretraining Compute (FLOPs)', breaks=flops_breaks, labels=flops_labels)
    + scale_y_log10(name='Loss-Matched Baseline Compute', limits=(min_ratio * 0.85, 2.5),
                    breaks=[0.001, 0.01, 0.1, 1.0], labels=[r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1.0$'])
    + scale_color_manual(values=mask_colors[1:])  # exclude nomask color
    + guides(color=guide_legend(nrow=1))
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(3.375, 2.75),
            strip_text=element_blank(),
            strip_background=element_blank(),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
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

p2a.save('plots/compute_ratio.png', dpi=300, width=3.375, height=2.75)
p2a.save('plots/compute_ratio.svg', dpi=300, width=3.375, height=2.75)
p2a.save('plots/compute_ratio.pdf', dpi=300, width=3.375, height=2.75)
print("Saved plots/compute_ratio.png")

# Plot 2b: Compute ratio - Medical and Biology (7 x 4)
p2b = (ggplot(compute_df, aes(x='flops', y='compute_ratio', color='mask'))
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    + geom_text(data=compute_facet_labels,
                mapping=aes(x='flops', y='compute_ratio', label='label'),
                ha='left', va='top', size=9, family='Helvetica Neue',
                color=text_color, inherit_aes=False)
    + facet_wrap('~loss_type_label', ncol=2)
    + scale_x_log10(name='Pretraining Compute (FLOPs)', breaks=flops_breaks, labels=flops_labels)
    + scale_y_log10(name='Loss-Matched Baseline Compute', limits=(min_ratio * 0.85, 2.5),
                    breaks=[0.001, 0.01, 0.1, 1.0], labels=[r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1.0$'])
    + scale_color_manual(values=mask_colors[1:])  # exclude nomask color
    + guides(color=guide_legend(nrow=1))
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(7, 3.25),
            strip_text=element_blank(),
            strip_background=element_blank(),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
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

p2b.save('plots/compute_ratio_full.png', dpi=300, width=7, height=3.25)
p2b.save('plots/compute_ratio_full.svg', dpi=300, width=7, height=3.25)
p2b.save('plots/compute_ratio_full.pdf', dpi=300, width=7, height=3.25)
print("Saved plots/compute_ratio_full.png")

# Plot 3: Loss frontiers
# Reload df without label replacement for frontier calculations
df_frontier = pd.read_csv('results/scaling.csv')
df_frontier['params'] = pd.to_numeric(df_frontier['params'])
df_frontier = df_frontier[~df_frontier['params'].isin(excluded_params)].copy()
df_frontier['flops'] = 6 * df_frontier['params'] * 20 * df_frontier['params']
df_frontier['average_parallel'] = (df_frontier['parallel'] + df_frontier['parallel_hard']) / 2

frontier_data = []
for _, row in df_frontier.iterrows():
    for loss_type in ['target', 'average_parallel']:
        frontier_data.append({
            'params': row['params'],
            'flops': row['flops'],
            'mask': row['mask'],
            'ood_loss': row['ood'],
            'x_loss': row[loss_type],
            'loss_type': loss_type
        })

frontier_df = pd.DataFrame(frontier_data)

# Labels for frontiers (with arrows for strip text)
x_loss_labels = {
    'target': r'Non-Medical Loss ($\downarrow$)',
    'average_parallel': r'Biology Loss ($\downarrow$)'
}
frontier_df['loss_type_label'] = frontier_df['loss_type'].map(x_loss_labels)

x_facet_order = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']
frontier_df['loss_type_label'] = pd.Categorical(frontier_df['loss_type_label'], categories=x_facet_order, ordered=True)

# Apply legend labels to mask column
frontier_df['mask'] = frontier_df['mask'].replace(legend_labels)
frontier_df['mask'] = pd.Categorical(frontier_df['mask'], categories=[legend_labels[m] for m in mask_order], ordered=True)

# Create iso-parameter lines (connect points of same model size, extrapolated to facet edges)
from scipy.interpolate import CubicSpline

def create_iso_param_data(df, loss_type, n_points=100, extend_factor=0.15):
    """Get points for iso-parameter lines with smooth interpolation and extrapolation"""
    iso_data = []
    
    # Get global bounds for this loss type (for extrapolation limits)
    loss_type_data = df[df['loss_type'] == loss_type]
    global_log_x_min = np.log10(loss_type_data['x_loss'].min())
    global_log_x_max = np.log10(loss_type_data['x_loss'].max())
    global_log_x_range = global_log_x_max - global_log_x_min
    
    # Extended bounds
    ext_log_x_min = global_log_x_min - extend_factor * global_log_x_range
    ext_log_x_max = global_log_x_max + extend_factor * global_log_x_range
    
    for params in df['params'].unique():
        param_data = df[(df['params'] == params) & (df['loss_type'] == loss_type)].copy()
        if len(param_data) < 2:
            continue
        
        flops = param_data['flops'].values[0]
        
        # Sort by x for interpolation
        param_data = param_data.sort_values('x_loss')
        log_x = np.log10(param_data['x_loss'].values)
        log_y = np.log10(param_data['ood_loss'].values)
        
        # Fit a polynomial for smooth interpolation and extrapolation
        # Use degree min(2, n_points-1) to avoid overfitting
        degree = min(2, len(log_x) - 1)
        coeffs = np.polyfit(log_x, log_y, degree)
        poly = np.poly1d(coeffs)
        
        # Generate x values across the extended range
        log_x_line = np.linspace(ext_log_x_min, ext_log_x_max, n_points)
        log_y_line = poly(log_x_line)
        
        # Clip y to reasonable bounds (within 15% of data range beyond actual data)
        log_y_min, log_y_max = log_y.min(), log_y.max()
        log_y_range = log_y_max - log_y_min
        log_y_clip_min = log_y_min - 0.15 * log_y_range
        log_y_clip_max = log_y_max + 0.15 * log_y_range
        log_y_line = np.clip(log_y_line, log_y_clip_min, log_y_clip_max)
        
        x_line = 10 ** log_x_line
        y_line = 10 ** log_y_line
        
        for i, (x, y) in enumerate(zip(x_line, y_line)):
            iso_data.append({
                'x_loss': x,
                'ood_loss': y,
                'params': params,
                'flops': flops,
                'loss_type': loss_type,
                'is_label_point': i == len(x_line) - 1
            })
    
    return pd.DataFrame(iso_data)

# Generate iso-parameter data for each loss type
iso_curves_list = []
for lt in ['target', 'average_parallel']:
    iso_curves_list.append(create_iso_param_data(frontier_df, lt))
iso_curves_df = pd.concat(iso_curves_list, ignore_index=True)
iso_curves_df['loss_type_label'] = iso_curves_df['loss_type'].map(x_loss_labels)
iso_curves_df['loss_type_label'] = pd.Categorical(iso_curves_df['loss_type_label'], categories=x_facet_order, ordered=True)

# Create FLOPs labels with simple e notation
def format_flops(f):
    if f <= 0:
        return ''
    exp = int(np.floor(np.log10(f)))
    mantissa = f / (10 ** exp)
    if mantissa >= 9.95:
        exp += 1
        mantissa = 1.0
    return f'{mantissa:.1f}e{exp}'

iso_curves_df['flops_label'] = iso_curves_df['flops'].apply(format_flops)

# Get label points and calculate rotation angles along the curve
label_points = iso_curves_df[iso_curves_df['is_label_point']].copy()

# Calculate angle for each label based on curve direction
def calculate_label_angles(iso_df, label_df):
    angles = []
    for _, label_row in label_df.iterrows():
        params = label_row['params']
        loss_type = label_row['loss_type']
        curve_data = iso_df[(iso_df['params'] == params) & (iso_df['loss_type'] == loss_type)].copy()
        curve_data = curve_data.sort_values('ood_loss')
        
        if len(curve_data) >= 2:
            # Get last two points to calculate angle
            x_vals = np.log10(curve_data['x_loss'].values)
            y_vals = np.log10(curve_data['ood_loss'].values)
            dx = x_vals[-1] - x_vals[-2]
            dy = y_vals[-1] - y_vals[-2]
            angle = np.degrees(np.arctan2(dy, dx))
        else:
            angle = 0
        angles.append(angle)
    return angles

label_points['angle'] = calculate_label_angles(iso_curves_df, label_points)

# Sort frontier_df for proper line connections
frontier_df = frontier_df.sort_values(['mask', 'loss_type', 'params'])

# Facet labels to use as x-axis titles (in facet order)
facet_x_titles = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']

p3 = (ggplot(frontier_df, aes(x='x_loss', y='ood_loss', color='mask', group='mask'))
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    + facet_wrap('~loss_type_label', ncol=2, scales='free')
    + scale_x_log10(name='')
    + scale_y_log10(name=r'Medical Loss ($\uparrow$)')
    + scale_color_manual(values=mask_colors)
    + guides(color=guide_legend(nrow=1))
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(3.375, 2.34),
            strip_text=element_blank(),
            strip_background=element_blank(),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            legend_position='top',
            legend_direction='horizontal',
            legend_box_just='center',
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
fig = p3.draw()
# The first two axes are the plot panels (facets)
for ax, title in zip(fig.axes[:2], facet_x_titles):
    ax.set_xlabel(title, fontsize=9, color=text_color, fontfamily='Helvetica Neue')
fig.savefig('plots/loss-frontiers.png', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/loss-frontiers.svg', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/loss-frontiers.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print("Saved plots/loss-frontiers.png")

# ============================================================================
# Plot 1b: Annotated Loss Scaling (Medical only, x-axis from 10^16, with arrows)
# ============================================================================

# Filter to Medical (ood) only
df_medical = df_long[df_long['loss_type'] == 'ood'].copy()

# Fit power laws for each mask type and get coefficients for extrapolation
fit_params = {}
for mask_label in df_medical['mask'].unique():
    subset = df_medical[df_medical['mask'] == mask_label]
    log_flops = np.log10(subset['flops'].values)
    log_loss = np.log10(subset['loss'].values)
    slope, intercept = np.polyfit(log_flops, log_loss, 1)
    fit_params[mask_label] = {'slope': slope, 'intercept': intercept}

# Create extended fit lines from 10^16 to beyond data
extended_flops = np.logspace(16, np.log10(df_medical['flops'].max()) + 0.1, 100)
fit_lines = []
for mask_label, params in fit_params.items():
    for f in extended_flops:
        log_loss = params['slope'] * np.log10(f) + params['intercept']
        fit_lines.append({
            'flops': f,
            'loss': 10 ** log_loss,
            'mask': mask_label
        })
fit_df = pd.DataFrame(fit_lines)
fit_df['mask'] = pd.Categorical(fit_df['mask'], categories=[legend_labels[m] for m in mask_order], ordered=True)

# Get largest model points for each mask type (except nomask)
largest_params = df_medical['params'].max()
largest_points = df_medical[df_medical['params'] == largest_params].copy()

# Calculate intersection points with baseline for arrows
baseline_params = fit_params[legend_labels['nomask']]
arrow_data = []
for mask_name in ['document', 'mask', 'remove']:
    mask_label = legend_labels[mask_name]
    point = largest_points[largest_points['mask'] == mask_label].iloc[0]
    target_loss = point['loss']

    # Find flops where baseline reaches this loss
    log_flops_baseline = (np.log10(target_loss) - baseline_params['intercept']) / baseline_params['slope']
    flops_baseline = 10 ** log_flops_baseline

    arrow_data.append({
        'mask': mask_label,
        'start_flops': point['flops'],
        'end_flops': flops_baseline,
        'loss': target_loss
    })

arrow_df = pd.DataFrame(arrow_data)

# Extended x-axis breaks (starting from 10^17 since limit is 3e16)
flops_breaks_ext = [1e17, 1e18, 1e19, 1e20, 1e21]
flops_labels_ext = [r'$10^{17}$', r'$10^{18}$', r'$10^{19}$', r'$10^{20}$', r'$10^{21}$']

# Create arrow segment data for geom_segment (arrow points left)
arrow_df['x'] = arrow_df['start_flops']
arrow_df['y'] = arrow_df['loss']
arrow_df['xend'] = arrow_df['end_flops']
arrow_df['yend'] = arrow_df['loss']

# Compute ratio (right / left = start_flops / end_flops) and format label
def format_ratio(ratio):
    # Round to 1 significant figure
    from math import floor, log10
    exp = floor(log10(ratio))
    rounded = round(ratio / (10 ** exp)) * (10 ** exp)
    return rf'$\mathbf{{{int(rounded)}\times\ Efficiency}}$'

arrow_df['ratio'] = arrow_df['start_flops'] / arrow_df['end_flops']
arrow_df['ratio_label'] = arrow_df['ratio'].apply(format_ratio)
# Position label at geometric mean of x positions, slightly above the line
arrow_df['label_x'] = np.sqrt(arrow_df['start_flops'] * arrow_df['end_flops'])
arrow_df['label_y'] = arrow_df['loss'] * 1.02  # just above

# Create base plot with plotnine (matching loss-frontiers style)
p_annot = (ggplot()
     + geom_line(data=fit_df, mapping=aes(x='flops', y='loss', color='mask'), size=1)
     + geom_point(data=df_medical, mapping=aes(x='flops', y='loss', color='mask'), size=2, stroke=0, alpha=0.9)
     + geom_point(data=df_medical, mapping=aes(x='flops', y='loss'), fill="none", stroke=0.7, size=2, color="#4f4f4f")
     + geom_segment(data=arrow_df, mapping=aes(x='x', y='y', xend='xend', yend='yend'),
                    color='black', size=0.8, arrow=arrow(length=0.08, type='closed'))
     + geom_label(data=arrow_df, mapping=aes(x='label_x', y='label_y', label='ratio_label'),
                  size=8, color='black', va='bottom', family='Helvetica Neue',
                  fill='#ffffff80', label_size=0, label_padding=0.15, boxstyle='square,pad=0.15')
     + scale_x_log10(name='Pretraining Compute (FLOPs)', breaks=flops_breaks_ext, labels=flops_labels_ext,
                     limits=(3e16, df_medical['flops'].max() * 1.5))
     + scale_y_log10(name='Medical Loss')
     + scale_color_manual(values=mask_colors)
     + guides(color=guide_legend(nrow=1))
     + base_theme(base_family='Helvetica Neue')
     + theme(figure_size=(3.375, 2.34),
             panel_grid_major=element_line(size=0.3, color=grid_color),
             panel_grid_minor=element_blank(),
             legend_title=element_blank(),
             legend_position='top',
             legend_direction='horizontal',
             legend_box_just='center',
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

p_annot.save('plots/scaling-annotated.png', dpi=300, width=3.375, height=3.0)
p_annot.save('plots/scaling-annotated.svg', dpi=300, width=3.375, height=3.0)
p_annot.save('plots/scaling-annotated.pdf', dpi=300, width=3.375, height=3.0)
print("Saved plots/scaling-annotated.png")

# ============================================================================
# Token-level Evaluation (loss on medical/non-medical tokens only)
# ============================================================================

def get_test_batch_tokens(batch_size=32):
    """Load test batches with token-level labels for masked loss computation."""
    
    test_target = np.memmap(os.path.join(data_dir, 'test_target.bin'), dtype=loader_dtype, mode='r')
    test_ood = np.memmap(os.path.join(data_dir, 'test_ood.bin'), dtype=loader_dtype, mode='r')
    test_parallel = np.memmap(os.path.join(data_dir, 'test_parallel.bin'), dtype=loader_dtype, mode='r')
    test_parallel_hard = np.memmap(os.path.join(data_dir, 'test_parallel_hard.bin'), dtype=loader_dtype, mode='r')
    
    # Load labels
    labels_target = np.memmap(os.path.join(data_dir, 'test_target_labels.bin'), dtype=bool, mode='r')
    labels_ood = np.memmap(os.path.join(data_dir, 'test_ood_labels.bin'), dtype=bool, mode='r')
    labels_parallel = np.memmap(os.path.join(data_dir, 'test_parallel_labels.bin'), dtype=bool, mode='r')
    labels_parallel_hard = np.memmap(os.path.join(data_dir, 'test_parallel_hard_labels.bin'), dtype=bool, mode='r')

    ix = torch.randint(len(test_target) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((test_target[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((test_target[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy((labels_target[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_ood) - block_size, (batch_size,))
    x_ood = torch.stack([torch.from_numpy((test_ood[i:i+block_size]).astype(np.int64)) for i in ix])
    y_ood = torch.stack([torch.from_numpy((test_ood[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    m_ood = torch.stack([torch.from_numpy((labels_ood[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_parallel) - block_size, (batch_size,))
    x_parallel = torch.stack([torch.from_numpy((test_parallel[i:i+block_size]).astype(np.int64)) for i in ix])
    y_parallel = torch.stack([torch.from_numpy((test_parallel[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    m_parallel = torch.stack([torch.from_numpy((labels_parallel[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_parallel_hard) - block_size, (batch_size,))
    x_parallel_hard = torch.stack([torch.from_numpy((test_parallel_hard[i:i+block_size]).astype(np.int64)) for i in ix])
    y_parallel_hard = torch.stack([torch.from_numpy((test_parallel_hard[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    m_parallel_hard = torch.stack([torch.from_numpy((labels_parallel_hard[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y, m = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), m.pin_memory().to(device, non_blocking=True)
        x_ood, y_ood, m_ood = x_ood.pin_memory().to(device, non_blocking=True), y_ood.pin_memory().to(device, non_blocking=True), m_ood.pin_memory().to(device, non_blocking=True)
        x_parallel, y_parallel, m_parallel = x_parallel.pin_memory().to(device, non_blocking=True), y_parallel.pin_memory().to(device, non_blocking=True), m_parallel.pin_memory().to(device, non_blocking=True)
        x_parallel_hard, y_parallel_hard, m_parallel_hard = x_parallel_hard.pin_memory().to(device, non_blocking=True), y_parallel_hard.pin_memory().to(device, non_blocking=True), m_parallel_hard.pin_memory().to(device, non_blocking=True)
    else:
        x, y, m = x.to(device), y.to(device), m.to(device)
        x_ood, y_ood, m_ood = x_ood.to(device), y_ood.to(device), m_ood.to(device)
        x_parallel, y_parallel, m_parallel = x_parallel.to(device), y_parallel.to(device), m_parallel.to(device)
        x_parallel_hard, y_parallel_hard, m_parallel_hard = x_parallel_hard.to(device), y_parallel_hard.to(device), m_parallel_hard.to(device)

    return (x, y, m), (x_ood, y_ood, m_ood), (x_parallel, y_parallel, m_parallel), (x_parallel_hard, y_parallel_hard, m_parallel_hard)

def compute_masked_loss(logits, targets, mask):
    """Compute cross-entropy loss only on masked tokens."""
    # logits: (batch, seq_len, vocab_size)
    # targets: (batch, seq_len)
    # mask: (batch, seq_len) - 1 for tokens to include, 0 for tokens to exclude
    
    from torch.nn import functional as F
    
    # Flatten for cross entropy
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    targets_flat = targets.view(B * T)
    mask_flat = mask.view(B * T).bool()
    
    # Compute per-token loss
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    # Apply mask and average
    masked_loss = loss_per_token[mask_flat]
    if len(masked_loss) == 0:
        return torch.tensor(float('nan'))
    
    return masked_loss.mean()

@torch.no_grad()
def estimate_test_loss_tokens(model, batch_size=32, eval_iters=10):
    """
    Estimate test loss on specific tokens:
    - For ood (medical): loss on medical tokens only (label=1)
    - For target/parallel/parallel_hard: loss on non-medical tokens only (label=0)
    """
    out = {'target': torch.zeros(eval_iters), 'ood': torch.zeros(eval_iters), 
           'parallel': torch.zeros(eval_iters), 'parallel_hard': torch.zeros(eval_iters)}
    model.eval()

    for k in range(eval_iters):
        (X, Y, M), (X_ood, Y_ood, M_ood), (X_parallel, Y_parallel, M_parallel), (X_parallel_hard, Y_parallel_hard, M_parallel_hard) = get_test_batch_tokens(batch_size=batch_size)

        with torch.no_grad():
            # Target: loss on NON-medical tokens (label=0)
            logits, _ = model(X, idx_filter=None, targets=Y, targets_filter=None)
            loss = compute_masked_loss(logits, Y, 1 - M)  # Invert mask: use label=0
            out['target'][k] = loss.item()
        
            # OOD: loss on MEDICAL tokens (label=1)
            logits, _ = model(X_ood, idx_filter=None, targets=Y_ood, targets_filter=None)
            loss = compute_masked_loss(logits, Y_ood, M_ood)  # Use label=1
            out['ood'][k] = loss.item()

            # Parallel: loss on NON-medical tokens (label=0)
            logits, _ = model(X_parallel, idx_filter=None, targets=Y_parallel, targets_filter=None)
            loss = compute_masked_loss(logits, Y_parallel, 1 - M_parallel)
            out['parallel'][k] = loss.item()

            # Parallel hard: loss on NON-medical tokens (label=0)
            logits, _ = model(X_parallel_hard, idx_filter=None, targets=Y_parallel_hard, targets_filter=None)
            loss = compute_masked_loss(logits, Y_parallel_hard, 1 - M_parallel_hard)
            out['parallel_hard'][k] = loss.item()
    
    return {'target': out['target'].mean(), 'ood': out['ood'].mean(), 
            'parallel': out['parallel'].mean(), 'parallel_hard': out['parallel_hard'].mean()}

# Check if label files exist
labels_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in 
                   ['test_target_labels.bin', 'test_ood_labels.bin', 
                    'test_parallel_labels.bin', 'test_parallel_hard_labels.bin'])

if labels_exist:
    print("\n" + "="*80)
    print("Token-level Evaluation (medical/non-medical tokens)")
    print("="*80 + "\n")
    
    if 'scaling-tokens.csv' not in os.listdir('results') or args.rerun:
        
        df_tokens = pd.DataFrame(columns=['params', 'mask', 'target', 'ood', 'parallel', 'parallel_hard'])
        for model_file in models:

            try:
                model = load_model(os.path.join(model_path, model_file))
            except Exception as e:
                print(f"error loading model {model_file}: {e}")
                continue
            
            model.eval()
            model.to(device)
            
            if '1030M' in model_file or '1816M' in model_file:
                loss = estimate_test_loss_tokens(model, batch_size=8, eval_iters=40)
            else:
                loss = estimate_test_loss_tokens(model, batch_size=16, eval_iters=20)
            
            df_tokens.loc[len(df_tokens)] = {
                'params'   : int(model_file.split('-')[1].split('.')[0][:-1]) * 1e6,
                'mask'     : model_file.split('-')[0],
                'target'   : loss['target'].item(),
                'ood'      : loss['ood'].item(),
                'parallel' : loss['parallel'].item(),
                'parallel_hard' : loss['parallel_hard'].item()
            }

            print(f"[tokens] model {model_file} done | target {loss['target']:.4f} | ood {loss['ood']:.4f} | parallel {loss['parallel']:.4f}")
            
        df_tokens.to_csv('results/scaling-tokens.csv', index=False)
    else:
        df_tokens = pd.read_csv('results/scaling-tokens.csv')

    df_tokens['params'] = pd.to_numeric(df_tokens['params'])
    
    # Filter out 13M and 28M models from plots
    df_tokens = df_tokens[~df_tokens['params'].isin(excluded_params)].copy()
    
    df_tokens['flops'] = 6 * df_tokens['params'] * 20 * df_tokens['params']

    # Compute average_parallel
    df_tokens['average_parallel'] = (df_tokens['parallel'] + df_tokens['parallel_hard']) / 2

    # Apply legend labels to mask column
    df_tokens['mask'] = df_tokens['mask'].replace(legend_labels)
    df_tokens['mask'] = pd.Categorical(df_tokens['mask'], categories=[legend_labels[m] for m in mask_order], ordered=True)

    # Create long dataframe for plotting - only target, ood, average_parallel
    df_tokens_long = pd.melt(df_tokens, 
                      id_vars=['params', 'mask', 'flops'], 
                      value_vars=['target', 'ood', 'average_parallel'],
                      var_name='loss_type', 
                      value_name='loss')
    
    token_loss_labels = {
        'target': r'Non-Medical',
        'ood': r'Medical', 
        'average_parallel': r'Biology'
    }
    df_tokens_long['loss_type_label'] = df_tokens_long['loss_type'].map(token_loss_labels)
    
    token_facet_order = [r'Medical', r'Non-Medical', r'Biology']
    df_tokens_long['loss_type_label'] = pd.Categorical(df_tokens_long['loss_type_label'], 
                                                        categories=token_facet_order, ordered=True)

    # Create facet label annotations for token plot
    max_loss_tokens = df_tokens_long['loss'].max()
    min_loss_tokens = df_tokens_long['loss'].min()
    min_flops_tokens = df_tokens_long['flops'].min()
    token_facet_labels_df = pd.DataFrame({
        'loss_type_label': pd.Categorical(token_facet_order, categories=token_facet_order, ordered=True),
        'label': token_facet_order,
        'flops': [min_flops_tokens * 1.3] * 3,
        'loss': [max_loss_tokens * 1.25] * 3  # higher to avoid overlap
    })

    # Plot: Token-level loss scaling
    p_tokens = (ggplot(df_tokens_long, aes(x='flops', y='loss', color='mask'))
         + geom_smooth(method='lm', se=False, size=1.2)
         + geom_point(size=2, stroke=0, alpha=0.9)
         + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
         + facet_wrap('~loss_type_label', ncol=3, scales='free_y')
         + scale_x_log10(name='Pretraining Compute (FLOPs)', breaks=flops_breaks, labels=flops_labels)
         + scale_y_log10(name='Loss', limits=(min_loss_tokens * 0.9, max_loss_tokens * 1.35))
         + scale_color_manual(values=mask_colors)
         + base_theme(base_family='Helvetica Neue')
         + theme(figure_size=(10, 5),
                 strip_text=element_text(size=9, color=text_color),
                 strip_background=element_blank(),
                 panel_grid_major=element_line(size=0.3, color=grid_color),
                 panel_grid_minor=element_blank(),
                 legend_title=element_blank(),
                 legend_position='top',
                 legend_direction='horizontal',
                 axis_title_x=element_text(size=9, color=text_color),
                 axis_title_y=element_text(size=9, color=text_color),
                 axis_text_x=element_text(size=7, color=text_color),
                 axis_text_y=element_text(size=7, color=text_color),
                 legend_text=element_text(size=7, color=text_color),
                 plot_background=element_rect(fill=bg_color),
                 panel_background=element_rect(fill=bg_color),
                 panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
                 legend_background=element_rect(fill=bg_color))
    )

    p_tokens.save('plots/scaling-tokens.png', dpi=300, width=10, height=5)
    p_tokens.save('plots/scaling-tokens.svg', dpi=300, width=10, height=5)
    p_tokens.save('plots/scaling-tokens.pdf', dpi=300, width=10, height=5)
    print("Saved plots/scaling-tokens.png")
    
    # Token-level loss frontiers - reload without label replacement
    df_tokens_frontier = pd.read_csv('results/scaling-tokens.csv')
    df_tokens_frontier['params'] = pd.to_numeric(df_tokens_frontier['params'])
    df_tokens_frontier = df_tokens_frontier[~df_tokens_frontier['params'].isin(excluded_params)].copy()
    df_tokens_frontier['flops'] = 6 * df_tokens_frontier['params'] * 20 * df_tokens_frontier['params']
    df_tokens_frontier['average_parallel'] = (df_tokens_frontier['parallel'] + df_tokens_frontier['parallel_hard']) / 2
    
    frontier_tokens_data = []
    for _, row in df_tokens_frontier.iterrows():
        for loss_type in ['target', 'average_parallel']:
            frontier_tokens_data.append({
                'params': row['params'],
                'flops': row['flops'],
                'mask': row['mask'],
                'ood_loss': row['ood'],
                'x_loss': row[loss_type],
                'loss_type': loss_type
            })

    frontier_tokens_df = pd.DataFrame(frontier_tokens_data)
    frontier_tokens_df['loss_type_label'] = frontier_tokens_df['loss_type'].map(x_loss_labels)
    frontier_tokens_df['loss_type_label'] = pd.Categorical(frontier_tokens_df['loss_type_label'], 
                                                            categories=x_facet_order, ordered=True)
    
    # Apply legend labels to mask column
    frontier_tokens_df['mask'] = frontier_tokens_df['mask'].replace(legend_labels)
    frontier_tokens_df['mask'] = pd.Categorical(frontier_tokens_df['mask'], categories=[legend_labels[m] for m in mask_order], ordered=True)
    frontier_tokens_df = frontier_tokens_df.sort_values(['mask', 'loss_type', 'params'])

    p_tokens_frontier = (ggplot(frontier_tokens_df, aes(x='x_loss', y='ood_loss', color='mask', group='mask'))
        + geom_line(size=1)
        + geom_point(size=2, stroke=0, alpha=0.9)
        + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
        + facet_wrap('~loss_type_label', ncol=2, scales='free')
        + scale_x_log10(name='')
        + scale_y_log10(name=r'Medical Loss ($\uparrow$)')
        + scale_color_manual(values=mask_colors)
        + base_theme(base_family='Helvetica Neue')
        + theme(figure_size=(9, 5),
                strip_text_x=element_text(size=9, color=text_color),
                strip_background=element_blank(),
                panel_grid_major=element_line(size=0.3, color=grid_color),
                panel_grid_minor=element_blank(),
                legend_title=element_blank(),
                legend_position='top',
                legend_direction='horizontal',
                axis_title_x=element_blank(),
                axis_text_x=element_text(size=7, color=text_color),
                axis_title_y=element_text(size=9, color=text_color),
                axis_text_y=element_text(size=7, color=text_color),
                legend_text=element_text(size=7, color=text_color),
                plot_background=element_rect(fill=bg_color),
                panel_background=element_rect(fill=bg_color),
                panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
                legend_background=element_rect(fill=bg_color))
    )

    p_tokens_frontier.save('plots/loss-frontiers-tokens.png', dpi=300, width=9, height=5)
    p_tokens_frontier.save('plots/loss-frontiers-tokens.svg', dpi=300, width=9, height=5)
    p_tokens_frontier.save('plots/loss-frontiers-tokens.pdf', dpi=300, width=9, height=5)
    print("Saved plots/loss-frontiers-tokens.png")
else:
    print("\nToken-level label files not found - skipping token-level evaluation")
