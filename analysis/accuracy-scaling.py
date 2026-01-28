"""
Evaluate model sweep on test sets and create accuracy frontier and auroc vs inverse slope plots
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc as compute_auc
from scipy.interpolate import splprep, splev
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
from colors import PROBE_COLORS, PROBE_ORDER, get_probe_color_list, MODEL_SIZE_COLORS, get_model_size_color_list, THEME_COLORS

# ============================================================================
# Setup and Configuration
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=str, default='cuda')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'fixed-accuracy'))
parser.add_argument('--data_path',  type=str, default=os.path.join(DATA_PATH, 'test'))
parser.add_argument('--eval_iters', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--block_size', type=int, default=1024)
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

# Model sizes to exclude from plots
excluded_params = [13e6, 28e6]

# ============================================================================
# Model Discovery
# ============================================================================

models = []
nomask_models = []
for file in os.listdir("../config/fixed-accuracy"):

    if '1030M' in file:
        continue
    
    if file.endswith('.yaml'):
        models.append('mask-' + '-'.join(file.split('-')[1:]).split('.')[0] + '.pt')

for file in os.listdir("../config/scaling"):
    if '1030M' in file:
        continue
    if 'nomask' in file:
        nomask_models.append(file.split('-')[2].split('.')[0] + '-' + file.split('-')[1] + '.pt')
    # if 'mask' in file:
    #     models.append(file.split('-')[2].split('.')[0] + '-' + file.split('-')[1] + '.pt')

# ============================================================================
# Model Loading and Evaluation Functions
# ============================================================================

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

# ============================================================================
# Evaluate Models and Save Results
# ============================================================================

if 'fixed-accuracy-scaling.csv' not in os.listdir('results') or args.rerun:

    df = pd.DataFrame(columns=['params', 'probe', 'target', 'ood', 'parallel', 'parallel_hard'])
    for model_file in models:

        try:
            model = load_model(os.path.join(model_path, model_file))
        
        except Exception as e:
            print(f"error loading model {model_file}: {e}")
            continue
        
        model.eval()
        model.to(device)
        
        loss = estimate_test_loss(model, batch_size=32, eval_iters=10)
        
        if 'roberta-edu' in model_file:
            probe_name = 'roberta-edu'
        elif 'edu-61M' in model_file:
            probe_name = 'edu-61M'
        elif 'ModernBERT-large' in model_file:
            probe_name = 'ModernBERT-large'
        # elif model_file.count('-') == 1:
        #     probe_name = int(224) * 1e6
        else:
            probe_name = int(model_file.split('-')[2].split('.')[0][:-1]) * 1e6
        
        df.loc[len(df)] = {
            'params'   : int(model_file.split('-')[1][:-1]) * 1e6,
            'probe'    : probe_name,
            'target'   : loss['target'].item(),
            'ood'      : loss['ood'].item(),
            'parallel' : loss['parallel'].item(),
            'parallel_hard' : loss['parallel_hard'].item()
        }

        print(f"model {model_file} done | target {loss['target']:.4f} | ood {loss['ood']:.4f}")
    
    for model_file in nomask_models:

        try:
            model = load_model(os.path.join(model_path.replace('fixed-accuracy', 'gpt'), model_file))
        
        except Exception as e:
            print(f"error loading model {model_file}: {e}")
            continue
        
        model.eval()
        model.to(device)
        
        loss = estimate_test_loss(model, batch_size=32, eval_iters=10)
        
        df.loc[len(df)] = {
            'params'   : int(model_file.split('-')[1].split('.')[0][:-1]) * 1e6,
            'probe'    : 0,
            'target'   : loss['target'].item(),
            'ood'      : loss['ood'].item(),
            'parallel' : loss['parallel'].item(),
            'parallel_hard' : loss['parallel_hard'].item()
        }

        print(f"model {model_file} done | target {loss['target']:.4f} | ood {loss['ood']:.4f}")
    
    df.to_csv('results/fixed-accuracy-scaling.csv', index=False)
else:
    df = pd.read_csv('results/fixed-accuracy-scaling.csv')

# ============================================================================
# Data Preparation
# ============================================================================

def get_probe_label(probe_size):

    if probe_size == 'roberta-edu':
        return 'RoBERTa'
    elif probe_size == 'edu-61M':
        return 'edu-61M'
    elif probe_size == 'ModernBERT-large':
        return 'ModernBERT'
    elif int(float(probe_size)) == 0:
        return 'No Filtering'
    else:
        return 'biLM-' + str(int(float(probe_size) // 1e6)) + 'M'

df['params'] = pd.to_numeric(df['params'])
df['probe_label'] = df['probe'].apply(get_probe_label)

# Compute average_parallel as average of parallel and parallel_hard
df['average_parallel'] = (df['parallel'] + df['parallel_hard']) / 2

# ============================================================================
# Filter nomask to only model scales that exist for masked models
# ============================================================================

# Get model scales that exist for non-nomask classifiers
masked_params = df[df['probe_label'] != 'No Filtering']['params'].unique()
print(f"Model scales for masked classifiers: {sorted(masked_params)}")

# Filter nomask to only these scales
nomask_mask = (df['probe_label'] != 'No Filtering') | (df['params'].isin(masked_params))
df = df[nomask_mask].copy()
print(f"After filtering, nomask model scales: {sorted(df[df['probe_label'] == 'No Filtering']['params'].unique())}")

# Filter out 13M and 28M models from plots
df = df[~df['params'].isin(excluded_params)].copy()

# Filter out biLM-13M and biLM-29M classifiers from all plots
excluded_probes = ['biLM-13M', 'biLM-29M']
df = df[~df['probe_label'].isin(excluded_probes)].copy()

# setup color palette for plots - use centralized probe colors (Blue-Yellow discrete)
probe_labels = ['No Filtering', 'ModernBERT', 'RoBERTa', 'edu-61M', 'biLM-61M', 'biLM-113M', 'biLM-224M']
hex_colors = get_probe_color_list(probe_labels)

# ============================================================================
# Plot 1: Accuracy Frontiers
# ============================================================================

# Calculate FLOPs for each model (using same formula as other scripts)
df['flops'] = 6 * df['params'] * 20 * df['params']

frontier_data = []
for _, row in df.iterrows():
    for loss_type in ['target', 'average_parallel']:
        frontier_data.append({
            'params': row['params'],
            'flops': row['flops'],
            'probe': row['probe'],
            'ood_loss': row['ood'],
            'x_loss': row[loss_type],
            'loss_type': loss_type
        })

frontier_df = pd.DataFrame(frontier_data)
frontier_df['probe_label'] = frontier_df['probe'].apply(get_probe_label)

loss_type_labels = {
    'target': r'Non-Medical Loss ($\downarrow$)',
    'average_parallel': r'Biology Loss ($\downarrow$)'
}
frontier_df['loss_type_label'] = frontier_df['loss_type'].map(loss_type_labels)

# force the order of facets by converting to categorical with specific order
facet_order = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']
frontier_df['loss_type_label'] = pd.Categorical(frontier_df['loss_type_label'], categories=facet_order, ordered=True)

frontier_df['probe_label'] = pd.Categorical(frontier_df['probe_label'], categories=probe_labels, ordered=True)

print(frontier_df)

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
iso_curves_df['loss_type_label'] = iso_curves_df['loss_type'].map(loss_type_labels)
iso_curves_df['loss_type_label'] = pd.Categorical(iso_curves_df['loss_type_label'], categories=facet_order, ordered=True)

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

# Sort for proper line connections
frontier_df = frontier_df.sort_values(['probe_label', 'loss_type', 'params'])

# Facet labels to use as x-axis titles (in facet order)
facet_x_titles = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']

p = (ggplot(frontier_df, aes(x='x_loss', y='ood_loss', color='probe_label', group='probe_label'))
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    + facet_wrap('~loss_type_label', ncol=2, scales='free_x')
    + scale_x_log10(name='')
    + scale_y_log10(name=r'Medical Loss ($\uparrow$)')
    + scale_color_manual(values=hex_colors)
    + guides(color=guide_legend(ncol=5))
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(3.375, 2.34),
            strip_text=element_blank(),
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
            legend_key_size=7,
            legend_key_spacing_x=2,
            plot_background=element_rect(fill=bg_color),
            panel_background=element_rect(fill=bg_color),
            panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
            legend_background=element_rect(fill=bg_color))
)

# Draw the plot and add per-facet x-axis labels using matplotlib
fig = p.draw()
for ax, title in zip(fig.axes[:2], facet_x_titles):
    ax.set_xlabel(title, fontsize=9, color=text_color, fontfamily='Helvetica Neue')
fig.savefig('plots/fixed-accuracy-frontiers.png', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/fixed-accuracy-frontiers.svg', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/fixed-accuracy-frontiers.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print("Saved plots/fixed-accuracy-frontiers.png")

# ============================================================================
# Plot 2: AUROC vs Relative AUC
# ============================================================================

# Load AUROC scores from fixed-probe-scaling.csv
auroc_probe_df = pd.read_csv('results/fixed-probe-scaling.csv')
auroc_probe_df = auroc_probe_df[(auroc_probe_df['dataset'] == 'test_tokens') & (auroc_probe_df['metric'] == 'auroc')]

# Map model names to probe labels
auroc_model_to_probe = {
    'roberta-edu': 'RoBERTa',
    'edu-61M': 'edu-61M',
    'pubmed-61M': 'biLM-61M',
    'pubmed-113M': 'biLM-113M',
    'pubmed-224M': 'biLM-224M',
}
auroc_probe_df['probe_label'] = auroc_probe_df['model'].map(auroc_model_to_probe)
auroc_scores = auroc_probe_df.set_index('probe_label')['score'].to_dict()

# Compute AUC under each frontier curve using trapezoidal rule
def compute_frontier_auc(df, probe_label, loss_type):
    """Compute area under the frontier curve (x_loss vs ood_loss) using trapezoidal rule."""
    data = df[(df['probe_label'] == probe_label) & (df['loss_type'] == loss_type)].copy()
    if len(data) < 2:
        return np.nan

    # Sort by x_loss for proper integration
    data = data.sort_values('x_loss')
    x = data['x_loss'].values
    y = data['ood_loss'].values

    # Use trapezoidal rule (np.trapz)
    return np.trapz(y, x)

# Compute AUC for each probe and loss type
auc_data = []
for loss_type in ['target', 'average_parallel']:
    for probe_label in frontier_df['probe_label'].unique():
        auc_val = compute_frontier_auc(frontier_df, probe_label, loss_type)
        auc_data.append({
            'probe_label': probe_label,
            'loss_type': loss_type,
            'auc': auc_val
        })

auc_df = pd.DataFrame(auc_data)

# Compute relative AUC as 1 - (nomask_auc / mask_auc)
relative_auc_data = []
for loss_type in ['target', 'average_parallel']:
    nomask_auc = auc_df[(auc_df['probe_label'] == 'No Filtering') & (auc_df['loss_type'] == loss_type)]['auc'].values[0]

    for _, row in auc_df[auc_df['loss_type'] == loss_type].iterrows():
        probe_label = row['probe_label']
        if probe_label == 'No Filtering':
            continue  # Skip baseline

        auroc = auroc_scores.get(probe_label, np.nan)
        if np.isnan(auroc):
            continue  # Skip if no AUROC score

        mask_auc = row['auc']
        relative_auc = 1 - (nomask_auc / mask_auc)
        relative_auc_data.append({
            'probe_label': probe_label,
            'loss_type': loss_type,
            'relative_auc': relative_auc,
            'auroc': auroc
        })

relative_auc_df = pd.DataFrame(relative_auc_data)
loss_type_labels_auc = {
    'target': r'Non-Medical ($\uparrow$)',
    'average_parallel': r'Biology ($\uparrow$)'
}
facet_order_auc = [r'Non-Medical ($\uparrow$)', r'Biology ($\uparrow$)']
relative_auc_df['loss_type_label'] = relative_auc_df['loss_type'].map(loss_type_labels_auc)
relative_auc_df['loss_type_label'] = pd.Categorical(relative_auc_df['loss_type_label'], categories=facet_order_auc, ordered=True)
relative_auc_df['probe_label'] = pd.Categorical(relative_auc_df['probe_label'], categories=probe_labels, ordered=True)

# Save results
relative_auc_df.to_csv('results/fixed-accuracy-auroc-vs-auc.csv', index=False)
print(f"Saved results/fixed-accuracy-auroc-vs-auc.csv")

# Create AUROC vs Relative AUC plot (excluding No Filtering from legend)
probe_labels_no_baseline = ['ModernBERT', 'RoBERTa', 'edu-61M', 'biLM-61M', 'biLM-113M', 'biLM-224M']
hex_colors_no_baseline = {p: c for p, c in zip(probe_labels_no_baseline, get_probe_color_list(probe_labels_no_baseline))}
relative_auc_df['probe_label'] = pd.Categorical(relative_auc_df['probe_label'], categories=probe_labels_no_baseline, ordered=True)

p2 = (ggplot(relative_auc_df, aes(x='auroc', y='relative_auc', color='probe_label'))
    + geom_point(size=3, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=3, color="#4f4f4f")
    + facet_wrap('~loss_type_label', ncol=2, scales='free')
    + scale_color_manual(values=hex_colors_no_baseline)
    + guides(color=guide_legend(nrow=2))
    + base_theme(base_family='Helvetica Neue')
    + labs(x='Classifier AUROC', y='Normalized AUC')
    + theme(figure_size=(3.375, 2.5),
            strip_text=element_text(size=9, color=text_color),
            strip_background=element_blank(),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            legend_position='top',
            legend_direction='horizontal',
            axis_title_x=element_text(size=9, color=text_color),
            axis_text_x=element_text(size=7, color=text_color),
            axis_title_y=element_text(size=9, color=text_color),
            axis_text_y=element_text(size=7, color=text_color),
            legend_text=element_text(size=7, color=text_color),
            legend_key_size=7,
            legend_key_spacing_x=2,
            plot_background=element_rect(fill=bg_color),
            panel_background=element_rect(fill=bg_color),
            panel_border=element_rect(color=line_color, size=0.5),
            axis_ticks=element_line(size=0.5),
            axis_ticks_minor=element_blank(),
            legend_background=element_rect(fill=bg_color))
)

p2.save('plots/fixed-accuracy-auroc-vs-auc.png', dpi=300, width=3.375, height=2.5)
p2.save('plots/fixed-accuracy-auroc-vs-auc.svg', dpi=300, width=3.375, height=2.5)
p2.save('plots/fixed-accuracy-auroc-vs-auc.pdf', dpi=300, width=3.375, height=2.5)
print("Saved plots/fixed-accuracy-auroc-vs-auc.png")
