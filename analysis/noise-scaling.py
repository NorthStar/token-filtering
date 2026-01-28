"""
Evaluate noise sweep models on test sets and create scaling law plots.
Models are trained with different noise levels applied to the mask.
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.linear_model import LinearRegression
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH
from colors import get_noise_colors, MODEL_SIZE_COLORS, get_model_size_color_list, THEME_COLORS

# ============================================================================
# Setup and Configuration
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=str, default='cuda')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'noise-sweep'))
parser.add_argument('--baseline_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'))
parser.add_argument('--data_path',  type=str, default=os.path.join(DATA_PATH, 'test'))
parser.add_argument('--eval_iters', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
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

# ============================================================================
# Accuracy Calculation
# ============================================================================

# Load base accuracy from probe-scaling.csv
probe_df = pd.read_csv('results/probe-scaling.csv')
base_accuracy = probe_df[
    (probe_df['model'] == 'pubmed-224M') & 
    (probe_df['dataset'] == 'test_tokens') & 
    (probe_df['metric'] == 'accuracy')
]['score'].values[0]

print(f"Base accuracy (A) for pubmed-224M: {base_accuracy:.4f}")

def noise_to_accuracy(noise_level, A=base_accuracy):
    """Convert noise level N to accuracy: accuracy = N + A(1 - 2N)"""
    return noise_level + A * (1 - 2 * noise_level)

# Define noise levels and compute corresponding accuracies
noise_levels = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
noise_to_acc = {n: noise_to_accuracy(n) for n in noise_levels}

print("\nNoise level to accuracy mapping:")
for n, acc in noise_to_acc.items():
    print(f"  Noise {n:.2f} -> Accuracy {acc:.4f}")

# ============================================================================
# Model Discovery
# ============================================================================

models = []
for file in os.listdir("../config/noise-sweep"):
    if file.endswith('.yaml'):
        # gpt-113M-0.15.yaml -> noise-113M-15.pt
        parts = file.replace('.yaml', '').split('-')
        params = parts[1]  # e.g., "113M"
        noise = float(parts[2])  # e.g., 0.15
        noise_int = int(noise * 100)  # e.g., 15
        models.append({
            'file': f'noise-{params}-{noise_int}.pt',
            'params': params,
            'noise': noise
        })

# Also include baseline mask models (noise = 0)
baseline_models = []
for file in os.listdir("../config/scaling"):
    if file.endswith('.yaml') and 'mask' in file and 'nomask' not in file:
        # gpt-113M-mask.yaml -> mask-113M.pt
        parts = file.replace('.yaml', '').split('-')
        params = parts[1]  # e.g., "113M"
        if params not in ['1030M', '1816M']:  # Skip very large models
            baseline_models.append({
                'file': f'mask-{params}.pt',
                'params': params,
                'noise': 0.0
            })

# Include nomask (unfiltered) models as baseline for compute ratio calculation
nomask_models = []
for file in os.listdir("../config/scaling"):
    if file.endswith('.yaml') and 'nomask' in file:
        # gpt-113M-nomask.yaml -> nomask-113M.pt
        parts = file.replace('.yaml', '').split('-')
        params = parts[1]  # e.g., "113M"
        if params not in ['1030M', '1816M']:  # Skip very large models
            nomask_models.append({
                'file': f'nomask-{params}.pt',
                'params': params,
            })

print(f"\nFound {len(models)} noise models, {len(baseline_models)} mask baselines, {len(nomask_models)} nomask baselines")

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
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    return model

loader_dtype = np.uint32

def get_test_batch():
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
def estimate_test_loss(model):
    out = {'target': torch.zeros(eval_iters), 'ood': torch.zeros(eval_iters), 
           'parallel': torch.zeros(eval_iters), 'parallel_hard': torch.zeros(eval_iters)}
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

if 'noise-scaling.csv' not in os.listdir('results') or args.rerun:
    
    df = pd.DataFrame(columns=['params', 'noise', 'accuracy', 'target', 'ood', 'parallel', 'parallel_hard'])
    
    # Evaluate noise models
    for model_info in models:
        try:
            model = load_model(os.path.join(model_path, model_info['file']))
        except Exception as e:
            print(f"Error loading model {model_info['file']}: {e}")
            continue
        
        print(f"Evaluating {model_info['file']}...")
        model.eval()
        model.to(device)
        
        loss = estimate_test_loss(model)
        
        params_val = int(model_info['params'].replace('M', '')) * 1e6
        accuracy = noise_to_accuracy(model_info['noise'])
        
        df.loc[len(df)] = {
            'params': params_val,
            'noise': model_info['noise'],
            'accuracy': accuracy,
            'target': loss['target'].item(),
            'ood': loss['ood'].item(),
            'parallel': loss['parallel'].item(),
            'parallel_hard': loss['parallel_hard'].item()
        }
        
        print(f"  target: {loss['target']:.4f} | ood: {loss['ood']:.4f}")
    
    # Evaluate baseline models (noise = 0)
    for model_info in baseline_models:
        try:
            model = load_model(os.path.join(args.baseline_path, model_info['file']))
        except Exception as e:
            print(f"Error loading baseline model {model_info['file']}: {e}")
            continue
        
        print(f"Evaluating baseline {model_info['file']}...")
        model.eval()
        model.to(device)
        
        loss = estimate_test_loss(model)
        
        params_val = int(model_info['params'].replace('M', '')) * 1e6
        accuracy = noise_to_accuracy(0.0)  # Base accuracy
        
        df.loc[len(df)] = {
            'params': params_val,
            'noise': 0.0,
            'accuracy': accuracy,
            'target': loss['target'].item(),
            'ood': loss['ood'].item(),
            'parallel': loss['parallel'].item(),
            'parallel_hard': loss['parallel_hard'].item()
        }
        
        print(f"  target: {loss['target']:.4f} | ood: {loss['ood']:.4f}")
    
    df.to_csv('results/noise-scaling.csv', index=False)
else:
    df = pd.read_csv('results/noise-scaling.csv')

# Evaluate nomask (unfiltered) models for baseline - stored separately
if 'noise-nomask-baseline.csv' not in os.listdir('results') or args.rerun:
    nomask_df = pd.DataFrame(columns=['params', 'target', 'ood', 'parallel', 'parallel_hard'])
    
    for model_info in nomask_models:
        try:
            model = load_model(os.path.join(args.baseline_path, model_info['file']))
        except Exception as e:
            print(f"Error loading nomask model {model_info['file']}: {e}")
            continue
        
        print(f"Evaluating nomask {model_info['file']}...")
        model.eval()
        model.to(device)
        
        loss = estimate_test_loss(model)
        
        params_val = int(model_info['params'].replace('M', '')) * 1e6
        
        nomask_df.loc[len(nomask_df)] = {
            'params': params_val,
            'target': loss['target'].item(),
            'ood': loss['ood'].item(),
            'parallel': loss['parallel'].item(),
            'parallel_hard': loss['parallel_hard'].item()
        }
        
        print(f"  target: {loss['target']:.4f} | ood: {loss['ood']:.4f}")
    
    nomask_df.to_csv('results/noise-nomask-baseline.csv', index=False)
else:
    nomask_df = pd.read_csv('results/noise-nomask-baseline.csv')

print(f"\nLoaded {len(df)} model results and {len(nomask_df)} nomask baselines")

# ============================================================================
# Data Preparation
# ============================================================================

df['params'] = pd.to_numeric(df['params'])
df['compute'] = 6 * df['params'] * 20 * df['params']

# Compute average_parallel as average of parallel and parallel_hard
df['average_parallel'] = (df['parallel'] + df['parallel_hard']) / 2
nomask_df['average_parallel'] = (nomask_df['parallel'] + nomask_df['parallel_hard']) / 2

# Create accuracy labels for legend
df['accuracy_label'] = df['accuracy'].apply(lambda a: f'{a:.1%}')

# Create parameter labels
df['params_label'] = df['params'].apply(lambda p: f'{int(p // 1e6)}M')

# Order accuracy levels for consistent coloring
unique_accuracies = sorted(df['accuracy'].unique(), reverse=True)
accuracy_labels = [f'{a:.1%}' for a in unique_accuracies]
df['accuracy_label'] = pd.Categorical(df['accuracy_label'], categories=accuracy_labels, ordered=True)

# Setup color palette - use Green-Gold for noise/accuracy levels
n_categories = len(unique_accuracies)
hex_colors = get_noise_colors(n_categories)

# ============================================================================
# Model sizes to exclude from plots
# ============================================================================

excluded_params = [13e6, 28e6]

# Add compute to nomask baseline
nomask_df['params'] = pd.to_numeric(nomask_df['params'])
nomask_df['compute'] = 6 * nomask_df['params'] * 20 * nomask_df['params']

# Filter out 13M and 28M models
df_filtered = df[~df['params'].isin(excluded_params)].copy()
nomask_filtered = nomask_df[~nomask_df['params'].isin(excluded_params)].copy()

# ============================================================================
# Accuracy vs Compute Ratio Plot (by model size) - relative to unfiltered baseline
# ============================================================================

# Use filtered data (excluding 13M and 28M)
acc_vs_compute_data = []

for loss_type in ['ood']:  # Focus on medical (target) loss only
    # Fit scaling law on nomask (unfiltered) baseline
    baseline_data = nomask_filtered.copy()
    if len(baseline_data) < 2:
        continue
    
    baseline_data['log_compute'] = np.log(baseline_data['compute'])
    baseline_data['log_loss'] = np.log(baseline_data[loss_type])
    
    X = baseline_data['log_compute'].values.reshape(-1, 1)
    y = baseline_data['log_loss'].values
    reg = LinearRegression().fit(X, y)
    
    # Calculate compute ratios for all filtered models
    for _, row in df_filtered.iterrows():
        log_loss = np.log(row[loss_type])
        log_baseline_compute_needed = (log_loss - reg.intercept_) / reg.coef_[0]
        baseline_compute_needed = np.exp(log_baseline_compute_needed)
        compute_ratio = baseline_compute_needed / row['compute']
        
        acc_vs_compute_data.append({
            'params': row['params'],
            'params_label': f"{int(row['params'] // 1e6)}M",
            'accuracy': row['accuracy'],
            'compute_ratio': compute_ratio,
            'loss_type': loss_type
        })

if len(acc_vs_compute_data) > 0:
    acc_compute_df = pd.DataFrame(acc_vs_compute_data)
    
    # Add error rate (1 - accuracy)
    acc_compute_df['error_rate'] = 1 - acc_compute_df['accuracy']
    
    # Order by params for consistent colors
    param_order = sorted(acc_compute_df['params'].unique())
    param_labels_ordered = [f'{int(p // 1e6)}M' for p in param_order]
    acc_compute_df['params_label'] = pd.Categorical(
        acc_compute_df['params_label'], 
        categories=param_labels_ordered, 
        ordered=True
    )
    
    # Get colors for params - use Blue-Yellow discrete (consistent with accuracy-scaling)
    hex_param_colors = get_model_size_color_list(param_labels_ordered)
    
    # Sort for proper line connections
    acc_compute_df = acc_compute_df.sort_values(['params_label', 'error_rate'])
    
    p_acc_compute = (ggplot(acc_compute_df, aes(x='error_rate', y='compute_ratio', color='params_label', group='params_label'))
        + geom_line(size=1)
        + geom_point(size=2, stroke=0, alpha=0.9)
        + geom_point(fill="none", stroke=0.5, size=2, color="#4f4f4f")
        + scale_x_log10(name='Classifier Error Rate', labels=lambda l: [f'{x:.0%}' for x in l])
        + scale_y_log10(name='Loss-Matched Baseline Compute')
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

    p_acc_compute.save('plots/noise-accuracy-vs-compute-ratio.png', dpi=300, width=3.375, height=2.5)
    p_acc_compute.save('plots/noise-accuracy-vs-compute-ratio.svg', dpi=300, width=3.375, height=2.5)
    p_acc_compute.save('plots/noise-accuracy-vs-compute-ratio.pdf', dpi=300, width=3.375, height=2.5)
    print("Saved plots/noise-accuracy-vs-compute-ratio.png")
