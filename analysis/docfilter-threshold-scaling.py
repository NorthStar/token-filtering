"""
Compare token-level filtering (threshold) vs document-level filtering (docfilter-threshold)
Creates loss frontier plots for 521M models.
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from plotnine import *
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
from colors import MASK_COLORS, MASK_LABELS, THEME_COLORS, get_mask_color_list

parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=str, default='cuda')
parser.add_argument('--threshold_path', type=str, default=os.path.join(MODEL_PATH, 'threshold-old'))
parser.add_argument('--docfilter_path', type=str, default=os.path.join(MODEL_PATH, 'docfilter-threshold'))
parser.add_argument('--data_path',  type=str, default=os.path.join(DATA_PATH, 'test'))
parser.add_argument('--eval_iters', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--block_size', type=int, default=2048)
parser.add_argument('--rerun', action='store_true', help='rerun model evaluation')
args = parser.parse_args()

device = args.device
data_dir = args.data_path
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

# Mask colors and labels for this comparison
mask_order = ['document', 'mask', 'remove']
mask_colors = get_mask_color_list(mask_order)
legend_labels = {m: MASK_LABELS[m] for m in mask_order}

# Model sizes to evaluate
target_params = [521]  # [521]

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

# ============================================================================
# Evaluate threshold models (token-level filtering)
# ============================================================================

if 'docfilter-threshold-scaling.csv' not in os.listdir('results') or args.rerun:

    df = pd.DataFrame(columns=['params', 'mask_type', 'threshold', 'target', 'ood', 'parallel', 'parallel_hard'])

    # Find and evaluate threshold models (token filtering)
    # Model files are named: mask-{params}M-{threshold_idx}.pt
    # where threshold_idx is int(threshold * 100)
    if os.path.exists(args.threshold_path):
        for model_file in os.listdir(args.threshold_path):
            if not model_file.endswith('.pt'):
                continue
            if not model_file.startswith('mask-'):
                continue
            
            # Parse model file name: mask-{params}M-{threshold_idx}.pt
            parts = model_file.replace('.pt', '').split('-')
            if len(parts) < 3:
                continue
            
            try:
                params_m = int(parts[1].replace('M', ''))
            except:
                continue
            
            # Only evaluate target model sizes
            if params_m not in target_params:
                continue

            try:
                threshold_idx = int(parts[2])
                threshold = threshold_idx / 100  # convert back to actual threshold
            except:
                continue
            
            try:
                model = load_model(os.path.join(args.threshold_path, model_file))
                print(f"evaluating threshold {model_file}...")
                model.eval()
                model.to(device)

                loss = estimate_test_loss(model, batch_size=16, eval_iters=20)

                df.loc[len(df)] = {
                    'params': params_m * 1e6,
                    'mask_type': 'mask',
                    'threshold': threshold,
                    'target': loss['target'].item(),
                    'ood': loss['ood'].item(),
                    'parallel': loss['parallel'].item(),
                    'parallel_hard': loss['parallel_hard'].item()
                }

                # Free memory
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: failed to load/evaluate {model_file}: {e}")
                continue

        # Find and evaluate remove models (token removal)
        for model_file in os.listdir(args.threshold_path):
            if not model_file.endswith('.pt'):
                continue
            if not model_file.startswith('remove-'):
                continue

            # Parse model file name: remove-{params}M-{threshold_idx}.pt
            parts = model_file.replace('.pt', '').split('-')
            if len(parts) < 3:
                continue

            try:
                params_m = int(parts[1].replace('M', ''))
            except:
                continue

            # Only evaluate target params models
            if params_m not in target_params:
                continue

            try:
                threshold_idx = int(parts[2])
                threshold = threshold_idx / 100  # convert back to actual threshold
            except:
                continue

            try:
                model = load_model(os.path.join(args.threshold_path, model_file))
                print(f"evaluating remove {model_file}...")
                model.eval()
                model.to(device)

                loss = estimate_test_loss(model, batch_size=16, eval_iters=20)

                df.loc[len(df)] = {
                    'params': params_m * 1e6,
                    'mask_type': 'remove',
                    'threshold': threshold,
                    'target': loss['target'].item(),
                    'ood': loss['ood'].item(),
                    'parallel': loss['parallel'].item(),
                    'parallel_hard': loss['parallel_hard'].item()
                }

                # Free memory
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: failed to load/evaluate {model_file}: {e}")
                continue

    # Find and evaluate docfilter-threshold models (document filtering)
    # Model files are named: docfilter-{params}M-{threshold_pct}.pt
    # where threshold_pct is int(threshold * 10000)
    if os.path.exists(args.docfilter_path):
        for model_file in os.listdir(args.docfilter_path):
            if not model_file.endswith('.pt'):
                continue
            if not model_file.startswith('docfilter-'):
                continue
            
            # Parse model file name: docfilter-{params}M-{threshold_pct}.pt
            parts = model_file.replace('.pt', '').split('-')
            if len(parts) < 3:
                continue
            
            try:
                params_m = int(parts[1].replace('M', ''))
            except:
                continue
            
            # Only evaluate target model sizes
            if params_m not in target_params:
                continue

            try:
                threshold_pct = int(parts[2])
                threshold = threshold_pct / 10000  # convert back to actual threshold
            except:
                continue
            
            try:
                model = load_model(os.path.join(args.docfilter_path, model_file))
                print(f"evaluating docfilter {model_file}...")
                model.eval()
                model.to(device)

                loss = estimate_test_loss(model, batch_size=16, eval_iters=20)

                df.loc[len(df)] = {
                    'params': params_m * 1e6,
                    'mask_type': 'document',
                    'threshold': threshold,
                    'target': loss['target'].item(),
                    'ood': loss['ood'].item(),
                    'parallel': loss['parallel'].item(),
                    'parallel_hard': loss['parallel_hard'].item()
                }

                # Free memory
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: failed to load/evaluate {model_file}: {e}")
                continue

    df.to_csv('results/docfilter-threshold-scaling.csv', index=False)
else:
    df = pd.read_csv('results/docfilter-threshold-scaling.csv')

df['params'] = pd.to_numeric(df['params'])

# Compute average_parallel as average of parallel and parallel_hard
df['average_parallel'] = (df['parallel'] + df['parallel_hard']) / 2

print(f"\nLoaded {len(df)} model evaluations")
print(df.groupby(['params', 'mask_type']).size())

# Document-level plotting removed - using token-level evaluation only

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

print("\n" + "="*80)
print("Token-level Evaluation (medical/non-medical tokens)")
print("="*80 + "\n")

if 'docfilter-threshold-scaling-tokens.csv' not in os.listdir('results') or args.rerun:

    df_tokens = pd.DataFrame(columns=['params', 'mask_type', 'threshold', 'target', 'ood', 'parallel', 'parallel_hard'])

    # Find and evaluate threshold models (token filtering)
    if os.path.exists(args.threshold_path):
        for model_file in os.listdir(args.threshold_path):
            if not model_file.endswith('.pt'):
                continue
            if not model_file.startswith('mask-'):
                continue

            parts = model_file.replace('.pt', '').split('-')
            if len(parts) < 3:
                continue

            try:
                params_m = int(parts[1].replace('M', ''))
            except:
                continue

            if params_m not in target_params:
                continue

            try:
                threshold_idx = int(parts[2])
                threshold = threshold_idx / 100
            except:
                continue

            try:
                model = load_model(os.path.join(args.threshold_path, model_file))
                print(f"[tokens] evaluating threshold {model_file}...")
                model.eval()
                model.to(device)

                loss = estimate_test_loss_tokens(model, batch_size=16, eval_iters=20)

                df_tokens.loc[len(df_tokens)] = {
                    'params': params_m * 1e6,
                    'mask_type': 'mask',
                    'threshold': threshold,
                    'target': loss['target'].item(),
                    'ood': loss['ood'].item(),
                    'parallel': loss['parallel'].item(),
                    'parallel_hard': loss['parallel_hard'].item()
                }

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: [tokens] failed to load/evaluate {model_file}: {e}")
                continue

        # Find and evaluate remove models (token removal)
        for model_file in os.listdir(args.threshold_path):
            if not model_file.endswith('.pt'):
                continue
            if not model_file.startswith('remove-'):
                continue

            parts = model_file.replace('.pt', '').split('-')
            if len(parts) < 3:
                continue

            try:
                params_m = int(parts[1].replace('M', ''))
            except:
                continue

            if params_m not in target_params:
                continue

            try:
                threshold_idx = int(parts[2])
                threshold = threshold_idx / 100
            except:
                continue

            try:
                model = load_model(os.path.join(args.threshold_path, model_file))
                print(f"[tokens] evaluating remove {model_file}...")
                model.eval()
                model.to(device)

                loss = estimate_test_loss_tokens(model, batch_size=16, eval_iters=20)

                df_tokens.loc[len(df_tokens)] = {
                    'params': params_m * 1e6,
                    'mask_type': 'remove',
                    'threshold': threshold,
                    'target': loss['target'].item(),
                    'ood': loss['ood'].item(),
                    'parallel': loss['parallel'].item(),
                    'parallel_hard': loss['parallel_hard'].item()
                }

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: [tokens] failed to load/evaluate {model_file}: {e}")
                continue

    # Find and evaluate docfilter-threshold models (document filtering)
    if os.path.exists(args.docfilter_path):
        for model_file in os.listdir(args.docfilter_path):
            if not model_file.endswith('.pt'):
                continue
            if not model_file.startswith('docfilter-'):
                continue

            parts = model_file.replace('.pt', '').split('-')
            if len(parts) < 3:
                continue

            try:
                params_m = int(parts[1].replace('M', ''))
            except:
                continue

            if params_m not in target_params:
                continue

            try:
                threshold_pct = int(parts[2])
                threshold = threshold_pct / 10000
            except:
                continue

            try:
                model = load_model(os.path.join(args.docfilter_path, model_file))
                print(f"[tokens] evaluating docfilter {model_file}...")
                model.eval()
                model.to(device)

                loss = estimate_test_loss_tokens(model, batch_size=16, eval_iters=20)

                df_tokens.loc[len(df_tokens)] = {
                    'params': params_m * 1e6,
                    'mask_type': 'document',
                    'threshold': threshold,
                    'target': loss['target'].item(),
                    'ood': loss['ood'].item(),
                    'parallel': loss['parallel'].item(),
                    'parallel_hard': loss['parallel_hard'].item()
                }

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: [tokens] failed to load/evaluate {model_file}: {e}")
                continue

    df_tokens.to_csv('results/docfilter-threshold-scaling-tokens.csv', index=False)
else:
    df_tokens = pd.read_csv('results/docfilter-threshold-scaling-tokens.csv')

df_tokens['params'] = pd.to_numeric(df_tokens['params'])
df_tokens['average_parallel'] = (df_tokens['parallel'] + df_tokens['parallel_hard']) / 2

print(f"\n[tokens] Loaded {len(df_tokens)} model evaluations")
print(df_tokens.groupby(['params', 'mask_type']).size())

# ============================================================================
# Evaluate unfiltered and random init 521M models
# ============================================================================

baseline_results_file = 'results/docfilter-baseline-521M.csv'
if baseline_results_file.split('/')[-1] not in os.listdir('results') or args.rerun:
    baseline_results = {}

    # Load unfiltered 521M model
    unfiltered_path = os.path.join(MODEL_PATH, 'gpt', 'nomask-521M.pt')
    if os.path.exists(unfiltered_path):
        print("Evaluating unfiltered 521M model...")
        model = load_model(unfiltered_path)
        model.eval()
        model.to(device)
        loss = estimate_test_loss_tokens(model, batch_size=16, eval_iters=20)
        baseline_results['unfiltered'] = {
            'ood': loss['ood'].item(),
            'parallel': loss['parallel'].item(),
            'parallel_hard': loss['parallel_hard'].item(),
            'average_parallel': (loss['parallel'].item() + loss['parallel_hard'].item()) / 2
        }
        print(f"Unfiltered 521M: ood={loss['ood']:.4f}, avg_parallel={baseline_results['unfiltered']['average_parallel']:.4f}")
        del model
        torch.cuda.empty_cache()
    else:
        print(f"Warning: unfiltered model not found at {unfiltered_path}")

    # Create random init 521M model
    print("Evaluating random init 521M model...")
    # Load config from an existing 521M model
    existing_521M = None
    for model_file in os.listdir(args.threshold_path):
        if '521M' in model_file and model_file.endswith('.pt'):
            existing_521M = os.path.join(args.threshold_path, model_file)
            break
    if existing_521M is None and os.path.exists(unfiltered_path):
        existing_521M = unfiltered_path

    if existing_521M:
        checkpoint = torch.load(existing_521M, map_location=device)
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)  # Random init, no weights loaded
        model.eval()
        model.to(device)
        loss = estimate_test_loss_tokens(model, batch_size=16, eval_iters=20)
        baseline_results['random_init'] = {
            'ood': loss['ood'].item(),
            'parallel': loss['parallel'].item(),
            'parallel_hard': loss['parallel_hard'].item(),
            'average_parallel': (loss['parallel'].item() + loss['parallel_hard'].item()) / 2
        }
        print(f"Random init 521M: ood={loss['ood']:.4f}, avg_parallel={baseline_results['random_init']['average_parallel']:.4f}")
        del model
        torch.cuda.empty_cache()
    else:
        print("Warning: could not find a 521M model config for random init")

    # Save baseline results
    baseline_df = pd.DataFrame([
        {'model_type': 'unfiltered', **baseline_results.get('unfiltered', {})},
        {'model_type': 'random_init', **baseline_results.get('random_init', {})}
    ])
    baseline_df.to_csv(baseline_results_file, index=False)
    print(f"Saved {baseline_results_file}")
else:
    baseline_df = pd.read_csv(baseline_results_file)
    baseline_results = {}
    for _, row in baseline_df.iterrows():
        baseline_results[row['model_type']] = {
            'ood': row['ood'],
            'average_parallel': row['average_parallel']
        }

# ============================================================================
# Token-level loss frontier plot: Medical vs Biology
# ============================================================================

# Create plot for 521M models (token-level)
df_tokens_521M = df_tokens[df_tokens['params'] == 521e6].copy()
if len(df_tokens_521M) > 0:
    # Apply legend labels
    df_tokens_521M['mask_label'] = df_tokens_521M['mask_type'].map(legend_labels)
    df_tokens_521M['mask_label'] = pd.Categorical(
        df_tokens_521M['mask_label'],
        categories=[legend_labels[m] for m in mask_order],
        ordered=True
    )

    # Sort for proper line connections
    df_tokens_521M = df_tokens_521M.sort_values(['mask_label', 'average_parallel'])

    # Get baseline point if available
    nomask_color = MASK_COLORS['nomask']
    baseline_df = None
    if 'unfiltered' in baseline_results:
        baseline_df = pd.DataFrame({
            'x': [baseline_results['unfiltered']['average_parallel']],
            'y': [baseline_results['unfiltered']['ood']],
            'mask_label': ['Baseline']
        })

    # Extended color list including baseline (nomask) first
    extended_colors = [nomask_color] + mask_colors
    extended_labels = ['Baseline'] + [legend_labels[m] for m in mask_order]

    # Base plot (plain version)
    p_tokens_521M = (ggplot(df_tokens_521M, aes(x='average_parallel', y='ood', color='mask_label', group='mask_label'))
        + geom_line(size=1)
        + geom_point(size=2, stroke=0, alpha=0.9)
        + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    )
    # Add baseline point if available
    if baseline_df is not None:
        p_tokens_521M = (p_tokens_521M
            + geom_point(aes(x='x', y='y', color='mask_label'), data=baseline_df, size=2, stroke=0, alpha=0.9,
                         inherit_aes=False)
            + geom_point(aes(x='x', y='y'), data=baseline_df, size=2,
                         fill="none", stroke=0.7, color="#4f4f4f", inherit_aes=False)
        )
    p_tokens_521M = (p_tokens_521M
        + scale_x_log10(name=r'Biology Loss ($\downarrow$)')
        + scale_y_log10(name=r'Medical Loss ($\uparrow$)')
        + scale_color_manual(values=extended_colors, limits=extended_labels)
        + guides(color=guide_legend(nrow=1))
        + base_theme(base_family='Helvetica Neue')
        + theme(figure_size=(3.375, 2.5),
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
                plot_background=element_rect(fill=bg_color),
                panel_background=element_rect(fill=bg_color),
                panel_border=element_rect(color=line_color, size=0.5),
                axis_ticks=element_line(size=0.5),
                axis_ticks_minor=element_blank(),
                legend_background=element_rect(fill=bg_color))
    )
    p_tokens_521M.save('plots/docfilter-threshold-frontiers-521M-tokens.png', dpi=300, width=3.375, height=2.5)
    p_tokens_521M.save('plots/docfilter-threshold-frontiers-521M-tokens.svg', dpi=300, width=3.375, height=2.5)
    p_tokens_521M.save('plots/docfilter-threshold-frontiers-521M-tokens.pdf', dpi=300, width=3.375, height=2.5)
    print("Saved plots/docfilter-threshold-frontiers-521M-tokens.png")

    # ============================================================================
    # Annotated plot with baseline, random init, and star
    # ============================================================================

    if 'unfiltered' in baseline_results and 'random_init' in baseline_results:
        unfiltered_x = baseline_results['unfiltered']['average_parallel']
        unfiltered_y = baseline_results['unfiltered']['ood']
        random_x = baseline_results['random_init']['average_parallel']
        random_y = baseline_results['random_init']['ood']

        # Intersection point: x from unfiltered, y from random_init
        star_x = unfiltered_x
        star_y = random_y

        # Get nomask (baseline) color
        nomask_color = MASK_COLORS['nomask']

        # Create dataframes for annotation points
        unfiltered_df = pd.DataFrame({'x': [unfiltered_x], 'y': [unfiltered_y]})
        random_df = pd.DataFrame({'x': [random_x], 'y': [random_y]})
        star_df = pd.DataFrame({'x': [star_x], 'y': [star_y]})

        # Dashed lines: vertical from unfiltered going up, horizontal from random going left
        vline_df = pd.DataFrame({
            'x': [unfiltered_x, unfiltered_x],
            'y': [unfiltered_y, star_y]
        })
        hline_df = pd.DataFrame({
            'x': [random_x, star_x],
            'y': [random_y, random_y]
        })

        # Annotated plot
        p_annotated = (ggplot(df_tokens_521M, aes(x='average_parallel', y='ood', color='mask_label', group='mask_label'))
            + geom_line(size=1)
            + geom_point(size=2, stroke=0, alpha=0.9)
            + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
            # Horizontal dashed line from random to star (at y=random_y)
            + geom_segment(aes(x=random_x, xend=star_x, y='y', yend='y'),
                           data=pd.DataFrame({'y': [random_y]}),
                           linetype='dashed', color='black', size=0.7, inherit_aes=False)
            # Vertical dashed line from unfiltered to star (at x=unfiltered_x)
            + geom_segment(aes(x='x', xend='x', y=unfiltered_y, yend=star_y),
                           data=pd.DataFrame({'x': [unfiltered_x]}),
                           linetype='dashed', color='black', size=0.7, inherit_aes=False)
            # Yellow star at intersection
            + geom_point(aes(x='x', y='y'), data=star_df, shape='*', size=3,
                         color='#FFD700', fill='#FFD700', inherit_aes=False)
            # Baseline model (nomask color)
            + geom_point(aes(x='x', y='y'), data=unfiltered_df, size=2, stroke=0, alpha=0.9,
                         color=nomask_color, fill=nomask_color, inherit_aes=False)
            + geom_point(aes(x='x', y='y'), data=unfiltered_df, size=2,
                         fill="none", stroke=0.7, color="#4f4f4f", inherit_aes=False)
            # Random init model (black)
            + geom_point(aes(x='x', y='y'), data=random_df, size=2, stroke=0, alpha=0.9,
                         color='black', fill='black', inherit_aes=False)
            + geom_point(aes(x='x', y='y'), data=random_df, size=2,
                         fill="none", stroke=0.7, color="#4f4f4f", inherit_aes=False)
            + scale_x_log10(name=r'Biology Loss ($\downarrow$)')
            + scale_y_log10(name=r'Medical Loss ($\uparrow$)')
            + scale_color_manual(values=mask_colors)
            + guides(color=guide_legend(nrow=1))
            + base_theme(base_family='Helvetica Neue')
            + theme(figure_size=(3.375, 2.5),
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
                    plot_background=element_rect(fill=bg_color),
                    panel_background=element_rect(fill=bg_color),
                    panel_border=element_rect(color=line_color, size=0.5),
                    axis_ticks=element_line(size=0.5),
                    axis_ticks_minor=element_blank(),
                    legend_background=element_rect(fill=bg_color))
        )
        p_annotated.save('plots/docfilter-threshold-frontiers-521M-tokens-annotated.png', dpi=300, width=3.375, height=2.5)
        p_annotated.save('plots/docfilter-threshold-frontiers-521M-tokens-annotated.svg', dpi=300, width=3.375, height=2.5)
        p_annotated.save('plots/docfilter-threshold-frontiers-521M-tokens-annotated.pdf', dpi=300, width=3.375, height=2.5)
        print("Saved plots/docfilter-threshold-frontiers-521M-tokens-annotated.png")
    else:
        print("Baseline results not available for annotated plot")
else:
    print("No 521M models found for token-level evaluation")