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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use Helvetica Neue for mathtext
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica Neue:bold'
plt.rcParams['mathtext.sf'] = 'Helvetica Neue'
import pickle
import tiktoken
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH
from colors import get_threshold_colors, THEME_COLORS

parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=str, default='cuda')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'threshold'))
parser.add_argument('--baseline_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'))
parser.add_argument('--data_path',  type=str, default=os.path.join(DATA_PATH, 'test'))
parser.add_argument('--eval_iters', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--block_size', type=int, default=1024)
parser.add_argument('--rerun_probe', action='store_true', help='rerun probe threshold accuracy computation')
parser.add_argument('--rerun_models', action='store_true', help='rerun model evaluation')
parser.add_argument('--rerun', action='store_true', help='rerun both probe and model evaluation')
args = parser.parse_args()

device = args.device
data_dir = args.data_path
model_path = args.model_path
eval_iters = args.eval_iters
batch_size = args.batch_size
block_size = args.block_size

# determine the device type
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']
base_theme = theme_bw

# Model sizes to exclude from plots
excluded_params = [13e6, 28e6]

# setup the context manager
# ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else torch.no_grad()

# ============================================================================
# Probe Evaluation at Different Thresholds
# ============================================================================

def load_gpt_for_probe(model_file):
    """Load GPT model from checkpoint for probe evaluation"""
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

def load_probe_model(model_name, probe_model_path):
    """Load the base model based on tokenizer type for probe"""
    print(f'Loading left-{model_name}.pt and right-{model_name}.pt')
    left_model = load_gpt_for_probe(os.path.join(probe_model_path, f'left-{model_name}.pt'))
    right_model = load_gpt_for_probe(os.path.join(probe_model_path, f'right-{model_name}.pt'))
    left_model.to(device)
    right_model.to(device)
    left_model.eval()
    right_model.eval()
    
    # create combined config
    model_config = left_model.config
    model_config.hidden_size = left_model.config.n_embd * 2  # Concatenated features
    model_config.num_hidden_layers = left_model.config.n_layer
    
    return {'left': left_model, 'right': right_model}, model_config

def load_probe(model_name, probe_model_path):
    """Load the trained probe"""
    probe_path = os.path.join(probe_model_path.replace('bidir', ''), f'probes/{model_name}-token.pkl')
    
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    
    return probe_data['probe'], probe_data['layer']

def get_gpt_hidden_states(left_model, right_model, x, layer):
    """Get concatenated hidden states from dual GPT models"""
    with torch.no_grad():
        # Left model
        tok_emb_left = left_model.transformer.wte(x)
        h_left = left_model.transformer.drop(tok_emb_left)
        if layer > 0:
            for i in range(layer):
                h_left = left_model.transformer.h[i](h_left)
        
        # Right model
        tok_emb_right = right_model.transformer.wte(x)
        h_right = right_model.transformer.drop(tok_emb_right)
        if layer > 0:
            for i in range(layer):
                h_right = right_model.transformer.h[i](h_right)
        
        # Concatenate
        h_concat = torch.cat([h_left, h_right], dim=-1)
    return h_concat

def extract_features_for_probe(model, tokens, layer):
    """Extract features from the model at specified layer"""
    x = torch.from_numpy(tokens.astype(np.int64)).to(device)
    features = get_gpt_hidden_states(model['left'], model['right'], x, layer)
    return features.cpu().numpy()

enc = tiktoken.get_encoding("cl100k_base")
special_token_ids = {0, enc.eot_token}

def extract_and_balance_features(model, hidden_size, layer, dataset, probe_folder='probe', n_batches=3200):
    """Extract features and labels from dataset, returning balanced set"""    
    # load data
    probe_data_path = data_dir.replace('test', probe_folder)
    tokens_path = os.path.join(probe_data_path, 'token', f'{dataset}.bin')
    labels_path = os.path.join(probe_data_path, 'token', f'{dataset.replace("tokens", "labels")}.bin')

    tokens = np.memmap(tokens_path, dtype=np.uint32, mode='r')
    labels = np.memmap(labels_path, dtype=bool, mode='r')

    all_features = []
    all_labels = []
    
    probe_block_size = 512
    probe_batch_size = 16
    chunk_size = probe_batch_size * probe_block_size
    n_chunks = min(n_batches // probe_batch_size, (len(tokens) - probe_block_size) // chunk_size)
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(tokens) - probe_block_size)
        
        if start_idx >= end_idx:
            break
            
        # Extract chunk
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        
        # Reshape into batches
        n_seqs = len(chunk_tokens) // probe_block_size
        if n_seqs == 0:
            continue
            
        chunk_tokens = chunk_tokens[:n_seqs * probe_block_size].reshape(n_seqs, probe_block_size)
        chunk_labels = chunk_labels[:n_seqs * probe_block_size].reshape(n_seqs, probe_block_size)
        
        # extract features
        features = extract_features_for_probe(model, chunk_tokens, layer)
        features = features.reshape(-1, hidden_size)
        chunk_labels = chunk_labels.reshape(-1)
        
        # filter out special tokens
        token_ids = chunk_tokens.reshape(-1)
        mask = np.ones(len(token_ids), dtype=bool)
        for special_id in special_token_ids:
            mask &= (token_ids != special_id)
        
        if mask.sum() > 0:
            all_features.append(features[mask])
            all_labels.append(chunk_labels[mask])
        
    # combine all features and labels
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # balance labels
    min_count = min(labels.sum(), (~labels).sum())
    print(f"balancing labels to {min_count} medical and {min_count} non-medical")
    
    medical_indices     = np.where(labels)[0]
    non_medical_indices = np.where(~labels)[0]

    np.random.seed(42)
    sampled_medical     = np.random.choice(medical_indices, min_count, replace=False)
    sampled_non_medical = np.random.choice(non_medical_indices, min_count, replace=False)
    
    balanced_indices = np.concatenate([sampled_medical, sampled_non_medical])
    np.random.shuffle(balanced_indices)

    features = features[balanced_indices]
    labels = labels[balanced_indices]
    
    return features, labels

def evaluate_at_thresholds(probabilities, labels, thresholds, dataset):
    """Evaluate metrics at multiple thresholds given pre-computed probabilities"""
    results = []
    
    # AUROC is threshold-independent
    auroc = roc_auc_score(labels, probabilities)
    
    for threshold in thresholds:
        # make predictions at this threshold
        predictions = (probabilities >= threshold).astype(int)
        
        # calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        
        results.append({
            'threshold': threshold,
            'dataset': dataset,
            'auroc': auroc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results

# Evaluate pubmed-224M probe at different thresholds
if 'threshold-accuracies.csv' not in os.listdir('results') or args.rerun or args.rerun_probe:
    print("\n" + "="*80)
    print("Evaluating pubmed-224M probe at different thresholds")
    print("="*80 + "\n")
    
    # Load thresholds from CSV
    threshold_csv = pd.read_csv('../config/thresholds.csv')
    threshold_values = threshold_csv['threshold'].dropna().tolist()
    
    # Load pubmed-224M probe and model ONCE
    probe_model_path = os.path.join(MODEL_PATH, 'bidir')
    model_name = 'pubmed-224M'
    
    print(f"Loading {model_name} model and probe...")
    probe_model, model_config = load_probe_model(model_name, probe_model_path)
    probe, layer = load_probe(model_name, probe_model_path)
    hidden_size = model_config.hidden_size
    
    print(f"Loaded probe for {model_name} at layer {layer}")
    print(f"Evaluating at thresholds: {threshold_values}")
    
    threshold_results = []
    
    # Process each dataset
    for dataset in ['tokens', 'test_tokens']:
        print(f"\nProcessing {dataset}...")
        
        # Extract features and get probabilities ONCE per dataset
        features, labels = extract_and_balance_features(
            probe_model, hidden_size, layer, dataset
        )
        
        print(f"Computing probabilities for {len(labels)} samples...")
        probabilities = probe.predict_proba(features)[:, 1]
        
        # Evaluate at all thresholds using the same probabilities
        print(f"Evaluating at {len(threshold_values)} thresholds...")
        results = evaluate_at_thresholds(probabilities, labels, threshold_values, dataset)
        threshold_results.extend(results)
        
        # Print results
        for result in results:
            print(f"  Threshold {result['threshold']:.5f}: F1={result['f1']:.4f}, Precision={result['precision']:.4f}, Recall={result['recall']:.4f}")
    
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv('results/threshold-accuracies.csv', index=False)
    print(f"\nSaved results to results/threshold-accuracies.csv")
else:
    print("Loading existing threshold-accuracies.csv")
    threshold_df = pd.read_csv('results/threshold-accuracies.csv')

# ============================================================================
# Model Evaluation at Different Thresholds
# ============================================================================

models = []
for file in os.listdir("../config/threshold"):
    
    if file.endswith('.yaml'):
        models.append('mask-' + file.split('-')[1] + '-' + file.split('-')[2].split('.')[0] +'.pt')

baseline_models = []
for file in os.listdir("../config/scaling"):
    if '1030M' in file:
        continue
    if file.endswith('.yaml') and 'nomask' in file:
        baseline_models.append(file.split('-')[2].split('.')[0] + '-' + file.split('-')[1] + '.pt')

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
    
    return {'target' : out['target'].mean(), 'ood' : out['ood'].mean(), 'parallel' : out['parallel'].mean(), 'parallel_hard' : out['parallel_hard'].mean()}

thresholds = pd.read_csv('../config/thresholds.csv')
thresholds['model_idx'] = [int(t * 100) for t in thresholds['threshold']]
thresholds['num_tokens'] = (0.5 * (0.5 ** np.arange(len(thresholds['threshold'])))).tolist()

model_to_tokens = {idx: tokens for idx, tokens in zip(thresholds['model_idx'], thresholds['num_tokens'])}

if 'threshold-scaling.csv' not in os.listdir('results') or args.rerun or args.rerun_models:

    df = pd.DataFrame(columns=['params', 'threshold', 'num_tokens', 'target', 'ood', 'parallel', 'parallel_hard'])

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

        df.loc[len(df)] = {
            'params'   : int(model_file.split('-')[1][:-1]) * 1e6,
            'threshold' : int(model_file.split('-')[-1].split('.')[0]) / 100,
            'num_tokens' : model_to_tokens[int(model_file.split('-')[-1].split('.')[0])],
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
            'threshold' : 1,
            'num_tokens' : 0,
            'target'   : loss['target'].item(),
            'ood'      : loss['ood'].item(),
            'parallel' : loss['parallel'].item(),
            'parallel_hard' : loss['parallel_hard'].item()
        }

    df.to_csv('results/threshold-scaling.csv', index=False)
else:
    df = pd.read_csv('results/threshold-scaling.csv')

df['params'] = pd.to_numeric(df['params'])

# Filter baseline (num_tokens=0) to only include model sizes that exist for threshold models
threshold_params = df[df['num_tokens'] > 0]['params'].unique()
df = df[(df['num_tokens'] > 0) | (df['params'].isin(threshold_params))].copy()

# Filter out 13M and 28M models from plots
df = df[~df['params'].isin(excluded_params)].copy()

# Compute average_parallel as average of parallel and parallel_hard
df['average_parallel'] = (df['parallel'] + df['parallel_hard']) / 2

# plotnine needs long dataframe
df_long = pd.melt(df, 
                  id_vars=['params', 'num_tokens'], 
                  value_vars=['target', 'ood', 'parallel', 'parallel_hard', 'average_parallel'],
                  var_name='loss_type', 
                  value_name='loss')

num_token_labels = {num_tokens: f'{int(num_tokens * 100)}%' for num_tokens in thresholds['num_tokens']}
num_token_labels[0.0] = 'Baseline'

loss_type_labels = {
    'target': 'Non-medical',
    'ood': 'Medical', 
    'parallel': 'Biology',
    'parallel_hard': 'Biochemistry',
    'average_parallel': 'Avg Parallel'
}
df_long['loss_type_label'] = df_long['loss_type'].map(loss_type_labels)

# force the order of facets by converting to categorical with specific order
facet_order = ['Non-medical', 'Medical', 'Biology', 'Biochemistry', 'Avg Parallel']
df_long['loss_type_label'] = pd.Categorical(df_long['loss_type_label'], categories=facet_order, ordered=True)

# Ensure proper ordering by num_tokens (numeric values)
unique_tokens = sorted(df_long['num_tokens'].unique())
df_long['num_tokens'] = pd.Categorical(
    df_long['num_tokens'], 
    categories=unique_tokens,
    ordered=True
)
df_long['num_tokens_label'] = df_long['num_tokens'].map(num_token_labels)

df['compute'] = 6 * df['params'] * 12 * df['params']

# Loss frontiers plot - facets by loss type, colors by num_tokens
frontier_data = []
for _, row in df.iterrows():
    for loss_type in ['target', 'average_parallel']:
        frontier_data.append({
            'params': row['params'],
            'compute': row['compute'],
            'num_tokens': row['num_tokens'],
            'ood_loss': row['ood'],
            'x_loss': row[loss_type],
            'loss_type': loss_type
        })

frontier_df = pd.DataFrame(frontier_data)

# apply labels for loss types
x_loss_labels = {
    'target': r'Non-Medical Loss ($\downarrow$)',
    'average_parallel': r'Biology Loss ($\downarrow$)'
}
frontier_df['loss_type_label'] = frontier_df['loss_type'].map(x_loss_labels)

# force the order of facets by loss type
x_facet_order = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']
frontier_df['loss_type_label'] = pd.Categorical(frontier_df['loss_type_label'], categories=x_facet_order, ordered=True)

# Fix ordering for frontier plot
frontier_unique_tokens = sorted(frontier_df['num_tokens'].unique())
frontier_df['num_tokens'] = pd.Categorical(
    frontier_df['num_tokens'], 
    categories=frontier_unique_tokens,
    ordered=True
)
frontier_df['num_tokens_label'] = frontier_df['num_tokens'].map(num_token_labels)

# Use Cold palette for threshold sweep, but override "Baseline" with nomask color
unique_labels = list(frontier_df['num_tokens_label'].unique())
threshold_labels = [l for l in unique_labels if l != 'Baseline']

# Sort threshold labels numerically (by the percentage value)
def extract_pct(label):
    return int(label.replace('%', ''))
threshold_labels_sorted = sorted(threshold_labels, key=extract_pct, reverse=True)

cold_colors = get_threshold_colors(len(threshold_labels_sorted))

# Create color mapping, overriding "Baseline" with nomask color
from colors import MASK_COLORS
hex_colors = {'Baseline': MASK_COLORS['nomask']}
for i, label in enumerate(threshold_labels_sorted):
    hex_colors[label] = cold_colors[i]

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
        
        compute = param_data['compute'].values[0]
        
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
                'compute': compute,
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

iso_curves_df['flops_label'] = iso_curves_df['compute'].apply(format_flops)

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
frontier_df = frontier_df.sort_values(['num_tokens_label', 'loss_type', 'params'])

# Facet labels to use as x-axis titles (in facet order)
facet_x_titles = [r'Non-Medical Loss ($\downarrow$)', r'Biology Loss ($\downarrow$)']

# Different colors for different num_tokens levels
p_frontier = (ggplot(frontier_df, aes(x='x_loss', y='ood_loss', color='num_tokens_label', group='num_tokens_label'))
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    + facet_wrap('~loss_type_label', ncol=2, scales='free_x')
    + scale_x_log10(name='')
    + scale_y_log10(name=r'Medical Loss ($\uparrow$)')
    + scale_color_manual(values=hex_colors, name='% Filtered')
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
fig.savefig('plots/threshold-loss-frontiers.png', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/threshold-loss-frontiers.svg', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/threshold-loss-frontiers.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print("Saved plots/threshold-loss-frontiers.png")
