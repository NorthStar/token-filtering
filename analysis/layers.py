"""
train a probe across layers of a masked vs. unmasked model
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
try:
    from cuml import LogisticRegression
    import cupy as cp
    USE_CUML = True
except ImportError:
    from sklearn.linear_model import LogisticRegression
    cp = None
    USE_CUML = False
from sklearn.metrics import precision_recall_curve
from plotnine import *
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH
from colors import MASK_COLORS, MASK_LABELS, THEME_COLORS, get_mask_color_list

# Theme configuration (matching noise-scaling.py and adversarial-finetune-analysis.py)
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']
base_theme = theme_bw

# Mask types to analyze
mask_types = ['nomask', 'mask', 'document', 'remove']
mask_order = ['document', 'mask', 'remove', 'nomask']
legend_labels = {
    'document': 'Document',
    'mask': 'Token (Masking)',
    'remove': 'Token (Removal)',
    'nomask': 'Baseline',
}
mask_colors_list = get_mask_color_list(mask_order)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'), help='path to model directory')
parser.add_argument('--data_path', type=str, default=os.path.join(DATA_PATH, 'probe', 'token'), help='directory containing probe data')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_batches', type=int, default=125)
parser.add_argument('--block_size', type=int, default=2048)
parser.add_argument('--rerun', action='store_true', help='rerun results')
args = parser.parse_args()

batch_size = args.batch_size
n_batches = args.n_batches
block_size = args.block_size

MAX_LENGTH = 2**10 - 1

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def load_model(model_path):

    print(f"loading model from {model_path}")

    checkpoint = torch.load(model_path)
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

def get_batch(data_path, split):
    
    if split == 'train':
        data_filename = os.path.join(data_path, 'tokens.bin')
        labels_filename = os.path.join(data_path, 'labels.bin')
    else:  # 'test'
        data_filename = os.path.join(data_path, 'test_tokens.bin')
        labels_filename = os.path.join(data_path, 'test_labels.bin')
    
    data = np.memmap(data_filename, dtype=np.uint32, mode='r')
    labels = np.memmap(labels_filename, dtype=bool, mode='r')

    indices = torch.randint(0, len(data) - block_size, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in indices])
    y = torch.stack([torch.from_numpy((labels[i:i+block_size]).astype(np.int64)) for i in indices])

    return x.to(device), y.to(device)

def collect_features_and_labels(model, layer, data_path, split):

    all_features = []
    all_labels = []
    
    print(f"collecting features for layer {layer} from {split} split...")
    for _ in tqdm(range(n_batches)):
        x, y = get_batch(data_path, split)

        # cheeky half forward pass
        with torch.no_grad():
            tok_emb = model.transformer.wte(x)
            h = model.transformer.drop(tok_emb)
            
            if layer > 0:
                for i in range(layer):
                    h = model.transformer.h[i](h)
            
            # features = h
            # # mean-pool over sequence length
            # doc_features = features.mean(dim=1)
            
        features = h.cpu().numpy()
        all_features.append(features.reshape(-1, features.shape[-1]))
        all_labels.append(y.cpu().numpy().reshape(-1))
    
    return np.vstack(all_features), np.concatenate(all_labels)

def batched_predict_proba(probe, features, batch_size=50000):
    """Predict probabilities in batches to avoid GPU OOM with large feature arrays."""
    n_samples = len(features)
    probas = []
    
    for i in range(0, n_samples, batch_size):
        batch = features[i:i+batch_size]
        proba = probe.predict_proba(batch)
        # Convert cupy array to numpy if needed
        if hasattr(proba, 'get'):
            proba = proba.get()
        probas.append(proba[:, 1])  # Get probability of positive class
        # Free GPU memory between batches
        cp.get_default_memory_pool().free_all_blocks()
    
    return np.concatenate(probas)

def compute_optimal_f1(y_true, y_proba):
    """Compute F1 score using threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores)
    return f1_scores[best_idx]

def train_probe(model, layer, data_path):

    train_features, train_labels = collect_features_and_labels(model, layer, data_path, 'train')
    test_features, test_labels = collect_features_and_labels(model, layer, data_path, 'test')

    print(f'train_features.shape: {train_features.shape}')
    print(f'train_labels.shape: {train_labels.shape}')
    
    probe = LogisticRegression(max_iter=1000)
    print(f"training probe @ layer {layer}...")
    probe.fit(train_features, train_labels)
    
    # Get probabilities for positive class using batched prediction
    train_proba = batched_predict_proba(probe, train_features)
    test_proba = batched_predict_proba(probe, test_features)
    
    # Compute F1 with optimal threshold
    train_f1 = compute_optimal_f1(train_labels, train_proba)
    test_f1 = compute_optimal_f1(test_labels, test_proba)
    
    print(f"layer {layer}: train f1 {train_f1:.4f}, test f1 {test_f1:.4f}")
    
    # Clean up GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    
    return test_f1

if f'probe-f1.csv' not in os.listdir('results') or args.rerun:

    # define model variants to test
    all_results = []
    for model_file in os.listdir(args.model_path):
        if not model_file.endswith('.pt'):
            continue
        
        if 'collapse' in model_file:
            continue

        # Parse params early to filter
        try:
            num_params = int(model_file.split('-')[1].split('M')[0]) * 1e6
        except (IndexError, ValueError):
            continue
        
        model_path = os.path.join(args.model_path, model_file)

        print(f"\nprocessing {model_file}...")
        model = load_model(model_path)
        model.to(device)
        model.eval()

        num_layers = model.config.n_layer
        mask_type = model_file.split('-')[0]

        # embedding layer
        test_f1 = train_probe(model, 0, args.data_path)
        all_results.append({'layer': 0, 'test_f1': test_f1, 'mask_type': mask_type, 'params': num_params})

        # transformer layers
        for layer in range(1, num_layers + 1):
            test_f1 = train_probe(model, layer, args.data_path)
            all_results.append({'layer': layer, 'test_f1': test_f1, 'mask_type': mask_type, 'params': num_params})
        
        # clean up gpu memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    # save results
    df = pd.DataFrame(all_results)
    df.to_csv(f'results/probe-f1.csv', index=False)
    print(f'results saved to probe-f1.csv')

else:
    df = pd.read_csv(f'results/probe-f1.csv')

# Filter out 13M and 28M models for plotting
df = df[~df['params'].isin([13e6, 28e6])]

# =============================================================================
# Data Preparation for Plots
# =============================================================================

# Apply legend labels to mask_type
df['mask_label'] = df['mask_type'].replace(legend_labels)
df['mask_label'] = pd.Categorical(df['mask_label'],
                                   categories=[legend_labels[m] for m in mask_order],
                                   ordered=True)

# Create params label
df['params_label'] = df['params'].map(lambda x: f'{int(x // 1e6)}M')
params_categories = sorted(df['params_label'].unique(), key=lambda x: int(x.replace('M', '')))
df['params_label'] = pd.Categorical(df['params_label'], categories=params_categories, ordered=True)

# Get max F1 for each mask/params combo
df_max_f1 = df.groupby(['mask_type', 'params']).agg({'test_f1': 'max'}).reset_index()
df_max_f1['mask_label'] = df_max_f1['mask_type'].replace(legend_labels)
df_max_f1['mask_label'] = pd.Categorical(df_max_f1['mask_label'],
                                          categories=[legend_labels[m] for m in mask_order],
                                          ordered=True)
df_max_f1['params_label'] = df_max_f1['params'].map(lambda x: f'{int(x // 1e6)}M')

# X-axis breaks and labels for model size plots
breaks = sorted(df_max_f1['params'].unique())
labels = [f'{int(p / 1e6)}M' for p in breaks]

# Get largest model size
largest_params = df['params'].max()
largest_params_label = f'{int(largest_params // 1e6)}M'

os.makedirs('plots', exist_ok=True)

# =============================================================================
# Plot 1: Max F1 across layers vs model size (3.375 x 2.5 in)
# =============================================================================

# Baseline F1 from probe-scaling.py: pubmed-224M validation F1
baseline_f1 = 0.854

p1 = (ggplot(df_max_f1, aes(x='params', y='test_f1', color='mask_label'))
    + geom_hline(yintercept=baseline_f1, color='#000000', size=0.5)
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.5, size=2, color="#4f4f4f")
    + scale_x_log10(name='Model Size', breaks=breaks, labels=labels)
    + scale_y_continuous(name='Forget vs. Retain Classification F1')
    + scale_color_manual(values=mask_colors_list)
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(3.375, 2.5),
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

p1.save('plots/probe-f1-scaling.png', dpi=300, width=3.375, height=2.5)
p1.save('plots/probe-f1-scaling.svg', dpi=300, width=3.375, height=2.5)
p1.save('plots/probe-f1-scaling.pdf', dpi=300, width=3.375, height=2.5)
print('Saved plots/probe-f1-scaling.png')

# =============================================================================
# Plot 2: F1 across layers, faceted by model size (7 x 3.5 in)
# =============================================================================

p2 = (ggplot(df, aes(x='layer', y='test_f1', color='mask_label'))
    + geom_hline(yintercept=baseline_f1, color='#000000', size=0.5)
    + geom_line(size=1)
    + geom_point(size=1.5, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.4, size=1.5, color="#4f4f4f")
    + facet_wrap('~params_label', ncol=3, scales='free_x')
    + scale_x_continuous(name='Layer')
    + scale_y_continuous(name='F1')
    + scale_color_manual(values=mask_colors_list)
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(7, 2.625),
            strip_text_x=element_text(size=9, color=text_color),
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

p2.save('plots/probe-f1-layers.png', dpi=300, width=7, height=2.625)
p2.save('plots/probe-f1-layers.svg', dpi=300, width=7, height=2.625)
p2.save('plots/probe-f1-layers.pdf', dpi=300, width=7, height=2.625)
print('Saved plots/probe-f1-layers.png')

# =============================================================================
# Plot 3: F1 across layers for largest model only (3.375 x 2.5 in)
# =============================================================================

df_largest = df[df['params'] == largest_params].copy()

p3 = (ggplot(df_largest, aes(x='layer', y='test_f1', color='mask_label'))
    + geom_hline(yintercept=baseline_f1, color='#000000', size=0.5)
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.5, size=2, color="#4f4f4f")
    + scale_x_continuous(name='Layer')
    + scale_y_continuous(name='F1')
    + scale_color_manual(values=mask_colors_list)
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(3.375, 2.5),
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

p3.save(f'plots/probe-f1-{largest_params_label}.png', dpi=300, width=3.375, height=2.5)
p3.save(f'plots/probe-f1-{largest_params_label}.svg', dpi=300, width=3.375, height=2.5)
p3.save(f'plots/probe-f1-{largest_params_label}.pdf', dpi=300, width=3.375, height=2.5)
print(f'Saved plots/probe-f1-{largest_params_label}.png')