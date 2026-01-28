"""
Train probes across layers to distinguish between domains:
- domain1 (infectious diseases) vs. normal text
- domain2 (neurology) vs. normal text
- domain1 vs. domain2
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
from sklearn.metrics import accuracy_score
from plotnine import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Configure matplotlib to use Helvetica Neue for mathtext
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica Neue:bold'
plt.rcParams['mathtext.sf'] = 'Helvetica Neue'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH
from colors import MASK_COLORS, MASK_LABELS, get_mask_color_list, THEME_COLORS

mask_order = ['nomask', 'document', 'mask', 'remove']
mask_colors = get_mask_color_list(mask_order)
legend_labels = [MASK_LABELS[m] for m in mask_order]

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'), help='path to model directory')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_batches', type=int, default=16)
parser.add_argument('--block_size', type=int, default=2048)
parser.add_argument('--rerun', action='store_true', help='rerun results')
args = parser.parse_args()

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

if 'probe-two-way.csv' not in os.listdir('results') or args.rerun:

    batch_size = args.batch_size
    n_batches = args.n_batches
    block_size = args.block_size

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Data paths
    domain1_path = os.path.join(DATA_PATH, 'two-way', 'domain1.bin')
    domain2_path = os.path.join(DATA_PATH, 'two-way', 'domain2.bin')
    normal_path = os.path.join(DATA_PATH, 'test', 'test_parallel.bin')

    # Contrasts to evaluate
    CONTRASTS = [
        ('domain1_vs_normal', domain1_path, normal_path, 'Non-medical vs.\nInfectious Diseases'),
        ('domain2_vs_normal', domain2_path, normal_path, 'Non-medical vs.\nNeurology'),
        ('domain1_vs_domain2', domain1_path, domain2_path, 'Infectious Diseases\nvs. Neurology'),
    ]

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

    def get_batch_from_files(path_class0, path_class1, effective_batch_size):
        """Sample sequences from two bin files and create binary labels."""
        data0 = np.memmap(path_class0, dtype=np.uint32, mode='r')
        data1 = np.memmap(path_class1, dtype=np.uint32, mode='r')

        half_batch = effective_batch_size // 2

        # Sample from class 0
        max_idx0 = len(data0) - block_size
        indices0 = torch.randint(0, max_idx0, (half_batch,))
        x0 = torch.stack([torch.from_numpy((data0[i:i+block_size]).astype(np.int64)) for i in indices0])

        # Sample from class 1
        max_idx1 = len(data1) - block_size
        indices1 = torch.randint(0, max_idx1, (half_batch,))
        x1 = torch.stack([torch.from_numpy((data1[i:i+block_size]).astype(np.int64)) for i in indices1])

        # Combine and create labels
        x = torch.cat([x0, x1], dim=0)
        y = torch.cat([torch.zeros(half_batch), torch.ones(half_batch)]).long()

        # Shuffle
        perm = torch.randperm(len(x))
        x = x[perm]
        y = y[perm]

        return x.to(device), y.to(device)

    def collect_features_and_labels(model, layer, path_class0, path_class1, effective_batch_size, effective_n_batches):
        """Collect features from a specific layer for binary classification."""
        all_features = []
        all_labels = []

        for _ in tqdm(range(effective_n_batches), desc=f'layer {layer}', leave=False):
            x, y = get_batch_from_files(path_class0, path_class1, effective_batch_size)

            with torch.no_grad():
                tok_emb = model.transformer.wte(x)
                h = model.transformer.drop(tok_emb)

                if layer > 0:
                    for i in range(layer):
                        h = model.transformer.h[i](h)

                # Mean-pool over sequence length
                features = h.mean(dim=1)

            all_features.append(features.cpu().numpy())
            all_labels.append(y.cpu().numpy())

        return np.vstack(all_features), np.concatenate(all_labels)

    def batched_predict(probe, features, batch_size=50000):
        """Predict in batches to avoid GPU OOM."""
        n_samples = len(features)
        preds = []

        for i in range(0, n_samples, batch_size):
            batch = features[i:i+batch_size]
            pred = probe.predict(batch)
            if hasattr(pred, 'get'):
                pred = pred.get()
            preds.append(pred)
            cp.get_default_memory_pool().free_all_blocks()

        return np.concatenate(preds)

    def train_probe(model, layer, path_class0, path_class1, effective_batch_size, effective_n_batches):
        """Train a probe at a specific layer and return accuracy."""
        # Use 20% for train, 80% for test
        train_batches = int(effective_n_batches * 0.2)
        test_batches = effective_n_batches - train_batches

        train_features, train_labels = collect_features_and_labels(
            model, layer, path_class0, path_class1, effective_batch_size, train_batches
        )
        test_features, test_labels = collect_features_and_labels(
            model, layer, path_class0, path_class1, effective_batch_size, test_batches
        )

        probe = LogisticRegression(max_iter=10000)
        probe.fit(train_features, train_labels)

        test_preds = batched_predict(probe, test_features)
        test_acc = accuracy_score(test_labels, test_preds)

        cp.get_default_memory_pool().free_all_blocks()

        return test_acc

    all_results = []
    
    # Find all 61M models
    model_files = [f for f in os.listdir(args.model_path) 
                   if f.endswith('.pt') and '-61M' in f and 'collapse' not in f]
    
    for model_file in model_files:
        model_path = os.path.join(args.model_path, model_file)
        mask_type = model_file.split('-')[0]
        
        print(f"\n{'='*60}")
        print(f"Processing {model_file} (mask_type={mask_type})")
        print(f"{'='*60}")
        
        model = load_model(model_path)
        model.to(device)
        model.eval()
        
        num_layers = model.config.n_layer
        
        for contrast_name, path0, path1, contrast_label in CONTRASTS:
            print(f"\n--- Contrast: {contrast_label} ---")
            
            # Embedding layer (layer 0)
            acc = train_probe(model, 0, path0, path1, batch_size, n_batches)
            all_results.append({
                'layer': 0,
                'accuracy': acc,
                'mask_type': mask_type,
                'contrast': contrast_name,
                'contrast_label': contrast_label
            })
            print(f"  layer 0: acc={acc:.4f}")
            
            # Transformer layers
            for layer in range(1, num_layers + 1):
                acc = train_probe(model, layer, path0, path1, batch_size, n_batches)
                all_results.append({
                    'layer': layer,
                    'accuracy': acc,
                    'mask_type': mask_type,
                    'contrast': contrast_name,
                    'contrast_label': contrast_label
                })
                print(f"  layer {layer}: acc={acc:.4f}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    df = pd.DataFrame(all_results)
    df.to_csv('results/probe-two-way.csv', index=False)
    print(f'\nResults saved to results/probe-two-way.csv')

else:
    df = pd.read_csv('results/probe-two-way.csv')

# Apply mask ordering and labels
df['mask_type'] = pd.Categorical(df['mask_type'], categories=mask_order, ordered=True)
df['mask_label'] = df['mask_type'].map(MASK_LABELS)
df['mask_label'] = pd.Categorical(df['mask_label'], categories=legend_labels, ordered=True)

# Format contrast labels with line breaks
contrast_label_map = {
    'Non-medical vs. Infectious Diseases': 'Infectious Diseases\nvs. Non-medical',
    'Non-medical vs. Neurology': 'Neurology\nvs. Non-medical',
    'Infectious Diseases vs. Neurology': 'Infectious Diseases\nvs. Neurology',
}
df['contrast_label'] = df['contrast_label'].replace(contrast_label_map)

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']
base_theme = theme_bw

# =============================================================================
# Plot: Bar plot of average accuracy across layers by contrast
# =============================================================================
df_avg = df.groupby(['mask_label', 'contrast', 'contrast_label']).agg({'accuracy': 'mean'}).reset_index()
df_avg['mask_label'] = pd.Categorical(df_avg['mask_label'], categories=legend_labels, ordered=True)

# Convert accuracy to percentage
df_avg['accuracy'] = df_avg['accuracy'] * 100

dodge_pos = position_dodge(width=0.8)
p = (
    ggplot(df_avg, aes(x='contrast_label', y='accuracy', fill='mask_label'))
    + geom_col(position=dodge_pos, width=0.7, color=line_color, size=0.3)
    + scale_y_continuous(name='Average Accuracy (%)', breaks=[90, 95, 100], labels=lambda x: [f'{int(v)}' for v in x])
    + coord_cartesian(ylim=(90, 100))
    + scale_x_discrete(name='')
    + scale_fill_manual(values=mask_colors)
    + guides(fill=guide_legend(nrow=1))
    + base_theme(base_family='Helvetica Neue')
    + theme(
        figure_size=(3.375, 2.5),
        panel_grid_major=element_line(size=0.3, color=grid_color),
        panel_grid_minor=element_blank(),
        legend_title=element_blank(),
        axis_title_x=element_text(size=9, color=text_color),
        axis_text_x=element_text(size=7, rotation=0, color=text_color, linespacing=1.5),
        axis_title_y=element_text(size=9, color=text_color),
        axis_text_y=element_text(size=7, color=text_color),
        legend_text=element_text(size=7, color=text_color),
        legend_key_size=7,
        legend_position='top',
        legend_direction='horizontal',
        plot_background=element_rect(fill=bg_color),
        panel_background=element_rect(fill=bg_color),
        panel_border=element_rect(color=line_color, size=0.5),
        axis_ticks=element_line(size=0.5),
        axis_ticks_minor=element_blank(),
        legend_background=element_rect(fill=bg_color)
    )
)

p.save('plots/probe-two-way.png', dpi=300, width=3.375, height=2.5)
p.save('plots/probe-two-way.svg', dpi=300, width=3.375, height=2.5)
p.save('plots/probe-two-way.pdf', dpi=300, width=3.375, height=2.5)
print("Plot saved as 'plots/probe-two-way.png', 'plots/probe-two-way.svg', and 'plots/probe-two-way.pdf'")
