"""
Analyze adversarial finetuning results: how many tokens to recover baseline performance?
"""

import os
import argparse
import glob
import torch
import pandas as pd
import numpy as np
from plotnine import *
from colors import MASK_COLORS, MASK_LABELS, THEME_COLORS, get_mask_color_list
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
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'))
parser.add_argument('--data_path', type=str, default=os.path.join(DATA_PATH, 'finetune'))
parser.add_argument('--results_dir', type=str, default='adversarial-finetune')
parser.add_argument('--eval_iters', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--block_size', type=int, default=2048)
parser.add_argument('--rerun_baseline', action='store_true', help='recompute baseline losses')
args = parser.parse_args()

device = args.device
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']
base_theme = theme_bw

# Use centralized mask colors and labels (matching mmlu.py/scaling.py style)
mask_order = ['document', 'mask', 'remove', 'rmu']
mask_colors_list = [MASK_COLORS[m] if m in MASK_COLORS else MASK_COLORS['unlearn'] for m in mask_order]
mask_colors_list[3] = MASK_COLORS['unlearn']  # rmu uses unlearn color
legend_labels = {
    'document': 'Document',
    'mask': 'Token (Masking)',
    'remove': 'Token (Removal)',
    'rmu': 'RMU',
}

# FLOPs breaks for x-axis (matching scaling.py)
flops_breaks = [1e18, 1e19, 1e20, 1e21]
flops_labels = [r'$10^{18}$', r'$10^{19}$', r'$10^{20}$', r'$10^{21}$']

# Model sizes to analyze
model_sizes = ['61M', '113M', '224M', '521M', '1030M', '1816M']
mask_types = ['mask', 'remove', 'document', 'rmu']

# Load pubmed test data
baseline_csv = os.path.join('results', 'adversarial-finetune-baseline.csv')
if not os.path.exists(baseline_csv) or args.rerun_baseline:
    test_data = np.memmap(os.path.join(args.data_path, 'pubmed_test.bin'), dtype=np.uint32, mode='r')
    print(f"Loaded pubmed_test.bin with {len(test_data):,} tokens")

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
    return model, model_args

def get_batch(batch_size, block_size):
    max_start_idx = len(test_data) - block_size - 1
    ix = torch.randint(0, max_start_idx, (batch_size,))
    x = torch.stack([torch.from_numpy((test_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((test_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, batch_size, block_size, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(batch_size, block_size)
        _, loss = model(x, idx_filter=None, targets=y, targets_filter=None)
        losses[k] = loss.item()
    return losses.mean().item()

# =============================================================================
# Step 1: Compute baseline losses for nomask models
# =============================================================================
os.makedirs('results', exist_ok=True)

if not os.path.exists(baseline_csv) or args.rerun_baseline:
    print("\n" + "="*60)
    print("Computing baseline losses for nomask models on pubmed_test")
    print("="*60 + "\n")
    
    baseline_data = []
    for size in model_sizes:
        model_file = os.path.join(args.model_path, f'nomask-{size}.pt')
        if not os.path.exists(model_file):
            print(f"Warning: {model_file} not found, skipping")
            continue
        
        print(f"Loading nomask-{size}...")
        model, model_args = load_model(model_file)
        model.to(device)
        block_size = model_args.get('block_size', args.block_size)
        
        # Use smaller batch for larger models
        batch_size = 4 if '1030M' in size or '1816M' in size else args.batch_size
        
        loss = estimate_loss(model, batch_size, block_size, args.eval_iters)
        print(f"  nomask-{size}: loss = {loss:.4f}")
        
        baseline_data.append({
            'size': size,
            'params': int(size[:-1]) * 1e6,
            'baseline_loss': loss
        })
        
        del model
        torch.cuda.empty_cache()
    
    baseline_df = pd.DataFrame(baseline_data)
    baseline_df.to_csv(baseline_csv, index=False)
    print(f"\nSaved baseline losses to {baseline_csv}")
else:
    baseline_df = pd.read_csv(baseline_csv)
    print(f"Loaded baseline losses from {baseline_csv}")

print("\nBaseline losses:")
print(baseline_df.to_string(index=False))

# Create a lookup dict for baseline losses
baseline_losses = dict(zip(baseline_df['size'], baseline_df['baseline_loss']))

# =============================================================================
# Step 2: Load finetuning results
# =============================================================================

print("\n" + "="*60)
print("Loading finetuning results")
print("="*60 + "\n")

# Find all result CSV files
csv_pattern = os.path.join(args.results_dir, '*-pubmed.csv')
csv_files = glob.glob(csv_pattern)
print(f"Found {len(csv_files)} result files")

all_results = []
for csv_file in csv_files:
    # Parse filename to get mask type and size
    # Format: {mask}-{size}-lr{lr}-pubmed.csv
    basename = os.path.basename(csv_file)
    parts = basename.replace('-pubmed.csv', '').split('-')
    
    if len(parts) < 3:
        print(f"  Skipping {basename}: unexpected format")
        continue
    
    mask_type = parts[0]
    size = parts[1]
    
    if mask_type not in mask_types or size not in model_sizes:
        print(f"  Skipping {basename}: mask={mask_type}, size={size}")
        continue
    
    df = pd.read_csv(csv_file)
    df['mask'] = mask_type
    df['size'] = size
    df['params'] = int(size[:-1]) * 1e6
    all_results.append(df)
    print(f"  Loaded {basename}: {len(df)} rows")

if not all_results:
    print("No valid result files found!")
    sys.exit(1)

results_df = pd.concat(all_results, ignore_index=True)
print(f"\nTotal rows: {len(results_df)}")

# =============================================================================
# Step 3: Find tokens required to reach baseline for each mask/size combo
# =============================================================================

print("\n" + "="*60)
print("Computing tokens to reach baseline")
print("="*60 + "\n")

tokens_to_baseline = []
for mask_type in mask_types:
    for size in model_sizes:
        subset = results_df[(results_df['mask'] == mask_type) & (results_df['size'] == size)]
        if len(subset) == 0:
            continue
        
        baseline = baseline_losses.get(size)
        if baseline is None:
            print(f"  No baseline for {size}, skipping")
            continue
        
        # Sort by tokens_seen
        subset = subset.sort_values('tokens_seen').reset_index(drop=True)
        
        # Find first index where test_loss <= baseline
        below_baseline_mask = subset['test_loss'] <= baseline
        
        if below_baseline_mask.any():
            # Get index of first point below baseline
            idx_after = below_baseline_mask.idxmax()
            
            if idx_after > 0:
                # Linear interpolation between point before and point after
                # Point A (before): higher loss
                # Point B (after): lower loss (below baseline)
                x_a = subset.loc[idx_after - 1, 'tokens_seen']
                y_a = subset.loc[idx_after - 1, 'test_loss']
                x_b = subset.loc[idx_after, 'tokens_seen']
                y_b = subset.loc[idx_after, 'test_loss']
                
                # Interpolate: find x where y = baseline
                # x = x_a + (baseline - y_a) * (x_b - x_a) / (y_b - y_a)
                if y_b != y_a:
                    tokens_needed = x_a + (baseline - y_a) * (x_b - x_a) / (y_b - y_a)
                else:
                    tokens_needed = x_b
            else:
                # First point is already below baseline
                tokens_needed = subset.loc[idx_after, 'tokens_seen']
            
            final_loss = subset.loc[idx_after, 'test_loss']
            reached = True
        else:
            # Didn't reach baseline - use the final tokens value
            tokens_needed = subset.iloc[-1]['tokens_seen']
            final_loss = subset.iloc[-1]['test_loss']
            reached = False
            print(f"  {mask_type}-{size}: did NOT reach baseline (final loss {final_loss:.4f} > baseline {baseline:.4f})")
        
        params = int(size[:-1]) * 1e6
        pretraining_tokens = 20 * params
        pct_of_pretraining = (tokens_needed / pretraining_tokens) * 100
        
        tokens_to_baseline.append({
            'mask': mask_type,
            'size': size,
            'params': params,
            'tokens_needed': tokens_needed,
            'baseline_loss': baseline,
            'final_loss': final_loss,
            'reached_baseline': reached,
            'pct_of_pretraining': pct_of_pretraining
        })
        
        status = "✓" if reached else "✗"
        print(f"  {status} {mask_type}-{size}: {tokens_needed:,.0f} tokens ({pct_of_pretraining:.2f}% of pretraining)")

tokens_df = pd.DataFrame(tokens_to_baseline)
tokens_df.to_csv(os.path.join('results', 'adversarial-finetune-tokens.csv'), index=False)

# =============================================================================
# Plot 1: Pretraining FLOPs vs % of pretraining compute
# =============================================================================

print("\n" + "="*60)
print("Creating plots")
print("="*60 + "\n")

os.makedirs('plots', exist_ok=True)

# Add FLOPs column (matching scaling.py: 6 * N * D where D = 20 * N)
tokens_df['flops'] = 6 * tokens_df['params'] * 20 * tokens_df['params']

# Apply legend labels to mask column
tokens_df['mask'] = tokens_df['mask'].replace(legend_labels)
tokens_df['mask'] = pd.Categorical(tokens_df['mask'],
                                    categories=[legend_labels[m] for m in mask_order],
                                    ordered=True)

p1 = (ggplot(tokens_df, aes(x='flops', y='pct_of_pretraining', color='mask'))
    + geom_line(size=1)
    + geom_point(size=2, stroke=0, alpha=0.9)
    + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
    + scale_x_log10(name='Pretraining Compute (FLOPs)',
                   breaks=flops_breaks,
                   labels=flops_labels)
    + scale_y_log10(name='Tokens of Adversarial Finetuning Required\nto Elicit Baseline (% of Pretraining)', breaks=[0.001, 0.01, 0.1])
    + scale_color_manual(values=mask_colors_list)
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(3.375, 2.5),
            panel_grid_major=element_line(size=0.3, color=grid_color),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            legend_position='top',
            legend_direction='horizontal',
            axis_title_x=element_text(size=9, color=text_color),
            axis_title_y=element_text(size=9, color=text_color, ha='center', margin={'r': -5}),
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

# Save with tight bbox, adjusting figure size to maintain target dimensions
target_width, target_height = 3.375, 2.5
fig = p1.draw()
fig.savefig('plots/adversarial-finetune-pct.png', dpi=300, bbox_inches='tight', pad_inches=0)
# Get the size of the tight bbox
from PIL import Image
img = Image.open('plots/adversarial-finetune-pct.png')
tight_width, tight_height = img.size[0] / 300, img.size[1] / 300
img.close()
plt.close(fig)
# Compute scale factors and adjust figure size
scale_w, scale_h = target_width / tight_width, target_height / tight_height
adjusted_width = target_width * scale_w
adjusted_height = target_height * scale_h
# Redraw with adjusted size
p1_adjusted = p1 + theme(figure_size=(adjusted_width, adjusted_height))
fig = p1_adjusted.draw()
fig.savefig('plots/adversarial-finetune-pct.png', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/adversarial-finetune-pct.svg', dpi=300, bbox_inches='tight', pad_inches=0)
fig.savefig('plots/adversarial-finetune-pct.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print("Saved plots/adversarial-finetune-pct.png")

# =============================================================================
# Plot 2: Finetuning curves (faceted by model size, log scale)
# =============================================================================

# Add baseline loss to results for plotting horizontal lines
results_df['baseline'] = results_df['size'].map(baseline_losses)

# Create baseline data for geom_hline
baseline_line_df = baseline_df.copy()
baseline_line_df['size'] = baseline_line_df['size'].astype(str)

# Order sizes for faceting
size_order = ['61M', '113M', '224M', '521M', '1030M', '1816M']

# Compute per-size axis limits
# x_max: 1.1x the max tokens needed to cross baseline across all mask types
# y_max: 1.1x the max loss for "remove" mask type
axis_limits = {}
for size in size_order:
    size_tokens = tokens_df[tokens_df['size'] == size]
    size_results = results_df[results_df['size'] == size]
    
    if len(size_tokens) == 0 or len(size_results) == 0:
        continue
    
    # x_max: 1.5x the max tokens needed to reach baseline
    x_max = size_tokens['tokens_needed'].max() * 1.5
    
    # y_max: 1.1x the max loss for "remove" mask type
    remove_data = size_results[size_results['mask'] == 'remove']
    if len(remove_data) > 0:
        remove_max_loss = remove_data['test_loss'].max()
    else:
        # Fallback to max across all masks if no remove data
        remove_max_loss = size_results['test_loss'].max()
    y_max = remove_max_loss * 1.1
    
    # Also get y_min for log scale (use baseline or min loss)
    y_min = size_results['test_loss'].min() * 0.95
    
    axis_limits[size] = {'x_max': x_max, 'y_max': y_max, 'y_min': y_min}
    print(f"  {size}: x_max={x_max:,.0f}, y_max={y_max:.3f} (remove max)")

# Filter results_df: filter x limits per size, keep all y data (will use coord_cartesian to crop)
# Also filter out steps 0 and 1 (start from step 2)
filtered_results = []
for size in size_order:
    if size not in axis_limits:
        continue
    size_data = results_df[results_df['size'] == size].copy()
    limits = axis_limits[size]
    # Filter out steps 0 and 1
    size_data = size_data[size_data['step'] >= 2]
    # Filter x only - keep all y data for coord_cartesian cropping
    size_data = size_data[size_data['tokens_seen'] <= limits['x_max']]
    filtered_results.append(size_data)

if filtered_results:
    plot_results_df = pd.concat(filtered_results, ignore_index=True)
else:
    plot_results_df = results_df.copy()

# Convert tokens to millions for cleaner axis
plot_results_df['tokens_M'] = plot_results_df['tokens_seen'] / 1e6

# Apply legend labels to mask column
plot_results_df['mask'] = plot_results_df['mask'].replace(legend_labels)
plot_results_df['mask'] = pd.Categorical(plot_results_df['mask'],
                                          categories=[legend_labels[m] for m in mask_order],
                                          ordered=True)

plot_results_df['size'] = pd.Categorical(plot_results_df['size'], categories=size_order, ordered=True)
baseline_line_df['size'] = pd.Categorical(baseline_line_df['size'], categories=size_order, ordered=True)

# Create combined faceted plot (log scale)
p2 = (ggplot(plot_results_df, aes(x='tokens_M', y='test_loss', color='mask'))
    + geom_line(size=1)
    + geom_hline(baseline_line_df, aes(yintercept='baseline_loss'),
                 linetype='solid', color='black', size=0.8)
    + facet_wrap('~size', ncol=3, scales='free')
    + scale_x_log10(name='Finetuning Tokens (M)')
    + scale_y_log10(name='Test Loss')
    + scale_color_manual(values=mask_colors_list)
    + base_theme(base_family='Helvetica Neue')
    + theme(figure_size=(7, 3.25),
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

p2.save('plots/adversarial-finetune-curves-log.png', dpi=300, width=7, height=3.25)
p2.save('plots/adversarial-finetune-curves-log.svg', dpi=300, width=7, height=3.25)
p2.save('plots/adversarial-finetune-curves-log.pdf', dpi=300, width=7, height=3.25)
print("Saved plots/adversarial-finetune-curves-log.png")

# =============================================================================
# Plot 3: Finetuning curves for 1816M only
# =============================================================================

size = '1816M'
if size in axis_limits:
    size_data = results_df[results_df['size'] == size].copy()
    size_baseline = baseline_line_df[baseline_line_df['size'] == size].copy()
    limits = axis_limits[size]

    # Filter out steps 0 and 1, filter to x limit
    size_data = size_data[size_data['step'] >= 2]
    size_data = size_data[size_data['tokens_seen'] <= limits['x_max']]

    # Convert tokens to millions for cleaner axis
    size_data['tokens_M'] = size_data['tokens_seen'] / 1e6

    # Apply legend labels to mask column
    size_data['mask'] = size_data['mask'].replace(legend_labels)
    size_data['mask'] = pd.Categorical(size_data['mask'],
                                        categories=[legend_labels[m] for m in mask_order],
                                        ordered=True)

    # Get x range for coord_cartesian (in millions)
    x_min = size_data['tokens_M'].min() if len(size_data) > 0 else 0
    x_max = limits['x_max'] / 1e6

    p3 = (ggplot(size_data, aes(x='tokens_M', y='test_loss', color='mask'))
        + geom_line(size=1)
        + geom_hline(size_baseline, aes(yintercept='baseline_loss'),
                     linetype='solid', color='black', size=0.7)
        + coord_cartesian(xlim=(x_min, x_max), ylim=(limits['y_min'], limits['y_max']))
        + scale_x_log10(name='Finetuning Tokens (M)')
        + scale_y_log10(name='Test Loss')
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

    p3.save('plots/adversarial-finetune-1816M.png', dpi=300, width=3.375, height=2.5)
    p3.save('plots/adversarial-finetune-1816M.svg', dpi=300, width=3.375, height=2.5)
    p3.save('plots/adversarial-finetune-1816M.pdf', dpi=300, width=3.375, height=2.5)
    print("Saved plots/adversarial-finetune-1816M.png")

print("\n" + "="*60)
print("Done!")
print("="*60)

