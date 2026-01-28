"""
generate sweeps for various chinchilla scales with fixed 25% filtering rate
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch

sys.path.append('..')
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH

probe_sizes = ['roberta-edu', 'edu-61M', 'ModernBERT-large', '29M', '61M', '113M', '224M']
thresholds_df = pd.read_csv('accuracy-sweep-thresholds.csv')

thresholds = dict()
for model, threshold in zip(thresholds_df['model'], thresholds_df['threshold']):
    thresholds[model] = threshold

print("Loaded thresholds:")
print(thresholds)

# approx 13M, 29M, 61M, 113M, 224M, 512M
params = {
    'n_layer' : [2,   4,   7,   10,  14,  20],
    'n_head' :  [4,   4,   8,   10,  14,  10],
    'n_embd' :  [128, 256, 448, 640, 896, 1280]
}

def get_optimal_tokens(n_layer, n_head, n_embd, block_size=2048, vocab_size=100256, bias=False, dropout=0.0):

    model_args = dict(
        n_layer=int(n_layer), 
        n_head=int(n_head), 
        n_embd=int(n_embd), 
        block_size=int(block_size),
        vocab_size=int(vocab_size),
        bias=bool(bias),
        dropout=float(dropout)
    )
    
    # initialize model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # get exact parameter count
    total_params = model.get_num_params()
    total_tokens = total_params * 20

    return int(total_params // 1e6), int(total_tokens)

total_params = []
tokens = []
for n_layer, n_head, n_embd in zip(params['n_layer'], params['n_head'], params['n_embd']):
    p, t = get_optimal_tokens(n_layer, n_head, n_embd)
    print(f'params: {p}M | ratio: {round(n_embd / n_layer, 2)}')
    total_params.append(p)
    tokens.append(t)

params['total_params'] = total_params
params['train_tokens'] = tokens

hparams = pd.read_csv('adamw-hparams.csv')
optimal_hparams = dict()
for i, row in hparams.iterrows():
    optimal_hparams[row['n_layer']] = {
        'lr' : row['best_lr'],
        'w_decay' : row['best_w_decay']
    }

# read template
with open('template.yaml', 'r') as template_file:
    template_config = yaml.safe_load(template_file)

template_config['wandb_project'] = 'medical-fixed-accuracy'
template_config['out_dir'] = os.path.join(MODEL_PATH, 'fixed-accuracy')
template_config['optimizer_type'] = 'adamw'
template_config['beta1'] = 0.9
template_config['beta2'] = 0.95
template_config['begin_filter_step'] = 0

os.makedirs('fixed-accuracy', exist_ok=True)
os.makedirs(os.path.join(MODEL_PATH, 'fixed-accuracy'), exist_ok=True)

for i, total_params_val in enumerate(total_params):

    wandb_run_name = f'{total_params_val}M'
    
    if total_params_val > 1000:
        tokens_per_iter = 5 * 8 * 4 * 2048 # grad_accum x num_gpus x batch_size x block_size
        grad_clip = 0.5
    else:
        tokens_per_iter = 5 * 2 * 16 * 2048 # grad_accum x num_gpus x batch_size x block_size
        grad_clip = 1.0
    
    max_iters = int(tokens[i] // tokens_per_iter)
    template_config['grad_clip'] = grad_clip
    
    # create config with mask=True
    config = template_config.copy()

    if total_params_val > 1000:
        config['batch_size'] = 4
        config['gradient_accumulation_steps'] = 5 * 8
    
    config['n_layer'] = params['n_layer'][i]
    config['n_head'] = params['n_head'][i]
    config['n_embd'] = params['n_embd'][i]
    config['mup_base_width'] = 256

    config['train_tokens'] = f"{tokens[i]:.2e}"
    config['max_iters'] = max_iters

    config['lr_decay_iters'] = max_iters
    config['warmup_iters'] = max_iters // 10
    config['hidden_learning_rate'] = optimal_hparams[params['n_layer'][i]]['lr']
    config['min_lr'] = config['hidden_learning_rate'] * 0.1
    config['weight_decay'] = optimal_hparams[params['n_layer'][i]]['w_decay']
    
    config['mask'] = True

    for probe_size in probe_sizes:

        config_threshold = config.copy()
        if probe_size == 'roberta-edu':
            config_threshold['data_path'] = os.path.join(DATA_PATH, 'filtered-roberta')
        elif probe_size == 'ModernBERT-large':
            config_threshold['data_path'] = os.path.join(DATA_PATH, 'filtered-bert')
        elif probe_size == 'edu-61M':
            config_threshold['data_path'] = os.path.join(DATA_PATH, 'filtered-edu')
        else:
            config_threshold['data_path'] = os.path.join(DATA_PATH, f'filtered-{probe_size}')
        
        config_threshold['wandb_run_name'] = f'mask-{wandb_run_name}-{probe_size}'
        config_threshold['mask_threshold'] = thresholds[probe_size]
        
        with open(f'fixed-accuracy/gpt-{wandb_run_name}-{probe_size}.yaml', 'w') as f:
            yaml.dump(config_threshold, f, default_flow_style=False, indent=2)

print(f"\nGenerated {len(total_params) * len(probe_sizes)} config files in fixed-accuracy/")

