"""
generate sweeps for various chinchilla scales
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

# approx 62M, 125M, 250M, 500M, 1B, n_embd/n_layer = 64
params = {
    'n_layer' : [2,   4,   7,   10,  14,  20],
    'n_head' :  [4,   4,   8,   10,  14,  10],
    'n_embd' :  [128, 256, 448, 640, 896, 1280]
}

begin_filter_steps = [0, 0.2, 0.4, 0.6, 0.8]

thresholds = pd.read_csv('probe-thresholds.csv')
thresholds = thresholds[thresholds['model'] == 'pubmed-224M']
thresholds = thresholds[thresholds['dataset'] == 'test_tokens']
threshold  = list(thresholds['threshold'])[0]
print(list(thresholds['threshold'])[0])
print(threshold)

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
# read template
with open('template.yaml', 'r') as template_file:
    template_config = yaml.safe_load(template_file)

template_config['wandb_project'] = 'delayed-filter'
template_config['out_dir'] = os.path.join(MODEL_PATH, 'delayed-filter')
template_config['data_path'] = os.path.join(DATA_PATH, 'filtered-224M')
template_config['optimizer_type'] = 'adamw'
template_config['mask_threshold'] = threshold
template_config['beta1'] = 0.9
template_config['beta2'] = 0.95

os.makedirs('delayed-filter', exist_ok=True)

for i, total_params_val in enumerate(total_params):

    wandb_run_name = f'{total_params_val}M'
    tokens_per_iter = 5 * 2 * 16 * 2048 # grad_accum x num_gpus x batch_size x block_size
    max_iters = int(tokens[i] // tokens_per_iter)
    
    # create config with mask=True
    config = template_config.copy()
    
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

    config['wandb_run_name'] = f'mask-{wandb_run_name}'
    config['mask'] = True

    for begin_filter_step in begin_filter_steps:
        config_begin_filter_step = config.copy()
        config_begin_filter_step['wandb_run_name'] = f'mask-{wandb_run_name}-{int(begin_filter_step * 100)}'
        begin_filter_idx = int(round(begin_filter_step * max_iters))
        config_begin_filter_step['begin_filter_step'] = begin_filter_idx
        
        with open(f'delayed-filter/gpt-{wandb_run_name}-{int(begin_filter_step * 100)}.yaml', 'w') as f:
            yaml.dump(config_begin_filter_step, f, default_flow_style=False, indent=2)