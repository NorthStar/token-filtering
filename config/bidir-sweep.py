"""
generate sweeps for training probe base models at various scales
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH

# approx 13M, 29M, 62M, 125M, 250M, 500M, 1B, n_embd/n_layer = 64
params = {
    'n_layer' : [2,   4,   7,   10,  14],
    'n_head' :  [4,   4,   8,   10,  14],
    'n_embd' :  [128, 256, 448, 640, 896]
}

def get_optimal_tokens(n_layer, n_head, n_embd, block_size=1024, vocab_size=100256, bias=False, dropout=0.0):

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
    total_tokens = total_params * 20 * 4 # 4x chinchilla

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

hparams = pd.read_csv('hparams.csv')
optimal_hparams = dict()
for i, row in hparams.iterrows():
    optimal_hparams[row['n_layer']] = {
        'lr' : 0.005,
        'w_decay' : 0.01
    }

# read template
with open('template.yaml', 'r') as template_file:
    template_config = yaml.safe_load(template_file)

template_config['wandb_project'] = 'roberta'
template_config['out_dir'] = os.path.join(MODEL_PATH, 'bidir')
template_config['data_path'] = os.path.join(DATA_PATH, 'roberta-pubmed')
template_config['optimizer_type'] = 'muon'

template_config['batch_size'] = 32
template_config['block_size'] = 1024

os.makedirs('bidir', exist_ok=True)

for i, total_params_val in enumerate(total_params):

    wandb_run_name = f'pubmed-{total_params_val}M'
    tokens_per_iter = 5 * 2 * 32 * 1024 # grad_accum x num_gpus x batch_size x block_size
    
    max_iters = int(tokens[i] // tokens_per_iter)    
    config = template_config.copy()

    config['n_layer'] = params['n_layer'][i]
    config['n_head'] = params['n_head'][i]
    config['n_embd'] = params['n_embd'][i]
    config['mup_base_width'] = params['n_embd'][i] # no mup scaling w/ muon
    
    config['train_tokens'] = f"{tokens[i]:.2e}"
    config['max_iters'] = max_iters

    config['lr_decay_iters'] = max_iters
    config['warmup_iters'] = max_iters // 10
    config['hidden_learning_rate'] = 0.005
    config['min_lr'] = 0.0005
    config['weight_decay'] = 0.01

    config['wandb_run_name'] = f'left-{wandb_run_name}'
    config['reverse'] = False

    with open(f'bidir/left-{wandb_run_name}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # create config with reverse=True
    config_right = config.copy()
    config_right['wandb_run_name'] = f'right-{wandb_run_name}'
    config_right['reverse'] = True

    with open(f'bidir/right-{wandb_run_name}.yaml', 'w') as f:
        yaml.dump(config_right, f, default_flow_style=False, indent=2)