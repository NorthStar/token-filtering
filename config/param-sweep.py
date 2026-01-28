"""
generate hparam sweeps for scaling laws (125M)
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
from itertools import product

# Add the parent directory to path to import model classes
sys.path.append('..')
from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH

# we will just train a tiny model (though we scale n_layer to match the eventual scales)
# params = {
#     'n_layer' : [2,   4,   7,   10,  14,  20,   26,   32],
#     'n_head' :  [4,   4,   4,    4,   4,   4,    4,    4],
#     'n_embd' :  [256, 256, 256, 256, 256, 256, 256, 256]
# }

params = {
    'n_layer' : [2,   4],
    'n_head' :  [4,   4],
    'n_embd' :  [128, 256]
}

# lr = [5e-4, 1e-3, 5e-3, 1e-2, 3e-3, 5e-2, 1e-1]
lr = [0.001, 0.002, 0.003, 0.004, 0.005]
weight_decay = [1e-3, 1e-2, 1e-1]
optimizers = ['adamw']
tokens_per_param = 20

def get_optimal_tokens(n_layer, n_head, n_embd, tokens_per_param=20, block_size=2048, vocab_size=100256, bias=False, dropout=0.0):
    """Initialize actual model and get exact parameter count"""
    
    # Create model configuration matching template.yaml defaults with proper types
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
    total_tokens = total_params * tokens_per_param

    return int(total_params // 1e6), int(total_tokens)

total_params = []
tokens = []

for n_layer, n_head, n_embd in zip(params['n_layer'], params['n_head'], params['n_embd']):
    p, t = get_optimal_tokens(n_layer, n_head, n_embd, tokens_per_param=tokens_per_param)
    total_params.append(p)
    tokens.append(t)

params['train_tokens'] = tokens

# read template
with open('template.yaml', 'r') as template_file:
    template_config = yaml.safe_load(template_file)

os.makedirs('adamw-sweep', exist_ok=True)
os.makedirs(os.path.join(MODEL_PATH, 'adamw-sweep'), exist_ok=True)

for learning_rate, w_decay in product(lr, weight_decay):

    for i, train_tokens in enumerate(tokens):

        wandb_run_name = f'l-{params["n_layer"][i]}-lr-{learning_rate}-wdecay-{w_decay}'
        
        tokens_per_iter = 5 * 2 * 16 * 2048 # grad_accum x num_gpus x batch_size x block_size
        max_iters = int(train_tokens // tokens_per_iter)
        
        # create config with mask=True
        config = template_config.copy()
        config['n_layer'] = params['n_layer'][i]
        config['n_head'] = params['n_head'][i]
        config['n_embd'] = params['n_embd'][i]
        config['optimizer_type'] = 'adamw'
        config['beta1'] = 0.9
        config['beta2'] = 0.999
        config['eval_test'] = False
        config['weight_decay'] = w_decay
        config['max_iters'] = max_iters
        config['lr_decay_iters'] = max_iters
        config['warmup_iters'] = max_iters // 10
        config['eval_interval'] = max_iters // 25
        config['eval_iters'] = 50
        config['train_tokens'] = f"{tokens[i]:.2e}"
        config['wandb_run_name'] = wandb_run_name
        config['mask'] = False
        config['hidden_learning_rate'] = learning_rate
        config['min_lr'] = learning_rate * 0.1
        config['wandb_project'] = 'medical-adamw-hparam-sweep'
        config['out_dir'] = os.path.join(MODEL_PATH, 'adamw-sweep')
        config['data_path'] = os.path.join(DATA_PATH, 'filtered-224M')

        with open(f'adamw-sweep/gpt-{wandb_run_name}.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)