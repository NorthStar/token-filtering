"""
generate sweeps for various roberta hyperparameters
"""
import os
import sys
import yaml
from itertools import product

params = {
    'lr' : [5e-5, 1e-4, 5e-4, 1e-3],
    'beta2' : [0.95, 0.98, 0.999],
    'w_decay' : [1e-3, 1e-2, 0.1],
}

# read template
with open('roberta.yaml', 'r') as template_file:
    template_config = yaml.safe_load(template_file)

os.makedirs('roberta', exist_ok=True)

for lr, beta2, w_decay in product(params['lr'], params['beta2'], params['w_decay']):

    wandb_run_name = f'lr-{lr}-b2-{beta2}-lambda-{w_decay}'    
    max_iters = 1000
    
    config = template_config.copy()
    config['learning_rate'] = lr
    config['beta2'] = beta2
    config['weight_decay'] = w_decay
    config['max_iters'] = max_iters
    config['wandb_run_name'] = wandb_run_name
    
    with open(f'roberta/{wandb_run_name}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)