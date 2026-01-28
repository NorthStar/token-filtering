"""
sweep across various ground truth labels (sentence, document)
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
import pickle
import tiktoken
from sklearn.metrics import precision_recall_curve

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH

# approx 13M, 29M, 62M, 125M, 250M, 500M, n_embd/n_layer = 64
params = {
    'n_layer' : [2,   4,   7,   10,  14,  20],
    'n_head' :  [4,   4,   8,   10,  14,  10],
    'n_embd' :  [128, 256, 448, 640, 896, 1280]
}

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 1024
batch_size = 16
n_batches = 3200

def load_gpt(model_file):
    """Load GPT model from checkpoint"""
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

def load_model(model_name='pubmed-61M'):
    """Load the bidirectional model"""
    model_path = os.path.join(MODEL_PATH, 'bidir')
    print(f'Loading left-{model_name}.pt and right-{model_name}.pt')
    left_model = load_gpt(os.path.join(model_path, f'left-{model_name}.pt'))
    right_model = load_gpt(os.path.join(model_path, f'right-{model_name}.pt'))
    left_model.to(device)
    right_model.to(device)
    left_model.eval()
    right_model.eval()
    
    # create combined config
    model_config = left_model.config
    model_config.hidden_size = left_model.config.n_embd * 2  # Concatenated features
    
    return {'left': left_model, 'right': right_model}, model_config

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

def extract_features(model, tokens, layer):
    """Extract features from the model at specified layer"""
    x = torch.from_numpy(tokens.astype(np.int64)).to(device)
    features = get_gpt_hidden_states(model['left'], model['right'], x, layer)
    return features.cpu().numpy()

enc = tiktoken.get_encoding("cl100k_base")
special_token_ids = {0, enc.eot_token}

def compute_optimal_threshold(probe_path, dataset='test_tokens'):
    """Compute optimal threshold for a probe on a dataset"""
    # Load probe
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    probe = probe_data['probe']
    layer = probe_data['layer']
    
    # Load model
    model, model_config = load_model('pubmed-61M')
    hidden_size = model_config.hidden_size
    
    # Load data
    tokens_path = os.path.join(DATA_PATH, f'probe/token/{dataset}.bin')
    labels_path = os.path.join(DATA_PATH, f'probe/token/{dataset.replace("tokens", "labels")}.bin')
    
    tokens = np.memmap(tokens_path, dtype=np.uint32, mode='r')
    labels = np.memmap(labels_path, dtype=bool, mode='r')
    
    all_features = []
    all_labels = []
    
    chunk_size = batch_size * block_size
    n_chunks = min(n_batches // batch_size, (len(tokens) - block_size) // chunk_size)
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(tokens) - block_size)
        
        if start_idx >= end_idx:
            break
        
        # Extract chunk
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        
        # Reshape into batches
        n_seqs = len(chunk_tokens) // block_size
        if n_seqs == 0:
            continue
        
        chunk_tokens = chunk_tokens[:n_seqs * block_size].reshape(n_seqs, block_size)
        chunk_labels = chunk_labels[:n_seqs * block_size].reshape(n_seqs, block_size)
        
        # Extract features
        features = extract_features(model, chunk_tokens, layer)
        features = features.reshape(-1, hidden_size)
        chunk_labels = chunk_labels.reshape(-1)
        
        # Filter out special tokens
        token_ids = chunk_tokens.reshape(-1)
        mask = np.ones(len(token_ids), dtype=bool)
        for special_id in special_token_ids:
            mask &= (token_ids != special_id)
        
        if mask.sum() > 0:
            all_features.append(features[mask])
            all_labels.append(chunk_labels[mask])
    
    # Combine all features and labels
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    # Balance labels
    min_count = min(labels.sum(), (~labels).sum())
    print(f"Balancing labels to {min_count} medical and {min_count} non-medical")
    
    medical_indices = np.where(labels)[0]
    non_medical_indices = np.where(~labels)[0]
    
    np.random.seed(42)
    sampled_medical = np.random.choice(medical_indices, min_count, replace=False)
    sampled_non_medical = np.random.choice(non_medical_indices, min_count, replace=False)
    
    balanced_indices = np.concatenate([sampled_medical, sampled_non_medical])
    np.random.shuffle(balanced_indices)
    
    features = features[balanced_indices]
    labels = labels[balanced_indices]
    
    # Make predictions and find optimal threshold
    probabilities = probe.predict_proba(features)[:, 1]
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_idx = np.argmax(f1_scores[:-1])
    
    threshold = thresholds[best_idx]
    print(f"Optimal threshold: {threshold}")
    
    return float(threshold)

# Compute thresholds for both probes
print("Computing threshold for sentence probe...")
sentence_probe_path = os.path.join(MODEL_PATH, 'probes/pubmed-61M-token-sentence.pkl')
sentence_threshold = compute_optimal_threshold(sentence_probe_path)

print("\nComputing threshold for document probe...")
document_probe_path = os.path.join(MODEL_PATH, 'probes/pubmed-61M-token-document.pkl')
document_threshold = compute_optimal_threshold(document_probe_path)

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

template_config['wandb_project'] = 'medical-label-sweep'
template_config['out_dir'] = os.path.join(MODEL_PATH, 'label-sweep')
template_config['optimizer_type'] = 'adamw'
template_config['beta1'] = 0.9
template_config['beta2'] = 0.95
template_config['begin_filter_step'] = 0

os.makedirs('label-sweep', exist_ok=True)
os.makedirs(os.path.join(MODEL_PATH, 'label-sweep'), exist_ok=True)

# Generate configs for both probes
probe_configs = [
    {'name': 'sentence', 'threshold': sentence_threshold, 'data_path': os.path.join(DATA_PATH, 'filtered-61M-token-sentence')},
    {'name': 'document', 'threshold': document_threshold, 'data_path': os.path.join(DATA_PATH, 'filtered-61M-token-document')}
]

for probe_config in probe_configs:
    probe_name = probe_config['name']
    threshold = probe_config['threshold']
    data_path = probe_config['data_path']
    
    print(f"\nGenerating configs for {probe_name} probe (threshold={threshold:.4f})")
    
    template = template_config.copy()
    template['data_path'] = data_path
    template['mask_threshold'] = threshold
    
    for i, total_params_val in enumerate(total_params):
        
        wandb_run_name = f'{total_params_val}M'
        
        if total_params_val > 1000:
            tokens_per_iter = 5 * 8 * 4 * 2048 # grad_accum x num_gpus x batch_size x block_size
            grad_clip = 0.5
        else:
            tokens_per_iter = 5 * 2 * 16 * 2048 # grad_accum x num_gpus x batch_size x block_size
            grad_clip = 1.0
        
        max_iters = int(tokens[i] // tokens_per_iter)
        template['grad_clip'] = grad_clip
        
        # create config with mask=True
        config = template.copy()
        
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
        
        config['wandb_run_name'] = f'{probe_name}-{wandb_run_name}'
        config['mask'] = True
        
        with open(f'label-sweep/gpt-{wandb_run_name}-{probe_name}.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

print(f"\nGenerated {len(total_params) * len(probe_configs)} config files in label-sweep/")