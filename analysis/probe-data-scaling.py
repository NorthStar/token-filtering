"""
Data-scaling weak-to-strong generalization experiment for medical token probes.

Similar to probe-weak-to-strong.py, but instead of varying model size, we vary
the amount of training data used for a fixed weak model (29M).

Data splits:
- First 40%: Train probes with varying amounts of data
- Middle 40%: Weak probes generate labels, 224M trains on these
- Last 20%: Held-out evaluation set

This script:
1. Trains 224M probe on first 40% with gold labels (strong baseline)
2. For each data amount (5 log-scaled increments):
   - Trains 29M probe on subset of first 40% with gold labels
   - Uses this 29M probe to generate weak labels on middle 40%
   - Trains 224M probe on weak labels
3. Evaluates all on final 20%:
   - Weak performance: 29M probe with limited data
   - Strong performance: 224M probe trained on gold labels
   - Weak-to-strong: 224M probe trained on weak labels
4. Computes PGR = (w2s - weak) / (strong - weak)
5. Creates visualization showing how weak probe data amount affects w2s generalization
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
try:
    from cuml import LogisticRegression
except:
    print('no cuml')
import tiktoken
import argparse
import pandas as pd
from tqdm import tqdm
from plotnine import *
from colors import THEME_COLORS
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.append('..')
from model import GPT, GPTConfig
from paths import DATA_PATH, MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'bidir'))
parser.add_argument('--data_path', type=str, default=DATA_PATH)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--block_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_iter', type=int, default=10000)
parser.add_argument('--reg_strength', type=float, default=1.0)
parser.add_argument('--probe_types', type=str, default='token,document', help='Comma-separated probe types')
parser.add_argument('--max_doc_samples', type=int, default=None, help='Max documents for validation (None=all)')
parser.add_argument('--n_data_points', type=int, default=5, help='Number of data scaling points')
parser.add_argument('--rerun', action='store_true', help='rerun all experiments')
args = parser.parse_args()

device = args.device
probe_types = args.probe_types.split(',')

def load_gpt(model_file):
    """Load GPT model from checkpoint"""
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
    return model

def load_model(model_name):
    """Load dual GPT models"""
    print(f'Loading {model_name}...')
    left_model = load_gpt(os.path.join(args.model_path, f'left-{model_name}.pt'))
    right_model = load_gpt(os.path.join(args.model_path, f'right-{model_name}.pt'))
    left_model.to(device)
    right_model.to(device)
    left_model.eval()
    right_model.eval()
    
    model_config = left_model.config
    model_config.hidden_size = left_model.config.n_embd * 2
    model_config.num_hidden_layers = left_model.config.n_layer
    
    return {'left': left_model, 'right': right_model}, model_config

def get_gpt_hidden_states(left_model, right_model, x, layer):
    """Get concatenated hidden states from dual GPT models"""
    with torch.no_grad():
        tok_emb_left = left_model.transformer.wte(x)
        h_left = left_model.transformer.drop(tok_emb_left)
        if layer > 0:
            for i in range(layer):
                h_left = left_model.transformer.h[i](h_left)
        
        tok_emb_right = right_model.transformer.wte(x)
        h_right = right_model.transformer.drop(tok_emb_right)
        if layer > 0:
            for i in range(layer):
                h_right = right_model.transformer.h[i](h_right)
        
        h_concat = torch.cat([h_left, h_right], dim=-1)
    return h_concat

enc = tiktoken.get_encoding("cl100k_base")
special_token_ids = {0, enc.eot_token}
eot_token = enc.eot_token

def pad_batch(data, doc_starts, doc_ends, ix, block_size, pad_token_id=0):
    """Pad documents to block_size"""
    padded = []
    attn_mask = []
    for i in ix:
        start = doc_starts[i]
        end = min(doc_ends[i], start + block_size)
        padded.append(torch.from_numpy(np.concatenate([data[start:end], [pad_token_id] * (start + block_size - end)]).astype(np.int64)))
        attn_mask.append(torch.from_numpy(np.concatenate([[1] * (end - start), [0] * (start + block_size - end)]).astype(bool)))
    
    return torch.stack(padded), torch.stack(attn_mask)

def collect_features_and_labels(model, model_config, data, labels, layer, probe_type='token', 
                                doc_starts=None, doc_ends=None, doc_labels=None, 
                                start_pct=0.0, end_pct=1.0, max_tokens=None, max_docs=None):
    """Collect features and labels from model at specified layer
    
    Args:
        probe_type: 'token' or 'document'
        start_pct: Starting percentage of data (0.0 to 1.0)
        end_pct: Ending percentage of data (0.0 to 1.0)
        max_tokens: Maximum number of tokens to collect (for token probes)
        max_docs: Maximum number of documents to collect (for document probes)
    """
    all_features = []
    all_labels = []
    tokens_collected = 0
    docs_collected = 0
    
    if probe_type == 'document':
        # For documents, calculate range based on document indices
        start_doc_idx = int(start_pct * len(doc_starts))
        end_doc_idx = int(end_pct * len(doc_starts))
        
        # Apply max_docs limit if specified
        if max_docs is not None:
            total_docs = end_doc_idx - start_doc_idx
            if total_docs > max_docs:
                end_doc_idx = start_doc_idx + max_docs
                print(f"Limiting to {max_docs} documents (from {total_docs} available)")
        
        n_batches = (end_doc_idx - start_doc_idx) // args.batch_size
        
        print(f"Iterating through {start_pct*100:.0f}%-{end_pct*100:.0f}% of documents ({n_batches} batches, {end_doc_idx - start_doc_idx} docs)")
        
        for batch_idx in tqdm(range(n_batches), desc=f"Layer {layer}"):
            ix = list(range(start_doc_idx + batch_idx * args.batch_size, 
                          min(start_doc_idx + (batch_idx + 1) * args.batch_size, end_doc_idx)))
            
            if len(ix) == 0:
                continue
            
            x, attn_mask = pad_batch(data, doc_starts, doc_ends, ix, args.block_size)
            y = torch.from_numpy(doc_labels[ix].astype(bool))
            
            x = x.to(device)
            y = y.to(device)
            attn_mask = attn_mask.to(device)
            
            # Get hidden states
            features = get_gpt_hidden_states(model['left'], model['right'], x, layer)
            
            # Aggregate features to document-level (mean pooling with attention mask)
            features = features * attn_mask.unsqueeze(-1)
            doc_features = features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1)
            
            all_features.append(doc_features.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            docs_collected += len(ix)
    
    else:  # token-level
        start_idx = int(start_pct * len(data))
        end_idx = int(end_pct * len(data))
        
        if start_idx >= end_idx:
            raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}")
        
        total_length = end_idx - start_idx - args.block_size
        n_batches = total_length // (args.batch_size * args.block_size)
        
        print(f"Iterating through {start_pct*100:.0f}%-{end_pct*100:.0f}% of data ({n_batches} batches)")
        
        for batch_idx in tqdm(range(n_batches), desc=f"Layer {layer}"):
            # Check if we've collected enough tokens
            if max_tokens is not None and tokens_collected >= max_tokens:
                print(f"Reached max_tokens limit: {tokens_collected}/{max_tokens}")
                break
            
            batch_indices = []
            for i in range(args.batch_size):
                idx = start_idx + batch_idx * args.batch_size * args.block_size + i * args.block_size
                if idx + args.block_size <= end_idx:
                    batch_indices.append(idx)
            
            x = torch.stack([torch.from_numpy(data[i:i+args.block_size].astype(np.int64)) for i in batch_indices])
            y = torch.stack([torch.from_numpy(labels[i:i+args.block_size].astype(np.int64)) for i in batch_indices])
        
            x = x.to(device)
            y = y.to(device)
            
            # Get hidden states
            features = get_gpt_hidden_states(model['left'], model['right'], x, layer)
            features = features.view(-1, model_config.hidden_size)
            
            # Extract token-level features
            labels_flat = y.view(-1)
            token_ids = x.view(-1)
            
            # Mask special tokens
            special_token_mask = torch.ones(token_ids.size(0), dtype=torch.bool, device=device)
            for special_id in special_token_ids:
                special_token_mask &= (token_ids != special_id)
            
            if special_token_mask.sum() > 0:
                masked_features = features[special_token_mask]
                masked_labels = labels_flat[special_token_mask]
                all_features.append(masked_features.cpu().numpy())
                all_labels.append(masked_labels.cpu().numpy())
                tokens_collected += masked_features.shape[0]
    
    if all_features:
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        
        # Balance dataset
        medical_mask = all_labels == 1
        non_medical_mask = all_labels == 0
        
        n_medical = medical_mask.sum()
        n_non_medical = non_medical_mask.sum()
        
        min_count = min(n_medical, n_non_medical)
        
        # If we have limits, respect them when balancing
        if probe_type == 'token' and max_tokens is not None:
            # max_tokens is the target total, so each class should have max_tokens/2
            target_per_class = max_tokens // 2
            min_count = min(min_count, target_per_class)
        elif probe_type == 'document' and max_docs is not None:
            # max_docs is the target total, so each class should have max_docs/2
            target_per_class = max_docs // 2
            min_count = min(min_count, target_per_class)
        
        if min_count > 0:
            medical_indices = np.where(medical_mask)[0]
            non_medical_indices = np.where(non_medical_mask)[0]
            
            np.random.seed(42)
            sampled_medical = np.random.choice(medical_indices, min_count, replace=False)
            sampled_non_medical = np.random.choice(non_medical_indices, min_count, replace=False)
            
            balanced_indices = np.concatenate([sampled_medical, sampled_non_medical])
            np.random.shuffle(balanced_indices)
            
            all_features = all_features[balanced_indices]
            all_labels = all_labels[balanced_indices]
        
        print(f"Collected {len(all_features)} balanced samples ({tokens_collected} tokens before balancing, {docs_collected} docs)")
    
    return all_features, all_labels

def train_probe(features, labels, reg_strength=1.0, max_iter=100):
    """Train a logistic regression probe"""
    if reg_strength is None:
        probe = LogisticRegression(penalty=None, max_iter=max_iter)
    else:
        probe = LogisticRegression(C=1/reg_strength, max_iter=max_iter)
    
    print(f"Training probe on {len(features)} samples...")
    probe.fit(features, labels)
    
    return probe

def generate_weak_labels(model, model_config, probe, layer, data, probe_type='token',
                        doc_starts=None, doc_ends=None, start_pct=0.0, end_pct=1.0):
    """Generate weak labels using a trained probe
    
    Returns weak labels for the specified data range
    """
    all_features = []
    
    if probe_type == 'document':
        start_doc_idx = int(start_pct * len(doc_starts))
        end_doc_idx = int(end_pct * len(doc_starts))
        
        # Apply max_doc_samples limit if specified for validation
        if args.max_doc_samples is not None:
            total_docs = end_doc_idx - start_doc_idx
            if total_docs > args.max_doc_samples:
                end_doc_idx = start_doc_idx + args.max_doc_samples
                print(f"Limiting to {args.max_doc_samples} documents (from {total_docs} available)")
        
        n_batches = (end_doc_idx - start_doc_idx) // args.batch_size
        
        print(f"Generating weak labels from {start_pct*100:.0f}%-{end_pct*100:.0f}% of documents ({n_batches} batches, {end_doc_idx - start_doc_idx} docs)...")
        
        for batch_idx in tqdm(range(n_batches)):
            ix = list(range(start_doc_idx + batch_idx * args.batch_size,
                          min(start_doc_idx + (batch_idx + 1) * args.batch_size, end_doc_idx)))
            
            if len(ix) == 0:
                continue
            
            x, attn_mask = pad_batch(data, doc_starts, doc_ends, ix, args.block_size)
            x = x.to(device)
            attn_mask = attn_mask.to(device)
            
            features = get_gpt_hidden_states(model['left'], model['right'], x, layer)
            features = features * attn_mask.unsqueeze(-1)
            doc_features = features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1)
            
            all_features.append(doc_features.cpu().numpy())
    
    else:  # token-level
        start_idx = int(start_pct * len(data))
        end_idx = int(end_pct * len(data))
        
        total_length = end_idx - start_idx - args.block_size
        n_batches = total_length // (args.batch_size * args.block_size)
        
        print(f"Generating weak labels from {start_pct*100:.0f}%-{end_pct*100:.0f}% of data ({n_batches} batches)...")
        for batch_idx in tqdm(range(n_batches)):
            batch_indices = []
            for i in range(args.batch_size):
                idx = start_idx + batch_idx * args.batch_size * args.block_size + i * args.block_size
                if idx + args.block_size <= end_idx:
                    batch_indices.append(idx)
            
            x = torch.stack([torch.from_numpy(data[i:i+args.block_size].astype(np.int64)) for i in batch_indices])
            x = x.to(device)
            
            features = get_gpt_hidden_states(model['left'], model['right'], x, layer)
            features = features.view(-1, model_config.hidden_size)
            
            # Filter special tokens
            token_ids = x.view(-1)
            special_token_mask = torch.ones(token_ids.size(0), dtype=torch.bool, device=device)
            for special_id in special_token_ids:
                special_token_mask &= (token_ids != special_id)
            
            if special_token_mask.sum() > 0:
                masked_features = features[special_token_mask]
                all_features.append(masked_features.cpu().numpy())
    
    all_features = np.vstack(all_features)
    
    # Generate predictions
    weak_labels = probe.predict(all_features)
    
    return weak_labels

# Load data for both token and document levels
if args.rerun:
    print("Loading data...")
    data_files = {}
    for ptype in probe_types:
        data_filename = os.path.join(args.data_path, f'probe/{ptype}/tokens.bin')
        labels_filename = os.path.join(args.data_path, f'probe/{ptype}/labels.bin')

        global_data = np.memmap(data_filename, dtype=np.uint32, mode='r')
        global_labels = np.memmap(labels_filename, dtype=bool, mode='r')

        data_dict = {
            'data': global_data,
            'labels': global_labels
        }

        # For document-level, also load document boundaries
        if ptype == 'document':
            doc_starts = np.concatenate([[0], np.where(global_data == eot_token)[0][:-1] + 1])
            doc_ends = np.where(global_data == eot_token)[0]
            doc_labels = global_labels[doc_starts]

            data_dict['doc_starts'] = doc_starts
            data_dict['doc_ends'] = doc_ends
            data_dict['doc_labels'] = doc_labels

            print(f"Loaded {len(global_data)} tokens, {len(doc_starts)} documents for {ptype}")
        else:
            print(f"Loaded {len(global_data)} tokens for {ptype}")

        data_files[ptype] = data_dict

    # Create output directory
    probe_dir = os.path.join(args.model_path.replace('bidir', ''), 'data-scaling-probes')
    os.makedirs(probe_dir, exist_ok=True)

results_file = 'results/data-scaling-probe-results.csv'

if not os.path.exists(results_file) or args.rerun:
    
    all_results = []
    
    # Fixed weak model
    weak_model_name = 'pubmed-29M'
    strong_model_name = 'pubmed-224M'
    
    # Loop over probe types
    for probe_type in probe_types:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT FOR PROBE TYPE: {probe_type.upper()}")
        print(f"{'='*80}\n")
        
        # Get data for this probe type
        data_dict = data_files[probe_type]
        global_data = data_dict['data']
        global_labels = data_dict['labels']
        doc_starts = data_dict.get('doc_starts', None)
        doc_ends = data_dict.get('doc_ends', None)
        doc_labels = data_dict.get('doc_labels', None)
        
        # Calculate data amounts to sweep over
        if probe_type == 'document':
            # For documents, sweep over number of documents
            total_docs = int(0.4 * len(doc_starts))  # First 40%
            # Log-scale from ~1% to 100% of available docs
            min_docs = max(10, int(total_docs * 0.01))
            data_amounts = np.logspace(np.log10(min_docs), np.log10(total_docs), args.n_data_points).astype(int)
            print(f"Data amounts (documents): {data_amounts}")
        else:
            # For tokens, we need to estimate based on balanced dataset size
            # After balancing, we get roughly min(n_medical, n_non_medical) * 2 samples
            # Let's aim for log-scaled token counts
            # A rough estimate: with 15 batches at full 40%, we get some number of tokens
            # Let's use data amounts as "number of batches to use" and convert to approximate tokens
            total_tokens_estimate = int(0.4 * len(global_data))
            min_tokens = max(1000, int(total_tokens_estimate * 0.001))  # At least 1k tokens
            max_tokens = int(total_tokens_estimate * 0.1)  # Use up to 10% of the total
            data_amounts = np.logspace(np.log10(min_tokens), np.log10(max_tokens), args.n_data_points).astype(int)
            print(f"Data amounts (tokens target): {data_amounts}")
        
        # Step 1: Train strong probe (224M) on first 40% with gold labels
        print(f"\n=== Step 1: Training strong {probe_type} probe (224M) on first 40% ===")
        
        model_224M, config_224M = load_model(strong_model_name)
        
        # Load layer info from existing probe
        orig_probe_path = os.path.join(args.model_path.replace('bidir', ''), f'probes/{strong_model_name}-token.pkl')
        with open(orig_probe_path, 'rb') as f:
            orig_probe_data = pickle.load(f)
            layer_224M = orig_probe_data['layer']
        
        print(f"Using layer {layer_224M} for 224M")
        
        # Collect features from first 40%
        strong_train_features, strong_train_labels = collect_features_and_labels(
            model_224M, config_224M, global_data, global_labels, layer_224M,
            probe_type=probe_type,
            doc_starts=doc_starts, doc_ends=doc_ends, doc_labels=doc_labels,
            start_pct=0.0, end_pct=0.4
        )
        
        # Train strong probe
        strong_probe = train_probe(strong_train_features, strong_train_labels, args.reg_strength, args.max_iter)
        
        # Save strong probe
        output = {
            'layer': layer_224M,
            'probe': strong_probe,
            'classifier_type': 'linear',
            'probe_type': probe_type,
        }
        probe_file = os.path.join(probe_dir, f'{strong_model_name}-gold-{probe_type}.pkl')
        with open(probe_file, 'wb') as f:
            pickle.dump(output, f)
        print(f"Saved strong probe to {probe_file}")
        
        # Load weak model (29M)
        model_29M, config_29M = load_model(weak_model_name)
        
        # Load layer info
        orig_probe_path = os.path.join(args.model_path.replace('bidir', ''), f'probes/{weak_model_name}-token.pkl')
        with open(orig_probe_path, 'rb') as f:
            orig_probe_data = pickle.load(f)
            layer_29M = orig_probe_data['layer']
        
        print(f"Using layer {layer_29M} for 29M")
        
        # Step 2: For each data amount, train weak probe and generate labels
        print(f"\n=== Step 2: Training weak {probe_type} probes with varying data amounts ===")
        
        weak_probes = {}
        weak_labels_data = {}
        
        for data_amount in data_amounts:
            print(f"\n--- Training with data amount: {data_amount} ---")
            
            # Collect features from subset of first 40%
            if probe_type == 'document':
                weak_train_features, weak_train_labels = collect_features_and_labels(
                    model_29M, config_29M, global_data, global_labels, layer_29M,
                    probe_type=probe_type,
                    doc_starts=doc_starts, doc_ends=doc_ends, doc_labels=doc_labels,
                    start_pct=0.0, end_pct=0.4,
                    max_docs=data_amount
                )
            else:  # token
                weak_train_features, weak_train_labels = collect_features_and_labels(
                    model_29M, config_29M, global_data, global_labels, layer_29M,
                    probe_type=probe_type,
                    doc_starts=doc_starts, doc_ends=doc_ends, doc_labels=doc_labels,
                    start_pct=0.0, end_pct=0.4,
                    max_tokens=data_amount
                )
            
            # Train weak probe
            weak_probe = train_probe(weak_train_features, weak_train_labels, args.reg_strength, args.max_iter)
            weak_probes[data_amount] = weak_probe
            
            # Save weak probe
            output = {
                'layer': layer_29M,
                'probe': weak_probe,
                'classifier_type': 'linear',
                'probe_type': probe_type,
                'data_amount': data_amount,
            }
            probe_file = os.path.join(probe_dir, f'{weak_model_name}-data{data_amount}-{probe_type}.pkl')
            with open(probe_file, 'wb') as f:
                pickle.dump(output, f)
            
            # Generate weak labels on middle 40% (40%-80%)
            print(f"Generating weak {probe_type} labels on middle 40%...")
            weak_labels = generate_weak_labels(
                model_29M, config_29M, weak_probe, layer_29M, global_data,
                probe_type=probe_type,
                doc_starts=doc_starts, doc_ends=doc_ends,
                start_pct=0.4, end_pct=0.8
            )
            weak_labels_data[data_amount] = weak_labels
        
        # Step 3: Train 224M probes on weak labels from middle 40%
        print(f"\n=== Step 3: Training 224M {probe_type} probes on weak labels ===")
        
        probes_weak_to_strong = {}
        
        for data_amount in data_amounts:
            print(f"\n--- Training 224M on weak labels from data_amount={data_amount} ---")
            
            weak_labels = weak_labels_data[data_amount]
            
            # Collect 224M features from middle 40% (matching weak labels)
            print(f"Collecting 224M {probe_type} features from 40%-80%...")
            
            features_224M_list = []
            
            if probe_type == 'document':
                start_doc_idx = int(0.4 * len(doc_starts))
                end_doc_idx = int(0.8 * len(doc_starts))
                
                # Apply max_doc_samples limit if specified
                if args.max_doc_samples is not None:
                    total_docs = end_doc_idx - start_doc_idx
                    if total_docs > args.max_doc_samples:
                        end_doc_idx = start_doc_idx + args.max_doc_samples
                
                n_batches = (end_doc_idx - start_doc_idx) // args.batch_size
                
                for batch_idx in tqdm(range(n_batches)):
                    ix = list(range(start_doc_idx + batch_idx * args.batch_size,
                                  min(start_doc_idx + (batch_idx + 1) * args.batch_size, end_doc_idx)))
                    if len(ix) == 0:
                        continue
                    
                    x, attn_mask = pad_batch(global_data, doc_starts, doc_ends, ix, args.block_size)
                    x = x.to(device)
                    attn_mask = attn_mask.to(device)
                    
                    features = get_gpt_hidden_states(model_224M['left'], model_224M['right'], x, layer_224M)
                    features = features * attn_mask.unsqueeze(-1)
                    doc_features = features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1)
                    features_224M_list.append(doc_features.cpu().numpy())
            else:  # token
                start_idx = int(0.4 * len(global_data))
                end_idx = int(0.8 * len(global_data))
                total_length = end_idx - start_idx - args.block_size
                n_batches = total_length // (args.batch_size * args.block_size)
                
                for batch_idx in tqdm(range(n_batches)):
                    batch_indices = []
                    for i in range(args.batch_size):
                        idx = start_idx + batch_idx * args.batch_size * args.block_size + i * args.block_size
                        if idx + args.block_size <= end_idx:
                            batch_indices.append(idx)
                    
                    x = torch.stack([torch.from_numpy(global_data[i:i+args.block_size].astype(np.int64)) for i in batch_indices])
                    x = x.to(device)
                    
                    features = get_gpt_hidden_states(model_224M['left'], model_224M['right'], x, layer_224M)
                    features = features.view(-1, config_224M.hidden_size)
                    
                    # Filter special tokens
                    token_ids = x.view(-1)
                    special_token_mask = torch.ones(token_ids.size(0), dtype=torch.bool, device=device)
                    for special_id in special_token_ids:
                        special_token_mask &= (token_ids != special_id)
                    
                    if special_token_mask.sum() > 0:
                        masked_features = features[special_token_mask]
                        features_224M_list.append(masked_features.cpu().numpy())
            
            features_224M = np.vstack(features_224M_list)
            
            # Verify matching shapes
            print(f"224M features shape: {features_224M.shape}, weak labels shape: {weak_labels.shape}")
            
            # Train 224M probe on weak labels
            probe_w2s = train_probe(features_224M, weak_labels, args.reg_strength, args.max_iter)
            probes_weak_to_strong[data_amount] = probe_w2s
            
            # Save probe
            output = {
                'layer': layer_224M,
                'probe': probe_w2s,
                'classifier_type': 'linear',
                'weak_data_amount': data_amount,
                'probe_type': probe_type,
            }
            probe_file = os.path.join(probe_dir, f'{strong_model_name}-from-29M-data{data_amount}-{probe_type}.pkl')
            with open(probe_file, 'wb') as f:
                pickle.dump(output, f)
        
        # Step 4: Evaluate all probes on held-out validation set (last 20%)
        print(f"\n=== Step 4: Evaluating {probe_type} probes on validation set (80%-100%) ===")
        
        # Evaluate strong probe
        strong_features_val, strong_labels_val = collect_features_and_labels(
            model_224M, config_224M, global_data, global_labels, layer_224M,
            probe_type=probe_type,
            doc_starts=doc_starts, doc_ends=doc_ends, doc_labels=doc_labels,
            start_pct=0.8, end_pct=1.0
        )
        strong_predictions = strong_probe.predict(strong_features_val)
        strong_acc = accuracy_score(strong_labels_val, strong_predictions)
        
        print(f"Strong {probe_type} probe (224M gold) accuracy: {strong_acc:.4f}")
        
        for data_amount in data_amounts:
            print(f"\nEvaluating for data_amount={data_amount}...")
            
            # Evaluate weak probe
            weak_probe = weak_probes[data_amount]
            
            weak_features_val, weak_labels_val = collect_features_and_labels(
                model_29M, config_29M, global_data, global_labels, layer_29M,
                probe_type=probe_type,
                doc_starts=doc_starts, doc_ends=doc_ends, doc_labels=doc_labels,
                start_pct=0.8, end_pct=1.0
            )
            weak_predictions = weak_probe.predict(weak_features_val)
            weak_acc = accuracy_score(weak_labels_val, weak_predictions)
            
            print(f"Weak {probe_type} probe (29M, data={data_amount}) accuracy: {weak_acc:.4f}")
            
            # Evaluate weak-to-strong probe
            w2s_probe = probes_weak_to_strong[data_amount]
            w2s_predictions = w2s_probe.predict(strong_features_val)
            w2s_acc = accuracy_score(strong_labels_val, w2s_predictions)
            
            print(f"Weak-to-strong {probe_type} probe accuracy: {w2s_acc:.4f}")
            
            # Compute PGR
            if strong_acc > weak_acc:
                pgr = (w2s_acc - weak_acc) / (strong_acc - weak_acc)
            else:
                pgr = 0.0
            
            print(f"Performance Gap Recovered: {pgr:.4f}")
            
            all_results.append({
                'probe_type': probe_type,
                'data_amount': data_amount,
                'weak_acc': weak_acc,
                'strong_acc': strong_acc,
                'weak_to_strong_acc': w2s_acc,
                'pgr': pgr
            })
    
    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs('results', exist_ok=True)
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

else:
    print(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)

# Create plot: accuracy vs weak data amount
print("\nCreating plot...")
print(f"Loaded data:\n{df}")

# Theme configuration
bg_color = THEME_COLORS['bg_color']
text_color = THEME_COLORS['text_color']
line_color = THEME_COLORS['line_color']
grid_color = THEME_COLORS['grid_color']

# Reshape data to long format for plotting (only Weak and Weak-to-Strong as lines)
df_long = []
for _, row in df.iterrows():
    df_long.append({
        'probe_type': row['probe_type'],
        'data_amount': row['data_amount'] / 1000,  # Convert to thousands
        'metric': 'Weak',
        'accuracy': row['weak_acc'] * 100  # Convert to %
    })
    df_long.append({
        'probe_type': row['probe_type'],
        'data_amount': row['data_amount'] / 1000,  # Convert to thousands
        'metric': 'Weak-to-Strong',
        'accuracy': row['weak_to_strong_acc'] * 100  # Convert to %
    })

df_plot = pd.DataFrame(df_long)

# Metric order and colors
metric_order = ['Weak', 'Weak-to-Strong']
df_plot['metric'] = pd.Categorical(df_plot['metric'], categories=metric_order, ordered=True)

# Colors for metrics
metric_colors = ['#DAD7CD', '#273C2C']  # Weak, Weak-to-Strong

# Probe type labels
probe_labels = {'token': 'Token', 'document': 'Document'}
df_plot['probe_label'] = df_plot['probe_type'].map(probe_labels)

# Get strong accuracy for hlines (one per probe type)
strong_acc_data = df.groupby('probe_type')['strong_acc'].first().reset_index()
strong_acc_data['strong_acc'] = strong_acc_data['strong_acc'] * 100  # Convert to %
strong_acc_data['probe_label'] = strong_acc_data['probe_type'].map(probe_labels)

# Create facet label annotations (inside top-left of each facet)
min_x_token = df_plot[df_plot['probe_type'] == 'token']['data_amount'].min()
min_x_doc = df_plot[df_plot['probe_type'] == 'document']['data_amount'].min()
facet_labels = pd.DataFrame({
    'probe_label': ['Token', 'Document'],
    'label': ['Token', 'Document'],
    'data_amount': [min_x_token, min_x_doc],
    'accuracy': [89.5, 89.5]  # Position near top with padding
})

os.makedirs('plots', exist_ok=True)

p = (ggplot(df_plot, aes(x='data_amount', y='accuracy', color='metric'))
     + geom_hline(data=strong_acc_data, mapping=aes(yintercept='strong_acc'),
                  color='black', linetype='solid', size=0.8)
     + geom_line(size=1)
     + geom_point(size=2, stroke=0, alpha=0.9)
     + geom_point(fill="none", stroke=0.7, size=2, color="#4f4f4f")
     + geom_text(data=facet_labels, mapping=aes(x='data_amount', y='accuracy', label='label'),
                 ha='left', va='top', size=8, color=text_color, fontweight='bold', inherit_aes=False)
     + facet_wrap('~probe_label', ncol=2, scales='free_x')
     + scale_color_manual(values=metric_colors)
     + scale_x_log10(name='Weak Probe Training Tokens (Thousands)')
     + scale_y_continuous(name='Accuracy (%)', limits=[70, 90])
     + guides(color=guide_legend(nrow=1))
     + theme_bw(base_family='Helvetica Neue')
     + theme(figure_size=(3.375, 2.5),
             strip_text=element_blank(),
             strip_background=element_blank(),
             panel_grid_major=element_line(size=0.3, color=grid_color),
             panel_grid_minor=element_blank(),
             legend_title=element_blank(),
             legend_position='top',
             legend_direction='horizontal',
             axis_title_x=element_text(size=9, color=text_color),
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

p.save('plots/data-scaling-probe-results.png', dpi=300, width=3.375, height=2.5)
p.save('plots/data-scaling-probe-results.svg', dpi=300, width=3.375, height=2.5)
p.save('plots/data-scaling-probe-results.pdf', dpi=300, width=3.375, height=2.5)
print(p)
print("Saved plots/data-scaling-probe-results.{png,svg,pdf}")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(df.to_string(index=False))

