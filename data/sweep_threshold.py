"""
Evaluate a pretrained probe on test data with threshold sweeping
   - takes a pretrained probe as an argument
   - evaluates it on test_tokens.bin and test_labels.bin test sets
   - sweeps out the classifier threshold
   - saves results to a csv
   - plots histogram of predicted probabilities
"""

import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import normalize

import tiktoken
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
from omegaconf import OmegaConf

import sys
sys.path.append('..')
from model import GPT, GPTConfig
from paths import DATA_PATH, MODEL_PATH

from plotnine import *

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pretrained probe with threshold sweeping')
    parser.add_argument('--probe_path', type=str, default=os.path.join(MODEL_PATH, 'probes', 'pubmed-113M-token.pkl'), help='Path to the pretrained probe pickle file')
    parser.add_argument('--cfg', type=str, default='probe.yaml', help='Config file path')
    parser.add_argument('--min_threshold', type=float, default=0.05, help='Minimum threshold value')
    parser.add_argument('--max_threshold', type=float, default=0.95, help='Maximum threshold value')
    parser.add_argument('--threshold_step', type=float, default=0.05, help='Threshold step size')
    return parser.parse_args()

def detect_probe_type(probe_path):
    """Detect probe type from filename suffix"""
    if probe_path.endswith('-doc.pkl'):
        return 'document'
    elif probe_path.endswith('-sent.pkl'):
        return 'sentence'
    elif probe_path.endswith('-token.pkl'):
        return 'token'
    else:
        # Default fallback - try to infer from path
        if 'doc' in probe_path.lower():
            return 'document'
        elif 'sent' in probe_path.lower():
            return 'sentence'
        else:
            return 'token'

def load_config(cfg_file):
    """Load configuration from yaml file and command line"""
    cfg = OmegaConf.load(cfg_file)
    cfg.update(OmegaConf.from_cli())
    
    # Convert to dict for easier access
    cfg_dict = OmegaConf.to_container(cfg)
    
    return cfg_dict

def load_gpt(model_file, device):
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

def get_gpt_hidden_states(left_model, right_model, x, layer):
    """
    Get hidden states from both GPT models at specified layer and concatenate them.
    Based on the approach from layers.py
    """
    with torch.no_grad():
        # Get hidden states from left model
        tok_emb_left = left_model.transformer.wte(x)
        h_left = left_model.transformer.drop(tok_emb_left)
        
        if layer > 0:
            for i in range(layer):
                h_left = left_model.transformer.h[i](h_left)
        
        # Get hidden states from right model
        tok_emb_right = right_model.transformer.wte(x)
        h_right = right_model.transformer.drop(tok_emb_right)
        
        if layer > 0:
            for i in range(layer):
                h_right = right_model.transformer.h[i](h_right)
        
        # Concatenate hidden states along the feature dimension
        h_concat = torch.cat([h_left, h_right], dim=-1)
        
    return h_concat

def prep_tokens(token_ids, pad_token_id=0):
    """Prepare tokens for model input"""
    return {'input_ids': token_ids, 'attention_mask': (token_ids != pad_token_id).long()}

def pad_batch(data, ix, doc_starts, doc_ends, block_size, pad_token_id=0):
    """Pad documents to block_size for batch processing"""
    padded = []
    attn_mask = []
    for i in ix:
        start = doc_starts[i]
        end = min(doc_ends[i], start + block_size)
        padded.append(torch.from_numpy(np.concatenate([data[start:end], [pad_token_id] * (start + block_size - end)]).astype(np.int64)))
        attn_mask.append(torch.from_numpy(np.concatenate([[1] * (end - start), [0] * (start + block_size - end)]).astype(bool)))
    
    return torch.stack(padded), torch.stack(attn_mask)

def get_test_batch(data_filename, labels_filename, batch_size, block_size, device, probe_type='token', doc_data=None):
    """Get a batch of test data"""
    data = np.memmap(data_filename, dtype=np.uint32, mode='r')
    labels = np.memmap(labels_filename, dtype=bool, mode='r')
    
    if probe_type == 'document':
        # For document probes, sample from documents
        doc_starts, doc_ends, doc_labels = doc_data
        ix = torch.randint(0, len(doc_starts), (batch_size,))
        x, attn_mask = pad_batch(data, ix, doc_starts, doc_ends, block_size)
        y = torch.from_numpy(doc_labels[ix].astype(bool))
        
        if device == 'cuda':
            x, y, attn_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), attn_mask.pin_memory().to(device, non_blocking=True)
        else:
            x, y, attn_mask = x.to(device), y.to(device), attn_mask.to(device)
        
        return x, y, attn_mask
    else:
        # For token/sentence probes, sample token sequences
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((labels[i:i+block_size]).astype(np.int64)) for i in ix])

        if device == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        
        return x, y, None

def collect_features_labels(
    model, layer, data_filename, labels_filename,
    batch_size, block_size, device, n_test_batches=100, tokenizer_type='roberta', probe_type='token', doc_data=None
):
    
    all_features = []
    all_labels = []

    for _ in tqdm(range(n_test_batches), desc="collecting test features"):
        if probe_type == 'document':
            x, y, attn_mask = get_test_batch(data_filename, labels_filename, batch_size, block_size, device, probe_type, doc_data)
        else:
            x, y, attn_mask = get_test_batch(data_filename, labels_filename, batch_size, block_size, device, probe_type)

        if tokenizer_type == 'gpt':
            # Handle dual GPT models
            features = get_gpt_hidden_states(model['left'], model['right'], x, layer)
            
            if probe_type == 'document':
                # For document probes, aggregate features using attention mask
                # features is [batch_size, block_size, n_embd * 2]
                # attn_mask is [batch_size, block_size]
                features = features * attn_mask.unsqueeze(-1)  # [batch_size, block_size, n_embd * 2]
                doc_features = features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1)  # [batch_size, n_embd * 2]
                all_features.append(doc_features.cpu().numpy())
                all_labels.append(y.cpu().numpy())
            else:
                features = features.view(-1, model['left'].config.n_embd * 2)  # concatenated features
                labels = y.view(-1)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        else:
            # Handle other models (BERT, RoBERTa)
            with torch.no_grad():
                outputs = model(**prep_tokens(x))
            features = outputs.hidden_states[layer]
            
            if probe_type == 'document':
                # For document probes, aggregate features using attention mask
                # features is [batch_size, block_size, hidden_size]
                # attn_mask is [batch_size, block_size]
                features = features * attn_mask.unsqueeze(-1)  # [batch_size, block_size, hidden_size]
                doc_features = features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1)  # [batch_size, hidden_size]
                all_features.append(doc_features.cpu().numpy())
                all_labels.append(y.cpu().numpy())
            else:
                features = features.view(-1, model.config.hidden_size)
                labels = y.view(-1)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    return (
        np.vstack(all_features),
        np.concatenate(all_labels),
    )

def evaluate_probe_with_thresholds(probe, features, labels, thresholds):
    """Evaluate probe with different thresholds"""
    # Get prediction probabilities
    probabilities = probe.predict_proba(features)[:, 1]  # probabilities for positive class
    
    results = []
    
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division='warn'
        )
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results, probabilities

def plot_probability_histogram(probabilities, labels, output_path):
    """Plot histogram of predicted probabilities using plotnine"""
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'probability': probabilities,
        'true_label': labels
    })
    plot_data.to_csv('results/probabilities.csv', index=False)

    plot_data['true_label_str'] = plot_data['true_label'].astype(str).replace({'0': 'non-medical', '1': 'medical'})
    p_stacked = (ggplot(plot_data, aes(x='probability', fill='true_label_str')) +
                geom_histogram(bins=30, alpha=0.7, color='black', position='stack') +
                scale_fill_manual(values=['#fc8d62', '#66c2a6']) +
                labs(
                    title='probability distribution by label',
                    x='p(label = medical)',
                    y='frequency',
                    fill='true label'
                ) +
                theme_bw(base_family='Palatino') +
                theme(
                    plot_title=element_text(size=14, ha='center'),
                    axis_title_x=element_text(size=12),
                    axis_title_y=element_text(size=12),
                    legend_title=element_text(size=11),
                    strip_text=element_text(size=11),
                    strip_background=element_blank(),
                    legend_position='bottom'
                ))
    
    p_stacked.save(output_path, width=10, height=6, dpi=300)

def plot_metrics_vs_threshold(results_df, output_path):
    """Plot metrics vs threshold with 4 facets in one row"""
    
    # Melt the dataframe to long format for faceting
    metrics_data = pd.melt(results_df, 
                          id_vars=['threshold'], 
                          value_vars=['accuracy', 'precision', 'recall', 'f1'],
                          var_name='metric', 
                          value_name='value')
    
    # Create plot with 4 facets in one row
    p = (ggplot(metrics_data, aes(x='threshold', y='value')) +
         geom_line(size=1, color='#67c2a5') +
         geom_point(size=3, stroke=0, alpha=0.9, fill = '#67c2a5') +
         geom_point(fill="none", stroke=0.5, size=3, color="#4f4f4f") +
         facet_wrap('metric', ncol=4, scales='free_y') +
         labs(
             x='threshold',
             y='value'
         ) +
         scale_y_continuous(limits=(0, 1)) +
         scale_color_brewer(type='qual', palette='Set2') +
         theme_bw(base_family='Palatino') +
         theme(
             plot_title=element_text(size=14, ha='center'),
             axis_title_x=element_text(size=12),
             axis_title_y=element_text(size=12),
             strip_text=element_text(size=11),
             strip_background=element_blank(),
             axis_text_x=element_text(angle=45, hjust=1),
             figure_size=(16, 4)
         ))
    
    p.save(output_path, width=16, height=4, dpi=300)

def main():
    args = parse_args()
    
    # Load configuration
    cfg = load_config(args.cfg)
    
    # Extract config values with defaults
    model_path = cfg.get('model_path', MODEL_PATH)
    # model_name = cfg.get('model_name', 'roberta-edu')
    model_name = 'pubmed-113M'
    data_path = cfg.get('data_path', DATA_PATH)
    batch_size = cfg.get('batch_size', 32)
    block_size = cfg.get('block_size', 512)
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_type = cfg.get('tokenizer', 'roberta')
    n_test_batches = 100
    
    # Detect probe type from filename
    probe_type = detect_probe_type(args.probe_path)
    print(f"Detected probe type: {probe_type}")
    
    # Load the pretrained probe
    with open(args.probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    
    probe = probe_data['probe']
    layer = probe_data.get('layer', probe_data.get('layers', [0])[0] if 'layers' in probe_data else 0)
    print(f"Probe layer: {layer}")

    # Handle multi-layer probes
    if isinstance(layer, list):
        layer = layer[0]  # Use first layer for now
        print(f"Multi-layer probe detected, using layer {layer}")
    
    if tokenizer_type == 'bert':
        long_model_name = model_name
        model_name = model_name.split('/')[-1]
        model = AutoModel.from_pretrained(long_model_name)
        model.to(device)
        model.config.output_hidden_states = True
    elif tokenizer_type == 'gpt':
        # Load both left and right models
        print(f'loading left-{model_name}.pt and right-{model_name}.pt')
        left_model = load_gpt(os.path.join(model_path, f'left-{model_name}.pt'), device)
        right_model = load_gpt(os.path.join(model_path, f'right-{model_name}.pt'), device)
        model = {'left': left_model, 'right': right_model}
        model['left'].to(device)
        model['right'].to(device)
        model['left'].eval()
        model['right'].eval()
    else:
        model = AutoModelForMaskedLM.from_pretrained(os.path.join(model_path, model_name))
        model.to(device)
        model.config.output_hidden_states = True
    
    # Initialize tokenizer to get special token IDs
    if tokenizer_type == 'bert':
        enc = AutoTokenizer.from_pretrained(long_model_name)
        special_token_ids = enc.all_special_tokens
    else:
        enc = tiktoken.get_encoding("cl100k_base")
        special_token_ids = {0, enc.eot_token}
    
    # Get test data paths based on probe type
    probe_dir = 'token' # if probe_type != 'token' else 'sentence'  # token probes use sentence directory
    test_data_filename = os.path.join(data_path, 'probe', f'{probe_dir}/test_tokens.bin')
    test_labels_filename = os.path.join(data_path, 'probe', f'{probe_dir}/test_labels.bin')
    
    # For document probes, prepare document boundaries
    doc_data = None
    if probe_type == 'document':
        # Load data to find document boundaries
        global_data = np.memmap(test_data_filename, dtype=np.uint32, mode='r')
        global_labels = np.memmap(test_labels_filename, dtype=bool, mode='r')
        
        # Find EOT token based on tokenizer type
        if tokenizer_type == 'gpt' or tokenizer_type == 'roberta':
            eot_token = enc.eot_token
        else:
            eot_token = enc.sep_token_id
        
        # Find document boundaries
        doc_starts = np.concatenate([[0], np.where(global_data == eot_token)[0][:-1] + 1])
        doc_ends = np.where(global_data == eot_token)[0]
        doc_labels = global_labels[doc_starts]
        
        doc_data = (doc_starts, doc_ends, doc_labels)
        print(f"Found {len(doc_starts)} documents in test data")
    
    print(f"Loading test data from {test_data_filename}")
    
    # Collect test features and labels
    features, labels = collect_features_labels(
        model, layer, test_data_filename, test_labels_filename,
        batch_size, block_size, device, n_test_batches, tokenizer_type, probe_type, doc_data
    )

    # features = normalize(features, norm='l2')
    
    if len(features) == 0:
        print("No valid test features found!")
        return
        
    # Generate thresholds
    thresholds = np.arange(args.min_threshold, args.max_threshold + args.threshold_step, args.threshold_step)

    # Evaluate probe with different thresholds
    results, probabilities = evaluate_probe_with_thresholds(probe, features, labels, thresholds)

    # Convert to DataFrame
    df = pd.DataFrame(results)
        
    df['layer'] = layer
    df['model_name'] = model_name
    df['probe_path'] = args.probe_path
    df['n_test_batches'] = n_test_batches

    df.to_csv('results/thresholds.csv', index=False)
    plot_probability_histogram(probabilities, labels, 'results/probabilities.png')
    plot_metrics_vs_threshold(df, 'results/thresholds.png')

    # Compute auroc
    try:
        auroc = roc_auc_score(labels, probabilities)
        print(f"auroc: {auroc:.4f}")
    except Exception as e:
        print(f"could not compute auroc: {e}")
    
    best_f1_idx = df['f1'].idxmax()
    best_acc_idx = df['accuracy'].idxmax()
    best_precision_recall_idx = np.argmin(np.abs(df['precision'] - df['recall']))
    
    print(f"best accuracy: {df.loc[best_acc_idx, 'accuracy']:.4f} @ p > {df.loc[best_acc_idx, 'threshold']:.3f}")
    print(f"best f1: {df.loc[best_f1_idx, 'f1']:.4f} @ p > {df.loc[best_f1_idx, 'threshold']:.3f}")
    print(f"best precision-recall: {df.loc[best_precision_recall_idx, 'f1']:.4f} @ p > {df.loc[best_precision_recall_idx, 'threshold']:.3f}")
    
if __name__ == "__main__":
    main() 