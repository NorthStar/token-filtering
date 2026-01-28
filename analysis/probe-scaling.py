import os
import sys
import pickle
import numpy as np
import torch
import tiktoken
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
import argparse
import pandas as pd
from plotnine import *
from pypalettes import load_cmap

# Add parent directory to path for model imports
sys.path.append('..')
from model import GPT, GPTConfig
from paths import DATA_PATH, MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'bidir'))
parser.add_argument('--data_path',  type=str, default=DATA_PATH)
parser.add_argument('--rerun', action='store_true', help='rerun results')
parser.add_argument('--device',     type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_batches', type=int, default=3200)
args = parser.parse_args()

def load_gpt(model_file):
    """Load GPT model from checkpoint"""
    checkpoint = torch.load(model_file, map_location=args.device)
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

def load_model(model_name, model_path=args.model_path):
    """Load the base model based on tokenizer type"""
    print(f'Loading left-{model_name}.pt and right-{model_name}.pt')
    left_model = load_gpt(os.path.join(model_path, f'left-{model_name}.pt'))
    right_model = load_gpt(os.path.join(model_path, f'right-{model_name}.pt'))
    left_model.to(args.device)
    right_model.to(args.device)
    left_model.eval()
    right_model.eval()
    
    # create combined config
    model_config = left_model.config
    model_config.hidden_size = left_model.config.n_embd * 2  # Concatenated features
    model_config.num_hidden_layers = left_model.config.n_layer
    
    return {'left': left_model, 'right': right_model}, model_config

def load_probe(model_name, model_path=args.model_path):
    """Load the trained probe"""
    probe_path = os.path.join(model_path.replace('bidir', ''), f'probes/{model_name}-token.pkl')
    
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    
    return probe_data['probe'], probe_data['layer']

def prep_tokens(token_ids, pad_token_id=0):
    """Prepare tokens for transformer models"""
    return {'input_ids': token_ids, 'attention_mask': (token_ids != pad_token_id).long()}

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

def extract_features(model, tokens, layer, model_type='gpt'):
    """Extract features from the model at specified layer"""
    x = torch.from_numpy(tokens.astype(np.int64)).to(args.device)
    if model_type == 'gpt':
        features = get_gpt_hidden_states(model['left'], model['right'], x, layer)
    else:
        
        with torch.no_grad():
            outputs = model(**prep_tokens(x))
        
        features = outputs.hidden_states[layer]
    return features.cpu().numpy()

enc = tiktoken.get_encoding("cl100k_base")
special_token_ids = {0, enc.eot_token}

def get_block_size(model_type):
    """Get appropriate block size for model type."""
    if model_type == 'bert':  # ModernBERT
        return 8192
    elif model_type == 'gpt':
        return 1024
    else:  # roberta
        return 512


def get_probabilities(model, hidden_size, probe, layer, dataset, model_type='gpt', probe_folder='probe'):
    """Extract features and compute probe probabilities for a dataset.

    Returns balanced labels and probabilities.
    """
    block_size = get_block_size(model_type)

    # load data
    tokens_path = os.path.join(args.data_path, f'{probe_folder}/token/{dataset}.bin')
    labels_path = os.path.join(args.data_path, f'{probe_folder}/token/{dataset.replace("tokens", "labels")}.bin')

    tokens = np.memmap(tokens_path, dtype=np.uint32, mode='r')
    labels = np.memmap(labels_path, dtype=bool, mode='r')

    all_features = []
    all_labels = []

    chunk_size = args.batch_size * block_size
    n_chunks = min(args.n_batches // args.batch_size, (len(tokens) - block_size) // chunk_size)

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(tokens) - block_size)

        if start_idx >= end_idx:
            break

        chunk_tokens = tokens[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]

        n_seqs = len(chunk_tokens) // block_size
        if n_seqs == 0:
            continue

        chunk_tokens = chunk_tokens[:n_seqs * block_size].reshape(n_seqs, block_size)
        chunk_labels = chunk_labels[:n_seqs * block_size].reshape(n_seqs, block_size)

        features = extract_features(model, chunk_tokens, layer, model_type)
        features = features.reshape(-1, hidden_size)
        chunk_labels = chunk_labels.reshape(-1)

        token_ids = chunk_tokens.reshape(-1)
        mask = np.ones(len(token_ids), dtype=bool)
        for special_id in special_token_ids:
            mask &= (token_ids != special_id)

        if mask.sum() > 0:
            all_features.append(features[mask])
            all_labels.append(chunk_labels[mask])

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    # balance labels
    min_count = min(labels.sum(), (~labels).sum())
    print(f"  {dataset}: balancing to {min_count} medical and {min_count} non-medical")

    medical_indices = np.where(labels)[0]
    non_medical_indices = np.where(~labels)[0]

    np.random.seed(42)
    sampled_medical = np.random.choice(medical_indices, min_count, replace=False)
    sampled_non_medical = np.random.choice(non_medical_indices, min_count, replace=False)

    balanced_indices = np.concatenate([sampled_medical, sampled_non_medical])
    np.random.shuffle(balanced_indices)

    features = features[balanced_indices]
    labels = labels[balanced_indices]

    probabilities = probe.predict_proba(features)[:, 1]

    return labels, probabilities


def compute_metrics_at_threshold(labels, probabilities, threshold):
    """Compute accuracy, precision, recall, F1 at a given threshold."""
    predictions = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return accuracy, precision, recall, f1


def find_optimal_threshold(labels, probabilities):
    """Find threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx]


def evaluate_probe(model, hidden_size, probe, layer, dataset, model_type='gpt', probe_folder='probe'):
    """Evaluate probe on a specific dataset (legacy function for compatibility)."""
    labels, probabilities = get_probabilities(model, hidden_size, probe, layer, dataset, model_type, probe_folder)

    auroc = roc_auc_score(labels, probabilities)
    threshold = find_optimal_threshold(labels, probabilities)
    accuracy, precision, recall, f1 = compute_metrics_at_threshold(labels, probabilities, threshold)

    results = [
        {'model' : model_name, 'dataset' : dataset, 'score' : auroc, 'metric' : 'auroc'},
        {'model' : model_name, 'dataset' : dataset, 'score' : accuracy, 'metric' : 'accuracy'},
        {'model' : model_name, 'dataset' : dataset, 'score' : precision, 'metric' : 'precision'},
        {'model' : model_name, 'dataset' : dataset, 'score' : recall, 'metric' : 'recall'},
        {'model' : model_name, 'dataset' : dataset, 'score' : f1, 'metric' : 'f1'}
    ]

    # compute F1 when exactly 25% of tokens are classified as medical
    fixed_threshold = np.percentile(probabilities, 75)
    fixed_accuracy, fixed_precision, fixed_recall, fixed_f1 = compute_metrics_at_threshold(
        labels, probabilities, fixed_threshold
    )

    fixed_results = [
        {'model' : model_name, 'dataset' : dataset, 'score' : auroc, 'metric' : 'auroc'},
        {'model' : model_name, 'dataset' : dataset, 'score' : fixed_accuracy, 'metric' : 'accuracy'},
        {'model' : model_name, 'dataset' : dataset, 'score' : fixed_precision, 'metric' : 'precision'},
        {'model' : model_name, 'dataset' : dataset, 'score' : fixed_recall, 'metric' : 'recall'},
        {'model' : model_name, 'dataset' : dataset, 'score' : fixed_f1, 'metric' : 'f1'}
    ]

    return results, [{'model' : model_name, 'dataset' : dataset, 'threshold' : threshold}], fixed_results


def evaluate_probe_cross_threshold(model, model_name, hidden_size, probe, layer, model_type='gpt', probe_folder='probe'):
    """Evaluate probe with cross-threshold analysis.

    Computes optimal threshold on train and test separately, then reports metrics
    for both datasets at each threshold.

    Returns list of dicts with columns:
        model, threshold_source, eval_dataset, accuracy, precision, recall, f1, auroc, threshold
    """
    # Get probabilities for both datasets
    print(f"  Computing probabilities for train and test...")
    train_labels, train_probs = get_probabilities(model, hidden_size, probe, layer, 'tokens', model_type, probe_folder)
    test_labels, test_probs = get_probabilities(model, hidden_size, probe, layer, 'test_tokens', model_type, probe_folder)

    # Compute AUROC for each dataset
    train_auroc = roc_auc_score(train_labels, train_probs)
    test_auroc = roc_auc_score(test_labels, test_probs)

    # Find optimal thresholds
    train_optimal_thresh = find_optimal_threshold(train_labels, train_probs)
    test_optimal_thresh = find_optimal_threshold(test_labels, test_probs)

    results = []

    # At train-optimal threshold: evaluate both datasets
    train_acc, train_prec, train_rec, train_f1 = compute_metrics_at_threshold(train_labels, train_probs, train_optimal_thresh)
    test_acc, test_prec, test_rec, test_f1 = compute_metrics_at_threshold(test_labels, test_probs, train_optimal_thresh)

    results.append({
        'model': model_name, 'threshold_source': 'train', 'eval_dataset': 'train',
        'accuracy': train_acc, 'precision': train_prec, 'recall': train_rec, 'f1': train_f1,
        'auroc': train_auroc, 'threshold': train_optimal_thresh
    })
    results.append({
        'model': model_name, 'threshold_source': 'train', 'eval_dataset': 'test',
        'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1,
        'auroc': test_auroc, 'threshold': train_optimal_thresh
    })

    # At test-optimal threshold: evaluate both datasets
    train_acc, train_prec, train_rec, train_f1 = compute_metrics_at_threshold(train_labels, train_probs, test_optimal_thresh)
    test_acc, test_prec, test_rec, test_f1 = compute_metrics_at_threshold(test_labels, test_probs, test_optimal_thresh)

    results.append({
        'model': model_name, 'threshold_source': 'test', 'eval_dataset': 'train',
        'accuracy': train_acc, 'precision': train_prec, 'recall': train_rec, 'f1': train_f1,
        'auroc': train_auroc, 'threshold': test_optimal_thresh
    })
    results.append({
        'model': model_name, 'threshold_source': 'test', 'eval_dataset': 'test',
        'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1,
        'auroc': test_auroc, 'threshold': test_optimal_thresh
    })

    # Print summary
    print(f"\n  === {model_name} Cross-Threshold Analysis ===")
    print(f"  Train optimal threshold: {train_optimal_thresh:.4f}")
    print(f"  Test optimal threshold:  {test_optimal_thresh:.4f}")
    print(f"\n  At train-optimal threshold ({train_optimal_thresh:.4f}):")
    print(f"    Train F1: {results[0]['f1']:.4f}, Acc: {results[0]['accuracy']:.4f}")
    print(f"    Test F1:  {results[1]['f1']:.4f}, Acc: {results[1]['accuracy']:.4f}")
    print(f"\n  At test-optimal threshold ({test_optimal_thresh:.4f}):")
    print(f"    Train F1: {results[2]['f1']:.4f}, Acc: {results[2]['accuracy']:.4f}")
    print(f"    Test F1:  {results[3]['f1']:.4f}, Acc: {results[3]['accuracy']:.4f}")

    return results

if 'probe-scaling.csv' not in os.listdir('results') or args.rerun:

    all_results = []
    all_thresholds = []
    all_fixed_results = []
    all_cross_threshold_results = []

    models = ['roberta-edu', 'edu-61M']
    for file in os.listdir("../config/bidir"):
        if 'left' not in file or 'fineweb' in file:
            continue

        if file.endswith('.yaml'):
            models.append('pubmed-' + file.split('-')[2].split('.')[0])

    for model_name in models:
        print(f"evaluating {model_name}...")

        if 'roberta' not in model_name and 'ModernBERT' not in model_name:
            model_type = 'gpt'
            model, model_config = load_model(model_name)
            hidden_size = model_config.hidden_size
        elif 'ModernBERT' not in model_name:
            model_type = 'roberta'
            model = AutoModelForMaskedLM.from_pretrained(os.path.join(args.model_path.replace('bidir', ''), model_name))
            model.config.output_hidden_states = True
            model.to(args.device)
            hidden_size = model.config.hidden_size
        else:
            model_type = 'bert'
            model = AutoModelForMaskedLM.from_pretrained('answerdotai/ModernBERT-large')
            model.config.output_hidden_states = True
            model.to(args.device)
            hidden_size = model.config.hidden_size

        probe, layer = load_probe(model_name)
        print(f"loaded probe for {model_name} at layer {layer}")

        if model_type == 'bert':
            probe_folder = 'ModernBERT-large'
        else:
            probe_folder = 'probe'

        # Cross-threshold analysis (evaluates both datasets, compares thresholds)
        cross_results = evaluate_probe_cross_threshold(
            model, model_name, hidden_size, probe, layer, model_type, probe_folder
        )
        all_cross_threshold_results.extend(cross_results)

        # Legacy per-dataset evaluation (for backwards compatibility)
        for dataset in ['tokens', 'test_tokens']:
            results, threshold, fixed_results = evaluate_probe(model, hidden_size, probe, layer, dataset, model_type, probe_folder)
            all_results.extend(results)
            all_thresholds.extend(threshold)
            all_fixed_results.extend(fixed_results)

    df = pd.DataFrame(all_results)
    df.to_csv('results/probe-scaling.csv', index=False)

    thresholds = pd.DataFrame(all_thresholds)
    print(thresholds)
    thresholds.to_csv('../config/probe-thresholds.csv', index=False)

    fixed_df = pd.DataFrame(all_fixed_results)
    fixed_df.to_csv('results/fixed-probe-scaling.csv', index=False)
    print(f"Saved fixed threshold (25% medical) results to results/fixed-probe-scaling.csv")

    cross_df = pd.DataFrame(all_cross_threshold_results)
    cross_df.to_csv('results/cross-threshold-scaling.csv', index=False)
    print(f"Saved cross-threshold analysis to results/cross-threshold-scaling.csv")

else:

    df = pd.read_csv('results/probe-scaling.csv')

df['params'] = df['model'].apply(lambda x: int(x.split('-')[1][:-1]) * 1e6)
df['dataset'] = df['dataset'].map({'tokens': 'Validation (in distribution)', 'test_tokens': 'Test (FineWeb-Edu)'})

breaks = [10e6, 25e6, 50e6, 100e6, 200e6]
labels = [f'{int(param // 1e6)}M' for param in breaks]

cmap = load_cmap("Hokusai3")
n_categories = df['metric'].nunique()
colors = [cmap((i+1)/(n_categories+1)) for i in range(n_categories)]
# colors = [cmap(i/n_categories) for i in range(n_categories)]
hex_colors = ['#%02x%02x%02x' % tuple(int(c*255) for c in col[:3]) for col in colors]

p = (ggplot(df, aes(x='params', y='score', color='metric'))
     + geom_line(size=1)
     + geom_point(size=3, stroke=0, alpha=0.9)
     + geom_point(fill="none", stroke=0.5, size=3, color="#4f4f4f")
     + scale_x_log10(name='parameters', breaks=breaks, labels = labels, limits=[10e6, 230e6])
     + scale_y_continuous(name='', limits=[0.6, 1.0])
     + facet_wrap('~dataset', ncol=2)
     + scale_color_manual(values=hex_colors) 
     + theme_bw(base_family='Helvetica Neue')
     + theme(figure_size=(8, 4),
             strip_text_x=element_text(size=12),
             panel_grid_major=element_line(size=0.3, color="#dddddd"),
             panel_grid_minor=element_blank(),
             strip_background=element_blank(),
             legend_margin=0,
             legend_position='right')
     + labs(color='metric')
)

p.save('plots/probe-scaling.png', dpi=300, width=9, height=4)
print(p)
print("plots saved as 'plots/probe-scaling.png'")