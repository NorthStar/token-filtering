import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import tiktoken
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize

import sys
sys.path.append('..')
from model import GPT, GPTConfig

# probe config
parser = argparse.ArgumentParser()
parser.add_argument('--probe_path', type=str, default='../../../workspace/med/models/probes')
parser.add_argument("--tokenizer", type=str, default='gpt')
parser.add_argument("--model_path", type=str, default='../../../workspace/med/models')
parser.add_argument("--model_name", type=str, default='pubmed-61M')
parser.add_argument("--probe_type", type=str, default='sentence')
parser.add_argument("--block_size", type=int, default=1024)
parser.add_argument("--device", type=str, default='cuda')
args = parser.parse_args()

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

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

if args.tokenizer == 'bert':
    long_model_name = args.model_name
    model_name = args.model_name.split('/')[-1]
    model = AutoModel.from_pretrained(long_model_name)
    df = pd.read_json(f'results/{model_name}/documents.jsonl', orient='records', lines=True)
elif args.tokenizer == 'gpt':
    # Load both left and right models
    model_name = args.model_name
    print(f'loading left-{model_name}.pt and right-{model_name}.pt')
    left_model = load_gpt(os.path.join(args.model_path, f'left-{model_name}.pt'), args.device)
    right_model = load_gpt(os.path.join(args.model_path, f'right-{model_name}.pt'), args.device)
    model = {'left': left_model, 'right': right_model}
    df = pd.read_json(f'results/{model_name}/documents.jsonl', orient='records', lines=True)
else:
    model_name = args.model_name
    model = AutoModelForMaskedLM.from_pretrained(os.path.join(args.model_path, args.model_name))
    df = pd.read_json(f'results/{model_name}/documents.jsonl', orient='records', lines=True)

# df = df.sort_values(by='percent_incorrect', ascending=False)
# df = df.reset_index(drop=True)
document_indices = list(range(len(df)))

if args.tokenizer == 'gpt':
    model['left'].to(args.device)
    model['right'].to(args.device)
    model['left'].eval()
    model['right'].eval()
else:
    model.to(args.device)
    model.config.output_hidden_states = True
    model.eval()

# initialize tokenizer to get special token IDs
if args.tokenizer == 'bert':
    enc = AutoTokenizer.from_pretrained(long_model_name)
    special_token_ids = enc.all_special_tokens
else:
    enc = tiktoken.get_encoding("cl100k_base")
    special_token_ids = {0, enc.eot_token}  

if args.probe_type == 'sentence':
    with open(os.path.join(args.probe_path, f'{model_name}-sent.pkl'), "rb") as f:
        probe_data = pickle.load(f)
elif args.probe_type == 'document':
    with open(os.path.join(args.probe_path, f'{model_name}-doc.pkl'), "rb") as f:
        probe_data = pickle.load(f)
else:
    with open(os.path.join(args.probe_path, f'{model_name}-old.pkl'), "rb") as f:
        probe_data = pickle.load(f)

probe = probe_data["probe"]
layer = probe_data["layer"]

def prep_tokens(token_ids, pad_token_id = 0):

    # token_ids = torch.tensor(token_ids).to(device) 

    return {
        "input_ids": token_ids,
        "attention_mask": (token_ids != pad_token_id).long(),
    }

def collect_features(model, layer, tokens):

    if args.tokenizer == 'gpt':
        # Handle dual GPT models
        features = get_gpt_hidden_states(model['left'], model['right'], tokens.unsqueeze(0), layer)
        features = features.view(-1, model['left'].config.n_embd * 2)  # concatenated features
        features = features.cpu().numpy()
    else:
        with torch.no_grad():
            outputs = model(**prep_tokens(tokens.unsqueeze(0)))
            features = outputs.hidden_states[layer].view(-1, model.config.hidden_size)
            features = features.cpu().numpy()
            # L2 normalize features to match training preprocessing
            # features = normalize(features, norm='l2')
    
    return features

def collect_document_features(model, layer, tokens):
    """Collect document-level features by averaging token features (similar to probe.py)"""
    
    # Truncate tokens to block_size
    tokens = tokens[:args.block_size]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(args.device)  # Add batch dimension
    
    # Create attention mask (all tokens are valid, no padding in this case)
    attn_mask = torch.ones_like(tokens_tensor, dtype=torch.bool)
    
    if args.tokenizer == 'gpt':
        # Handle dual GPT models
        features = get_gpt_hidden_states(model['left'], model['right'], tokens_tensor, layer)
        features = features.view(1, -1, model['left'].config.n_embd * 2)  # [1, seq_len, hidden_size]
    else:
        with torch.no_grad():
            outputs = model(**prep_tokens(tokens_tensor))
            features = outputs.hidden_states[layer]  # [1, seq_len, hidden_size]
    
    # Apply attention mask and average features across sequence length
    features = features * attn_mask.unsqueeze(-1).float()  # [1, seq_len, hidden_size]
    doc_features = features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1).float()  # [1, hidden_size]
    
    return doc_features.cpu().numpy()

def collect_doc_predictions(document_idx):

    if args.probe_type == 'document':
        # For document probe, use the 'tokens' field and make document-level prediction
        tokens = df.iloc[document_idx]['tokens']
        tokens = tokens[:args.block_size]
        
        # Get document-level features and prediction
        doc_features = collect_document_features(model, layer, tokens)
        doc_prob = probe.predict_proba(doc_features)[0, 1]  # Single document probability
        
        # Create token-level visualization by assigning the same probability to all tokens
        probs = [doc_prob] * len(tokens)
        token_strings = [enc.decode([token]) for token in tokens]
        
        return token_strings, probs
    else:
        # For token/sentence probe, use the 'document' field (legacy)
        tokens = df.iloc[document_idx]['document']
        tokens = tokens[:args.block_size]
        tokens = torch.tensor(tokens).to(args.device)

        features = collect_features(model, layer, tokens)
        probs    = probe.predict_proba(features)[:, 1]
        tokens   = [enc.decode([token]) for token in tokens.cpu().numpy().tolist()] # convert to list

        return tokens, probs

cmap = plt.get_cmap('RdBu_r')
def logit_color(val):
    r, g, b, _ = cmap(val)
    return f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.75)'

def logit_color_ground_truth(val):
    if val == 1:
        return 'rgba(232, 191, 86, 0.75)'
    if val == 2:
        return 'rgba(154, 70, 207, 0.75)'
    else:
        return 'rgba(255, 255, 255, 0.75)'

def percent_incorrect(probs, labels):
    assert len(probs) == len(labels)
    preds = np.array(probs) > 0.5
    labs = np.array(labels)
    return ((~preds) == labs).mean(), ((preds & ~labs) + 2 * (~preds & labs)).tolist()

def annotate_document(tokens, probs, false_mask, pct_incorrect, source):

    html_tokens = []
    for tok, val in zip(tokens, probs):
        tok = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_tokens.append(f'<span style="background-color: {logit_color(val)}">{tok}</span>')

    ground_truth = []
    for tok, val in zip(tokens, false_mask):
        tok = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ground_truth.append(f'<span style="background-color: {logit_color_ground_truth(val)}">{tok}</span>')

    if args.probe_type == 'document':
        # For document probe, add document-level prediction info
        doc_prob = probs[0] if probs else 0.0  # All tokens have same probability
        doc_label = false_mask[0] if false_mask else 0  # All tokens have same label
        doc_pred = 1 if doc_prob > 0.5 else 0
        
        status = "CORRECT" if pct_incorrect < 0.01 else "INCORRECT"
        return f"""<h3>{source} | Document-level: {status} | Prob: {doc_prob:.3f} | Label: {doc_label} | Pred: {doc_pred}</h3>
{''.join(html_tokens)}
<hr>
{''.join(ground_truth)}"""
    else:
        return f"""<h3>{source} | {pct_incorrect*100:.2f}% incorrect</h3>
{''.join(html_tokens)}
<hr>
{''.join(ground_truth)}"""

html_content = f"""
<html>
<head>
<meta charset="utf-8">
<title>annotated</title>
</head>
<body style="font-family:Jet Brains Mono, monospace; font-size:14px; white-space:pre-wrap;">
"""

docs = {
    'tokens': [],
    'probs': [],
    'percent_incorrect': [],
    'false_mask': [],
    'labels' : [],
    'source' : []
}

seen_ids = set()
for idx in document_indices:
    if df['doc_id'][idx] in seen_ids:
        continue
    seen_ids.add(df['doc_id'][idx])
    try:
        tokens, probs = collect_doc_predictions(idx)
    except Exception as e:
        print(f"error collecting predictions for document {idx}: {e}")
        continue
    
    if args.probe_type == 'document':
        # For document probe, create labels array with same value for all tokens
        doc_label = df['label'][idx]
        labels = [doc_label] * len(tokens)
    else:
        # For token/sentence probe, use the 'labels' field
        labels = df['labels'][idx]
    
    docs['tokens'].append(tokens)
    docs['probs'].append(probs)
    pct_incorrect, false_mask = percent_incorrect(probs, labels)
    docs['percent_incorrect'].append(pct_incorrect)
    docs['false_mask'].append(false_mask)
    docs['labels'].append(labels)
    docs['source'].append(df['source'][idx])

df = pd.DataFrame(docs)
df = df.sort_values(by='percent_incorrect', ascending=False)

if args.probe_type == 'document':
    # For document probe, show documents that are incorrect (100% incorrect since all tokens have same label/prediction)
    # Also show some correct ones for comparison
    incorrect_docs = df[df['percent_incorrect'] > 0.99]  # Documents that are incorrect
    correct_docs = df[df['percent_incorrect'] < 0.01]   # Documents that are correct
    
    # Take top incorrect and some correct for comparison
    n_incorrect = min(20, len(incorrect_docs))
    n_correct = min(10, len(correct_docs))
    
    df = pd.concat([
        incorrect_docs.head(n_incorrect),
        correct_docs.head(n_correct)
    ]).sort_values(by='percent_incorrect', ascending=False).reset_index(drop=True)
else:
    # For token/sentence probe, use original filtering
    df = df[df['percent_incorrect'] > 0.2]
    df = df.reset_index(drop=True)

for idx in range(len(df)):
    html_tokens = annotate_document(df['tokens'][idx], df['probs'][idx], df['false_mask'][idx], df['percent_incorrect'][idx], df['source'][idx])
    html_content += html_tokens

html_content += "</body>\n</html>"
if args.tokenizer == 'bert':
    with open(f"results/{model_name}/annotated.html", "w", encoding="utf-8") as f:
        f.write(html_content)
else:
    with open(f"results/{model_name}/annotated.html", "w", encoding="utf-8") as f:
        f.write(html_content)