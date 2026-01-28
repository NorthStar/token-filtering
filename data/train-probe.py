"""
train a probe on top of RoBERTa/GPT to predict labels from synthetic data
   - layer # is a hyperparameter we sweep over
   - infra is frankensteined from huggingface and nanoGPT
"""

import os
import sys
import pickle
import gc
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.linear_model import LogisticRegression
from cuml import LogisticRegression
import cupy as cp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import normalize

import wandb
import tiktoken
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from omegaconf import OmegaConf

sys.path.append('..')
from model import GPT, GPTConfig

class MLPClassifier(nn.Module):
    """MLP classifier with ReLU activation functions"""
    
    def __init__(self, input_size, num_layers=2, hidden_size=None):
        super(MLPClassifier, self).__init__()
        
        if hidden_size is None:
            hidden_size = input_size // 2
        
        layers = []
        current_size = input_size
        
        # Add hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def predict(self, x):
        """sklearn-compatible predict method"""
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            # Move tensor to same device as model
            x = x.to(next(self.parameters()).device)
            probs = self.forward(x)
            return (probs > 0.5).cpu().numpy().astype(int)
    
    def predict_proba(self, x):
        """sklearn-compatible predict_proba method"""
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            # Move tensor to same device as model
            x = x.to(next(self.parameters()).device)
            probs = self.forward(x).cpu().numpy()
            return np.column_stack([1 - probs, probs])

# evil config magic
cfg_file = 'probe.yaml'
for i, arg in enumerate(sys.argv):
    if arg[:3] == 'cfg':
        cfg_file = arg.split('=')[1]
        sys.argv.pop(i)

cfg = OmegaConf.load(cfg_file)
cfg.update(OmegaConf.from_cli())

for key in cfg:
    try:
        exec(key + '=' + str(cfg[key]))
    except (NameError, SyntaxError) as e:
        exec(key + '="' + cfg[key] + '"')

cfg = OmegaConf.to_container(cfg)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set default for tokens_per_window if not specified (block_size = old behavior)
if 'tokens_per_window' not in cfg:
    tokens_per_window = block_size

effective_n_batches = n_batches * (block_size // tokens_per_window)
print(f"tokens_per_window={tokens_per_window} → {effective_n_batches} iterations (expected ~{effective_n_batches * batch_size * tokens_per_window} training tokens from {effective_n_batches * batch_size} windows)")

# Validate finetune classifier configuration
if classifier_type == 'finetune' and tokenizer == 'gpt':
    raise ValueError("finetune classifier_type is not supported with GPT tokenizer")

def load_gpt(model_file):
    checkpoint = torch.load(model_file, map_location=device)
    model_args = checkpoint['model_args']
    # model_args['mup_width_multiplier'] = 1

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    return model

if probe_name is None:
    probe_name = f"{model_name}-{probe_type}-{label_type}"
else:
    probe_name = f"{probe_name}-{probe_type}-{label_type}"

if tokenizer == 'bert':
    long_model_name = model_name
    model_name = model_name.split('/')[-1]
    tokenizer_name = model_name
    if classifier_type == 'finetune':
        model = AutoModelForSequenceClassification.from_pretrained(long_model_name, num_labels=2)
    else:
        model = AutoModel.from_pretrained(long_model_name)
elif tokenizer == 'roberta':
    tokenizer_name = 'probe'
    if classifier_type == 'finetune':
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, model_name), num_labels=2)
    else:
        model = AutoModelForMaskedLM.from_pretrained(os.path.join(model_path, model_name))
elif tokenizer == 'gpt':
    # Load both left and right models
    print(f'loading left-{model_name}.pt and right-{model_name}.pt')
    left_model = load_gpt(os.path.join(model_path, f'left-{model_name}.pt'))
    right_model = load_gpt(os.path.join(model_path, f'right-{model_name}.pt'))
    model = {'left': left_model, 'right': right_model}
    tokenizer_name = 'probe'

if tokenizer == 'gpt':
    model['left'].to(device)
    model['right'].to(device)
    model['left'].eval()
    model['right'].eval()
    # Use left model config as reference, assuming both models have same architecture
    model_config = model['left'].config
    # Double the hidden size since we'll concatenate features
    model_config.hidden_size = model['left'].config.n_embd * 2
    model_config.num_hidden_layers = model['left'].config.n_layer
else:
    model.to(device)
    if classifier_type != 'finetune':
        model.config.output_hidden_states = True
    model_config = model.config

# initialize tokenizer to get special token IDs
if tokenizer == 'bert':
    enc = AutoTokenizer.from_pretrained(long_model_name)
    # Convert special token strings to IDs
    special_token_ids = set(enc.convert_tokens_to_ids(enc.all_special_tokens))
else:
    # enc = tiktoken.get_encoding("cl100k_base")
    enc = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    special_token_ids = set(enc.convert_tokens_to_ids(enc.all_special_tokens))

# Validate probe_type and label_type constraints
if label_type in ['token', 'sentence'] and probe_type != 'token':
    raise ValueError(f"When label_type is '{label_type}', probe_type must be 'token' (got '{probe_type}')")
if probe_type not in ['token', 'document']:
    raise ValueError(f"probe_type must be 'token' or 'document' (got '{probe_type}')")
if label_type not in ['token', 'sentence', 'document']:
    raise ValueError(f"label_type must be 'token', 'sentence', or 'document' (got '{label_type}')")

# get data - use label_type to determine which directory to load from
label_type = label_type
data_filename = os.path.join(data_path, tokenizer_name, f'{label_type}/tokens.bin')
labels_filename = os.path.join(data_path, tokenizer_name, f'{label_type}/labels.bin')

if label_type == 'document':

    if tokenizer == 'gpt' or tokenizer == 'roberta':
        eot_token = enc.eos_token_id
    else:
        eot_token = enc.sep_token_id
    
    global_data   = np.memmap(data_filename,   dtype=np.uint16, mode='r')
    global_labels = np.memmap(labels_filename, dtype=bool,      mode='r')
    doc_starts = np.concatenate([[0], np.where(global_data == eot_token)[0][:-1] + 1])
    doc_ends = np.where(global_data == eot_token)[0]

    print('total docs', len(doc_starts))
    
    # For document-level labels, we need to get the label for each document
    # Since all tokens in a document have the same label, we can use the first token of each document
    doc_labels = global_labels[doc_starts]

def pad_batch(data, ix, doc_starts, doc_ends, block_size=block_size, pad_token_id=0):

    padded = []
    attn_mask = []
    for i in ix:

        start = doc_starts[i]
        end = min(doc_ends[i], start + block_size)
        padded.append(torch.from_numpy(np.concatenate([data[start:end], [pad_token_id] * (start + block_size - end)]).astype(np.int64)))
        attn_mask.append(torch.from_numpy(np.concatenate([[1] * (end - start), [0] * (start + block_size - end)]).astype(bool)))
    
    return torch.stack(padded), torch.stack(attn_mask)

# cheap dataloader
def get_batch(split = 'train', tts = 0.8):

    if split == 'train' or split == 'val':
        tokens_file = data_filename
        labels_file = labels_filename
    
    data = np.memmap(tokens_file, dtype=np.uint16, mode='r')
    labels = np.memmap(labels_file, dtype=bool, mode='r')

    if probe_type == 'token':
        # Sample token windows for token-level probing
        if split == 'train':
            ix = torch.randint(0, int(tts * (len(data) - block_size)), (batch_size,))
        elif split == 'val':
            ix = torch.randint(int(tts * (len(data) - block_size)), len(data) - block_size, (batch_size,))

        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((labels[i:i+block_size]).astype(np.int64)) for i in ix])
        attn_mask = torch.zeros_like(x, dtype=torch.bool)

    elif probe_type == 'document':
        # Sample full documents for document-level probing

        split_idx = int(len(doc_starts) * tts)
        if split == 'train':
            ix = torch.randperm(split_idx)[:batch_size]
        elif split == 'val':
            ix = torch.randperm(len(doc_starts) - split_idx)[:batch_size] + split_idx
                
        x, attn_mask = pad_batch(global_data, ix, doc_starts, doc_ends, block_size=block_size) # (batch_size, block_size)
        y = torch.from_numpy(doc_labels[ix].astype(bool))
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, attn_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), attn_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, attn_mask = x.to(device), y.to(device), attn_mask.to(device)
    
    return x, y, attn_mask

def prep_tokens(token_ids, pad_token_id=0):
   return {'input_ids' : token_ids, 'attention_mask' : (token_ids != pad_token_id).long()}

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

def collect_features_and_labels_for_layer(split='train', n_batches=16, target_layer=0):
    """Collect features and labels for a SINGLE layer to reduce memory usage."""
    
    all_features = []
    all_labels = []

    # Scale iterations so total tokens is always ~n_batches * batch_size * block_size
    # regardless of tokens_per_window setting
    effective_n_batches = n_batches * (block_size // tokens_per_window)
    
    for _ in range(effective_n_batches):
        
        x, y, attn_mask = get_batch(split)
        
        if tokenizer == 'gpt':
            # Handle dual GPT models - only compute for target layer
            features = get_gpt_hidden_states(model['left'], model['right'], x, target_layer)
            # features shape: [batch_size, block_size, hidden_size]
            
            if probe_type == 'token':
                # For each window, identify valid (non-special) token positions
                # x shape: [batch_size, block_size]
                batch_features = []
                batch_labels = []
                
                for b in range(x.size(0)):
                    window_tokens = x[b]  # [block_size]
                    window_features = features[b]  # [block_size, hidden_size]
                    window_labels = y[b]  # [block_size]
                    
                    # Find valid (non-special) token positions
                    valid_mask = torch.ones(window_tokens.size(0), dtype=torch.bool, device=device)
                    for special_id in special_token_ids:
                        valid_mask &= (window_tokens != special_id)
                    
                    valid_indices = torch.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # Randomly sample tokens_per_window positions from valid tokens
                        n_to_sample = min(tokens_per_window, len(valid_indices))
                        sampled_indices = valid_indices[torch.randperm(len(valid_indices), device=device)[:n_to_sample]]
                        
                        # Clone to avoid keeping reference to full features tensor
                        batch_features.append(window_features[sampled_indices].clone())
                        batch_labels.append(window_labels[sampled_indices].clone())
                
                # Free full features tensor before accumulating
                del features
                torch.cuda.empty_cache()
                
                if batch_features:
                    all_features.append(torch.cat(batch_features).cpu().numpy())
                    all_labels.append(torch.cat(batch_labels).cpu().numpy())
                    del batch_features, batch_labels
            
            elif probe_type == 'document':
                # Aggregate features to document-level
                labels = y
                features = features.view(-1, model_config.hidden_size)

                features = features * attn_mask.view(-1).unsqueeze(1)
                features = features.view(batch_size, -1, model_config.hidden_size)
                doc_features = features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1)
                
                all_features.append(doc_features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                del features, doc_features
        else:
            # Handle other models (BERT, RoBERTa)
            with torch.no_grad():
                inputs = prep_tokens(x)
                outputs = model(**inputs)
            
            features = outputs.hidden_states[target_layer]  # [batch_size, block_size, hidden_size]
            
            # Free hidden states for other layers immediately
            del outputs
            torch.cuda.empty_cache()

            if probe_type == 'token':
                # For each window, identify valid (non-special) token positions
                batch_features = []
                batch_labels = []
                
                for b in range(x.size(0)):
                    window_tokens = x[b]  # [block_size]
                    window_features = features[b]  # [block_size, hidden_size]
                    window_labels = y[b]  # [block_size]
                    
                    # Find valid (non-special) token positions
                    valid_mask = torch.ones(window_tokens.size(0), dtype=torch.bool, device=device)
                    for special_id in special_token_ids:
                        valid_mask &= (window_tokens != special_id)
                    
                    valid_indices = torch.where(valid_mask)[0]
                    
                    if len(valid_indices) > 0:
                        # Randomly sample tokens_per_window positions from valid tokens
                        n_to_sample = min(tokens_per_window, len(valid_indices))
                        sampled_indices = valid_indices[torch.randperm(len(valid_indices), device=device)[:n_to_sample]]
                        
                        # Clone to avoid keeping reference to full features tensor
                        batch_features.append(window_features[sampled_indices].clone())
                        batch_labels.append(window_labels[sampled_indices].clone())
                
                # Free full features tensor before accumulating
                del features
                torch.cuda.empty_cache()
                
                if batch_features:
                    all_features.append(torch.cat(batch_features).cpu().numpy())
                    all_labels.append(torch.cat(batch_labels).cpu().numpy())
                    del batch_features, batch_labels
            
            elif probe_type == 'document':
                # Aggregate features to document-level
                labels = y
                features = features.view(-1, model_config.hidden_size)

                features = features * attn_mask.view(-1).unsqueeze(1)
                features = features.view(batch_size, -1, model_config.hidden_size)
                doc_features = features.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1)
                
                all_features.append(doc_features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                del features, doc_features
        
    if all_features:
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)

        # For token-level probing, balance the dataset by sampling equal numbers of medical and non-medical tokens
        if probe_type == 'token' and split == 'train':
            # Count medical (label=1) and non-medical (label=0) tokens
            medical_mask = all_labels == 1
            non_medical_mask = all_labels == 0
            
            n_medical = medical_mask.sum()
            n_non_medical = non_medical_mask.sum()
            
            print(f"Layer {target_layer}: Found {n_medical} medical tokens, {n_non_medical} non-medical tokens")
            
            # Sample the minimum count from both classes
            min_count = min(n_medical, n_non_medical)
            if min_count > 0:
                # Get indices of medical and non-medical tokens
                medical_indices = np.where(medical_mask)[0]
                non_medical_indices = np.where(non_medical_mask)[0]
                
                # Randomly sample min_count from each class
                np.random.seed(42)  # For reproducibility
                sampled_medical = np.random.choice(medical_indices, min_count, replace=False)
                sampled_non_medical = np.random.choice(non_medical_indices, min_count, replace=False)
                
                # Combine and shuffle the balanced indices
                balanced_indices = np.concatenate([sampled_medical, sampled_non_medical])
                np.random.shuffle(balanced_indices)
                
                # Apply balanced sampling
                all_features = all_features[balanced_indices]
                all_labels = all_labels[balanced_indices]
                
                print(f"Layer {target_layer}: Balanced dataset to {min_count} medical + {min_count} non-medical = {len(balanced_indices)} total tokens")
            else:
                print(f"Layer {target_layer}: Warning - no tokens of one or both classes found, skipping balancing")
    else:
        all_features = np.array([])
        all_labels = np.array([])
        
    return all_features, all_labels

def batched_predict(probe, features, batch_size=50000):
    """Predict in batches to avoid GPU OOM with large feature arrays."""
    n_samples = len(features)
    predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch = features[i:i+batch_size]
        pred = probe.predict(batch)
        # Convert cupy array to numpy if needed
        if hasattr(pred, 'get'):
            pred = pred.get()
        predictions.append(pred)
        # Free GPU memory between batches
        cp.get_default_memory_pool().free_all_blocks()
    
    return np.concatenate(predictions)

def train_probe(features, labels, layer = 0, reg_strength = 1.0, max_iter = 100, finetune_model = None):
    
    if classifier_type == 'linear' or classifier_type == 'multi_linear':
        # initialize sklearn logistic regression
        # convert lr to regularization strength (C = 1/alpha, roughly inverse relationship)
        if reg_strength is None:
            probe = LogisticRegression(
                penalty=None,
                max_iter=max_iter
            )
        else:
            probe = LogisticRegression(
                C=1/reg_strength,
                max_iter=max_iter
            )
        
        # train the probe
        print(f"training sklearn probe on {len(features['train'])} training samples...")
        probe.fit(features['train'], labels['train'])
        
    elif classifier_type == 'mlp':
        # initialize MLP classifier
        input_size = features['train'].shape[1]
        probe = MLPClassifier(input_size, num_layers=mlp_layers).to(device)
        
        # convert data to tensors
        X_train = torch.from_numpy(features['train']).float().to(device)
        y_train = torch.from_numpy(labels['train'].astype(np.float32)).to(device)
        X_val = torch.from_numpy(features['val']).float().to(device)
        y_val = torch.from_numpy(labels['val'].astype(np.float32)).to(device)
        
        # setup training
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(probe.parameters(), lr=0.001, weight_decay=1/reg_strength if reg_strength else 0.0)
        
        print(f"training MLP probe on {len(features['train'])} training samples...")
        
        # training loop
        probe.train()
        for epoch in range(max_iter):
            optimizer.zero_grad()
            outputs = probe(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        probe.eval()
    
    elif classifier_type == 'finetune':
        # For finetune, we use the model directly as the probe
        probe = finetune_model if finetune_model is not None else model
        
        # setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(probe.parameters(), lr=1e-4, weight_decay=1/reg_strength if reg_strength else 0.0)
        
        print(f"finetuning model on training data...")
        
        # training loop
        probe.train()
        for epoch in range(max_iter):
            optimizer.zero_grad()
            
            # Get a batch of training data
            x_batch, y_batch, attn_mask_batch = get_batch('train')
            
            # Prepare inputs for the model
            inputs = prep_tokens(x_batch)
            inputs['attention_mask'] = attn_mask_batch
            
            # Forward pass
            outputs = probe(**inputs)
            logits = outputs.logits
            
            if probe_type == 'document':
                # For document classification, aggregate features
                loss = criterion(logits.view(-1, 2), y_batch.long())
            else:
                # For token classification, flatten and mask special tokens
                logits_flat = logits.view(-1, 2)
                labels_flat = y_batch.view(-1).long()
                token_ids_flat = x_batch.view(-1)
                
                # create mask to exclude special tokens
                special_token_mask = torch.ones(token_ids_flat.size(0), dtype=torch.bool, device=device)
                for special_id in special_token_ids:
                    special_token_mask &= (token_ids_flat != special_id)
                
                if special_token_mask.sum() > 0:
                    masked_logits = logits_flat[special_token_mask]
                    masked_labels = labels_flat[special_token_mask]
                    loss = criterion(masked_logits, masked_labels)
                else:
                    continue  # Skip if no valid tokens
            
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        probe.eval()
    
    # evaluate on training and validation sets
    if classifier_type == 'finetune':
        # For finetune, we need to evaluate differently since we don't have pre-extracted features
        def evaluate_finetune_model(split='train'):
            all_predictions = []
            all_labels = []
            
            probe.eval()
            with torch.no_grad():
                for _ in range(16):  # Use same number of batches as feature collection
                    x_batch, y_batch, attn_mask_batch = get_batch(split)
                    
                    inputs = prep_tokens(x_batch)
                    inputs['attention_mask'] = attn_mask_batch
                    
                    outputs = probe(**inputs)
                    logits = outputs.logits
                    
                    if probe_type == 'document':
                        predictions = torch.argmax(logits, dim=-1)
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(y_batch.cpu().numpy())
                    else:
                        # For token-level classification, flatten and mask special tokens
                        logits_flat = logits.view(-1, 2)
                        labels_flat = y_batch.view(-1)
                        token_ids_flat = x_batch.view(-1)
                        
                        # create mask to exclude special tokens
                        special_token_mask = torch.ones(token_ids_flat.size(0), dtype=torch.bool, device=device)
                        for special_id in special_token_ids:
                            special_token_mask &= (token_ids_flat != special_id)
                        
                        if special_token_mask.sum() > 0:
                            masked_logits = logits_flat[special_token_mask]
                            masked_labels = labels_flat[special_token_mask]
                            predictions = torch.argmax(masked_logits, dim=-1)
                            all_predictions.extend(predictions.cpu().numpy())
                            all_labels.extend(masked_labels.cpu().numpy())
            
            return np.array(all_predictions), np.array(all_labels)
        
        train_predictions, train_labels_eval = evaluate_finetune_model('train')
        val_predictions, val_labels_eval = evaluate_finetune_model('val')
        
        train_acc = accuracy_score(train_labels_eval, train_predictions)
        val_acc = accuracy_score(val_labels_eval, val_predictions)
        
        # calculate training metrics
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels_eval, train_predictions, average='binary', zero_division=0
        )
        
        # calculate validation metrics
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels_eval, val_predictions, average='binary', zero_division=0
        )
    else:
        # evaluate on training set using batched predictions to avoid GPU OOM
        train_predictions = batched_predict(probe, features['train'])
        train_acc = accuracy_score(labels['train'], train_predictions)
            
        # calculate training metrics
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            labels['train'], train_predictions, average='binary', zero_division=0
        )

        val_predictions = batched_predict(probe, features['val'])
        val_acc = accuracy_score(labels['val'], val_predictions)
            
        # calculate validation metrics
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            labels['val'], val_predictions, average='binary', zero_division=0
        )
    
    # log metrics
    try:
        wandb.log({
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })
    except:
        pass
    
    # Clean up GPU memory from cuml
    cp.get_default_memory_pool().free_all_blocks()
    
    return probe, val_acc

def greedy_search(train_features, train_labels, val_features, val_labels, layers, reg = 1.0, max_iter = 100, num_layers = 1):

    best_layers = []
    best_probe = None
    best_val_acc = 0

    for i in range(num_layers):

        best_layer = None
        all_layers = sorted(list(set(layers) - set(best_layers)))
        for layer in all_layers:

            features = {}
            labels = {}

            features['train'], labels['train'] = train_features[layer], train_labels[layer]
            features['val'], labels['val'] = val_features[layer], val_labels[layer]

            for l in best_layers:
                features['train'] = np.concatenate([features['train'], train_features[l]], axis=-1)
                features['val'] = np.concatenate([features['val'], val_features[l]], axis=-1)
            
            if len(features['train']) == 0 or len(features['val']) == 0:
                print("no valid features found, skipping this layer")
                continue

            probe, val_acc = train_probe(features, labels, layer, reg, max_iter, None)
            print(f"layer {layer} | val acc {val_acc:.3f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_layer = layer
        
        if best_layer is None:
            print("no valid layers found, stopping greedy search")
            break

        best_layers.append(best_layer)
        best_probe = probe

    return best_probe, best_layers, best_val_acc

if layers is None:
    layers = range(model_config.num_hidden_layers)
else:
    layers = layers

if reg_strength is None:
    reg_strength = [None]
else:
    reg_strength = reg_strength

best_probe = None
best_val_acc = 0

os.makedirs(os.path.join('/'.join(model_path.split('/')[:-1]), 'probes'), exist_ok=True)
os.makedirs(os.path.join('/'.join(model_path.split('/')[:-1]), 'probes', model_name), exist_ok=True)

# For finetune, we don't need to pre-extract features
# For other classifiers, we now collect features per-layer to save memory

if classifier_type == 'multi_linear':
    # For multi_linear, we need all layer features - collect them one at a time
    print("collecting features for multi_linear (layer by layer)...")
    train_features = {}
    train_labels = {}
    val_features = {}
    val_labels = {}
    
    for layer in layers:
        print(f"  collecting layer {layer}...")
        train_features[layer], train_labels[layer] = collect_features_and_labels_for_layer('train', n_batches, layer)
        val_features[layer], val_labels[layer] = collect_features_and_labels_for_layer('val', n_test_batches, layer)
        gc.collect()

    best_probe, best_layers, best_val_acc = greedy_search(train_features, train_labels, val_features, val_labels, layers, reg_strength[0], max_iter, multi_layers)
    
    # Clean up features
    del train_features, train_labels, val_features, val_labels
    gc.collect()

    output = {
        'probe': best_probe,
        'layers': best_layers,
        'classifier_type': classifier_type,
    }

    print(f'best model: layers {best_layers} val acc {best_val_acc:.3f}')

    # Use both probe_type and label_type in filename
    filename = f'{model_name}-multi-{probe_type}-{label_type}.pkl'
    with open(os.path.join('/'.join(model_path.split('/')[:-1]), f'probes/{filename}'), 'wb') as f:
        pickle.dump(output, f)

elif classifier_type == 'finetune':
    # For finetune, we don't iterate over layers, just train the full model once
    for reg in reg_strength:
        print(f"finetuning model with reg strength {reg}")
        
        run_name = f"{model_name}-finetune-r{reg}-b{n_batches}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                'model': model_name,
                'n_batches': n_batches,
                'reg_strength': reg,
                'max_iter': max_iter,
                'batch_size': batch_size,
                'block_size': block_size,
                'model_num_layers': model_config.num_hidden_layers,
                'model_hidden_size': model_config.hidden_size,
                'classifier_type': classifier_type,
            }
        )

        # For finetune, we pass empty features/labels since training happens inside train_probe
        probe, val_acc = train_probe({'train': None, 'val': None}, {'train': None, 'val': None}, 0, reg, max_iter, model)
        wandb.finish()
        
        output = {
            'probe': probe,
            'classifier_type': classifier_type,
        }

        # Use both probe_type and label_type in filename
        filename = f'{model_name}/finetune-{probe_type}-{label_type}.pkl'
        with open(os.path.join('/'.join(model_path.split('/')[:-1]), f'probes/{filename}'), 'wb') as f:
            pickle.dump(output, f)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_probe = probe

            print(f'best model: finetune reg {reg} val acc {val_acc:.3f}')

            best_filename = f'{model_name}-{probe_type}-{label_type}.pkl'
            with open(os.path.join('/'.join(model_path.split('/')[:-1]), f'probes/{best_filename}'), 'wb') as f:
                pickle.dump(output, f)

elif classifier_type != 'finetune':
    for layer in layers:

        # Collect features for this layer only (memory efficient)
        print(f"collecting training features for layer {layer}...")
        train_feat, train_lab = collect_features_and_labels_for_layer('train', n_batches, layer)
        print(f"collecting validation features for layer {layer}...")
        val_feat, val_lab = collect_features_and_labels_for_layer('val', n_test_batches, layer)
        
        features = {'train': train_feat, 'val': val_feat}
        labels = {'train': train_lab, 'val': val_lab}
        
        if len(features['train']) == 0 or len(features['val']) == 0:
            print("no valid features found, skipping this layer")
            del train_feat, train_lab, val_feat, val_lab, features, labels
            gc.collect()
            continue

        for reg in reg_strength:
            print(f"training probe for layer {layer}/{model_config.num_hidden_layers} with reg strength {reg}")
            
            classifier_suffix = f"-{classifier_type}" + (f"-{mlp_layers}layers" if classifier_type == 'mlp' else "")
            run_name = f"{model_name}-l{layer}-r{reg}-b{n_batches}{classifier_suffix}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    'layer': layer,
                    'model': model_name,
                    'n_batches': n_batches,
                    'reg_strength': reg,
                    'max_iter': max_iter,
                    'batch_size': batch_size,
                    'block_size': block_size,
                    'model_num_layers': model_config.num_hidden_layers,
                    'model_hidden_size': model_config.hidden_size,
                    'classifier_type': classifier_type,
                    'mlp_layers': mlp_layers if classifier_type == 'mlp' else None,
                }
            )

            probe, val_acc = train_probe(features, labels, layer, reg, max_iter, None)
            wandb.finish()
            output = {
                'layer': layer,
                'probe': probe,
                'classifier_type': classifier_type,
                'mlp_layers': mlp_layers if classifier_type == 'mlp' else None,
            }

            # Use both probe_type and label_type in filename
            filename = f'{model_name}/layer-{layer}-{probe_type}-{label_type}.pkl'
            with open(os.path.join('/'.join(model_path.split('/')[:-1]), f'probes/{filename}'), 'wb') as f:
                pickle.dump(output, f)
            
            if val_acc > best_val_acc:

                best_val_acc = val_acc
                best_probe = probe

                print(f'best model: layer {layer} reg {reg} val acc {val_acc:.3f}')

                best_filename = f'{probe_name}.pkl'
                with open(os.path.join('/'.join(model_path.split('/')[:-1]), f'probes/{best_filename}'), 'wb') as f:
                    pickle.dump(output, f)
            
            else:
                del probe
        
        # Clean up this layer's features before moving to next layer
        del train_feat, train_lab, val_feat, val_lab, features, labels
        gc.collect()
        
        wandb.finish()