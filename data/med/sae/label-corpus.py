"""
creates a set of documents for training a probe on medical content
   - pull from DCLM and from pre-prepared data mixture
   - pass through gemma SAE
   - if feature is above threshold on >= min_features features, add to + set
   - if feature is below threshold on < min_features features, add to - set
   - repeat until 5k DCLM documents + 15K other documents in each set
   - save to file
"""

import argparse
import torch
import os
import sys
import json
import random
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
import tiktoken
import numpy as np
from tqdm import tqdm

sys.path.append('../../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=os.path.join(DATA_PATH, 'probe', 'token'))
parser.add_argument("--docs_file", type=str, default=os.path.join(DATA_PATH, 'docs', 'sentences.jsonl'))
parser.add_argument("--features_file", type=str, default='features.jsonl')
parser.add_argument("--min_features", type=float, default=2.0)
parser.add_argument("--threshold_bin", type=int, default=-1)
parser.add_argument("--window_threshold_bin", type=int, default=0)
parser.add_argument("--min_score", type=float, default=0.9)
parser.add_argument("--target_docs", type=int, default=1400)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--window", action='store_true')
parser.add_argument("--weighted", action='store_true')
parser.add_argument("--cross_check", action='store_true')
parser.add_argument("--test_docs", type=int, default=5000, help="number of fineweb-edu docs to annotate for test set")
args = parser.parse_args()

sae_layer = 31
model     = AutoModelForCausalLM.from_pretrained('google/gemma-2-9b')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b')
tiktokenizer = tiktoken.get_encoding("cl100k_base")

model.to(args.device)
sae = SAE.from_pretrained(
    release='gemma-scope-9b-pt-res-canonical',
    sae_id=f'layer_{sae_layer}/width_16k/canonical',
    device=args.device
)

class Classifier:
    def __init__(self, features_file = args.features_file):

        """initialize the classifier with safety-tooling API."""
        features = pd.read_json('features.jsonl', orient='records', lines=True)
        # filter features by minimum score
        features = features[features['score'] >= args.min_score]
        
        # Store both thresholds and scores for weighted algorithm
        features_thresholds = features.set_index('idx')['thresholds'].to_dict()
        features_scores = features.set_index('idx')['score'].to_dict()

        max_feature_idx = max(features_thresholds.keys()) if features_thresholds else 0 # we only need the medical ones!
        self.thresholds = torch.full((max_feature_idx + 1,), float('-inf')).to(args.device)
        self.window_thresholds = torch.full((max_feature_idx + 1,), float('-inf')).to(args.device)
        self.scores = torch.zeros((max_feature_idx + 1,)).to(args.device)
        
        for idx, threshold in features_thresholds.items():
            self.thresholds[idx] = threshold[args.threshold_bin]
            self.window_thresholds[idx] = 0
            self.scores[idx] = features_scores[idx]
        
        self.feature_mask = self.thresholds > float('-inf') # convert to boolean mask
    
    def window_batch(self, medical, feature_counts, special_tokens_mask, window=True):
        """
        Apply windowing to expand medical token annotations
        
        Arguments:
            medical: tensor of initial medical annotations, shape [batch_size, seq_len]
            feature_counts: tensor of feature counts per token, shape [batch_size, seq_len]
            special_tokens_mask: tensor mask for non-special tokens, shape [batch_size, seq_len]
            window: bool, whether to apply windowing (default True)
        
        Returns:
            tensor of windowed annotations, shape [batch_size, seq_len]
        """
        if not window:
            return medical
            
        batch_size, seq_len = medical.shape
        windowed_medical = medical.clone()
        
        for batch_idx in range(batch_size):
            # Get the current sequence
            seq_medical = windowed_medical[batch_idx].clone()
            seq_feature_counts = feature_counts[batch_idx]
            seq_special_mask = special_tokens_mask[batch_idx]
            
            # Expand window until no tokens are marked
            changed = True
            while changed:
                changed = False
                
                for i in range(seq_len):
                    # Skip if already marked or if it's a special token
                    if seq_medical[i] or not seq_special_mask[i]:
                        continue
                    
                    # Check if token has >= 0 features active (with threshold 0, meaning any feature active)
                    if seq_feature_counts[i] > 0:
                        # Check if neighboring an annotated token
                        left_neighbor = seq_medical[i-1] if i > 0 else False
                        right_neighbor = seq_medical[i+1] if i < seq_len - 1 else False
                        
                        if left_neighbor or right_neighbor:
                            seq_medical[i] = True
                            changed = True
            
            # Update the batch tensor
            windowed_medical[batch_idx] = seq_medical
        
        return windowed_medical
    
    def process_batch(self, batch, input_ids, window=True):
        """
        process a batch of documents with optional windowing
        
        Arguments:
            batch: tensor of SAE features, shape [batch_size, seq_len,n_sae_features]
            input_ids: tensor of input token ids, shape [batch_size, seq_len]
            window: bool, whether to apply windowing (default True)
        
        Returns:
            tensor of annotations, shape [batch_size, seq_len]
        """        
        # we only need to broadcast along the relevant medical features
        relevant_features = batch[:, :, :self.thresholds.shape[0]]
        relevant_thresholds = self.thresholds[:relevant_features.shape[2]]
        relevant_window_thresholds = self.window_thresholds[:relevant_features.shape[2]]
        relevant_scores = self.scores[:relevant_features.shape[2]]

        # apply feature mask to only consider features with valid thresholds
        exceeds_threshold = relevant_features > relevant_thresholds
        valid_exceeds = exceeds_threshold & self.feature_mask[:relevant_features.shape[2]]

        # count features that exceed threshold
        if args.weighted:
            weighted_features = valid_exceeds.float() * relevant_scores.unsqueeze(0).unsqueeze(0)
            feature_counts = weighted_features.sum(dim=2)  # [batch_size, seq_len]
        else:
            feature_counts = valid_exceeds.sum(dim=2).int()
        
        # count features that exceed zero (baseline)
        exceeds_window = relevant_features > relevant_window_thresholds
        valid_exceeds_window = exceeds_window & self.feature_mask[:relevant_features.shape[2]]
        if args.weighted:
            weighted_baseline_features = valid_exceeds_window.float() * relevant_scores.unsqueeze(0).unsqueeze(0)
            baseline_feature_counts = weighted_baseline_features.sum(dim=2)  # [batch_size, seq_len]
        else:
            baseline_feature_counts = valid_exceeds_window.sum(dim=2).int()
        
        # create mask to ignore <bos> and <eos> tokens in outputs
        special_tokens_mask = (input_ids != tokenizer.bos_token_id) & (input_ids != tokenizer.eos_token_id)
        feature_counts = feature_counts * special_tokens_mask.int()  # set <bos> and <eos> positions to 0
        baseline_feature_counts = baseline_feature_counts * special_tokens_mask.int()

        # Step 1: Initial medical token detection (>= min_features active)
        medical = (feature_counts >= args.min_features)
        
        # Step 2: Apply windowing if requested
        medical = self.window_batch(medical, baseline_feature_counts, special_tokens_mask, window)

        return medical

# data = pd.read_json('../gold.jsonl', orient='records', lines=True)
# data = data[data['source'].isin(['biorxiv', 'medrxiv', 'chemrxiv', 'arxiv_bio', 'pubmed', 'textbook'])]
# data['doc_id'] = data['text'].apply(lambda x: hash(x) % 2**16)
data = pd.read_json(args.docs_file, orient='records', lines=True)
# data['doc_id'] = data['text'].apply(lambda x: hash(x) % 2**16)
# print(data.head())
data = data.sort_values(['doc_id', 'sent_id']).groupby(['source', 'doc_id'], as_index=False).agg({
    'text': ' '.join
    # 'text': [' '.join, list],
    # 'medical': list
})
data.columns = ['source', 'doc_id', 'text'] #, 'sentence_list', 'medical_list']
data = data[['source', 'doc_id', 'text']] #, 'sentence_list', 'medical_list']]
data = Dataset.from_pandas(data)
data = data.shuffle(seed=42)
data = data.select(range(args.target_docs))

print(len(data))

# print(data['text'][0])
# print(data['sentence_list'][0])
# print(data['medical_list'][0])

# gold = pd.read_json('../gold.jsonl', orient='records', lines=True)
# gold = Dataset.from_pandas(gold)
# gold = gold.shuffle(seed=42)

batch_size = args.batch_size # reasonable batch size
classifier = Classifier(features_file = args.features_file)

total_processed = 0
offset = 0

tokens_file = os.path.join(args.data_path, f'gemma_tokens.bin')
labels_file = os.path.join(args.data_path, f'gemma_labels.bin')
sources_file = os.path.join(args.data_path, f'gemma_sources.bin')
doc_ids_file = os.path.join(args.data_path, f'gemma_doc_ids.bin')

os.makedirs(args.data_path, exist_ok=True)
with open(tokens_file, 'wb') as f:
    pass
with open(labels_file, 'wb') as f:
    pass
with open(sources_file, 'wb') as f:
    pass
with open(doc_ids_file, 'wb') as f:
    pass

source_to_idx = {
    'fineweb' : 0,
    'biorxiv' : 1,
    'medrxiv' : 2,
    'chemrxiv' : 3,
    'arxiv_bio' : 4,
    'arxiv_no_bio' : 5,
    'pubmed' : 6,
    'textbook' : 7,
    'stanford' : 8,
    'wmdp' : 9
}

pad_token_id = tokenizer.pad_token_id
def get_batch_mask(batch, seq_len=1024):

    mask = torch.zeros(len(batch), seq_len, dtype=torch.bool)

    for i, row in enumerate(batch):

        tokens = tokenizer(row['sentence_list'], max_length=seq_len, truncation=True, padding=True)
        lengths = [sum(l) for l in tokens['attention_mask']]
        r = np.repeat(row['medical_list'], lengths)[:seq_len]
        mask[i, -len(r):] = torch.from_numpy(r)
    
    return mask.bool().to(model.device)


# Process fineweb-edu test data
print(f"\nProcessing {args.test_docs} fineweb-edu documents for test set...")

# Load fineweb-edu dataset similar to prepare-bidir-corpus.py
workspace_base = os.path.dirname(os.path.dirname(DATA_PATH))
fineweb_dataset = load_dataset(
    'HuggingFaceFW/fineweb-edu', 
    name='sample-100BT', 
    cache_dir=os.path.join(workspace_base, 'fineweb-100BT'), 
    download_mode='reuse_dataset_if_exists'
)

# Sample random documents
rng = np.random.RandomState(42)
indices = rng.choice(len(fineweb_dataset['train']), args.test_docs, replace=False)
test_data = fineweb_dataset['train'].select(indices)

# Create test file paths
test_tokens_file = os.path.join(args.data_path, 'gemma_test_tokens.bin')
test_labels_file = os.path.join(args.data_path, 'gemma_test_labels.bin')

# Initialize test files
with open(test_tokens_file, 'wb') as f:
    pass
with open(test_labels_file, 'wb') as f:
    pass

# Process test data in batches
for i in tqdm(range(0, args.test_docs, batch_size), desc="Processing test data"):
    batch_end = min(i + batch_size, args.test_docs)
    batch_texts = [test_data[j]['text'] for j in range(i, batch_end)]
    
    inputs = tokenizer(
        batch_texts,
        return_tensors='pt',
        max_length=1024,
        padding=True,
        truncation=True
    ).to(model.device)

    test_tokens = []
    test_labels = []
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        outputs = outputs.hidden_states[sae_layer] # [batch_size, seq_len, n_embd]
        
        sae_output = sae.encode(outputs) # [batch_size, seq_len, n_sae_features]
        if args.window:
            annotations = classifier.process_batch(sae_output, inputs.input_ids, window=True) # [batch_size, seq_len]
        else:
            annotations = classifier.process_batch(sae_output, inputs.input_ids, window=False) # [batch_size, seq_len]

        attention_mask = inputs['attention_mask'] # [batch_size, seq_len]
        
        flattened_tokens = inputs.input_ids[attention_mask.bool()]
        flattened_annotations = annotations[attention_mask.bool()]
        
        test_tokens.append(flattened_tokens.detach().cpu().numpy().astype(np.uint32))
        test_labels.append(flattened_annotations.detach().cpu().numpy().astype(bool))

    # Append to test files
    with open(test_tokens_file, 'ab') as f:
        f.write(np.concatenate(test_tokens).tobytes())

    with open(test_labels_file, 'ab') as f:
        f.write(np.concatenate(test_labels).tobytes())

sys.exit()

for i in tqdm(range(0, args.target_docs, batch_size)):
    batch = data.skip(i).take(batch_size)
    inputs = tokenizer(
        list(batch['text']),
        return_tensors='pt',
        max_length=1024,
        padding=True,
        truncation=True
    ).to(model.device)

    if args.cross_check:
        cross_check_mask = get_batch_mask(batch, seq_len=inputs.input_ids.shape[1])
        
    all_tokens = []
    all_labels = []
    all_sources = []
    all_doc_ids = []
    with torch.no_grad():

        outputs = model(**inputs, output_hidden_states=True)
        outputs = outputs.hidden_states[sae_layer] # [batch_size, seq_len, n_embd]
        
        sae_output = sae.encode(outputs) # [batch_size, seq_len, n_sae_features]
        if args.window:
            annotations = classifier.process_batch(sae_output, inputs.input_ids, window=True) # [batch_size, seq_len]
        else:
            annotations = classifier.process_batch(sae_output, inputs.input_ids, window=False) # [batch_size, seq_len]

        if args.cross_check:
            annotations = annotations * cross_check_mask

        attention_mask = inputs['attention_mask'] # [batch_size, seq_len]
        
        sources = torch.tensor([source_to_idx[s] for s in batch['source']]).unsqueeze(1).expand(-1, attention_mask.shape[1]).to(args.device) # [batch_size, seq_len]
        doc_ids = torch.tensor(batch['doc_id']).unsqueeze(1).expand(-1, attention_mask.shape[1]).to(args.device) # [batch_size, seq_len]

        flattened_tokens = inputs.input_ids[attention_mask.bool()]
        flattened_annotations = annotations[attention_mask.bool()]
        flattened_sources = sources[attention_mask.bool()]
        flattened_doc_ids = doc_ids[attention_mask.bool()]
        
        all_tokens.append(flattened_tokens.detach().cpu().numpy().astype(np.uint32))
        all_labels.append(flattened_annotations.detach().cpu().numpy().astype(bool))
        all_sources.append(flattened_sources.detach().cpu().numpy().astype(np.uint32))
        all_doc_ids.append(flattened_doc_ids.detach().cpu().numpy().astype(np.uint32))

    with open(tokens_file, 'ab') as f:
        f.write(np.concatenate(all_tokens).tobytes())

    with open(labels_file, 'ab') as f:
        f.write(np.concatenate(all_labels).tobytes())
    
    with open(sources_file, 'ab') as f:
        f.write(np.concatenate(all_sources).tobytes())

    with open(doc_ids_file, 'ab') as f:
        f.write(np.concatenate(all_doc_ids).tobytes())