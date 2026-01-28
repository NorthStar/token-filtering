"""
Label test set tokens using SAE-based classification
- Reads tokenized .bin files from tokenize-test.py (tiktoken)
- Decodes to text, re-encodes with Gemma tokenizer
- Passes through Gemma SAE for labeling
- Re-tokenizes labels back to tiktoken space
- Saves labels as .bin file
"""

import argparse
import torch
import os
import sys
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
import tiktoken
from tqdm import tqdm

sys.path.append('../../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, required=True, help='name of the test set (matches tokenize-test output)')
parser.add_argument('--data_path', type=str, default=os.path.join(DATA_PATH, 'test'))
parser.add_argument('--features_file', type=str, default='../sae/features.jsonl')
parser.add_argument('--min_features', type=float, default=2.0)
parser.add_argument('--threshold_bin', type=int, default=-1)
parser.add_argument('--min_score', type=float, default=0.9)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--window', action='store_true')
parser.add_argument('--weighted', action='store_true')
args = parser.parse_args()

# Initialize tokenizers
tiktoken_enc = tiktoken.get_encoding("cl100k_base")
gemma_tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b')

# Initialize model and SAE
sae_layer = 31
model = AutoModelForCausalLM.from_pretrained('google/gemma-2-9b')
model.to(args.device)
sae = SAE.from_pretrained(
    release='gemma-scope-9b-pt-res-canonical',
    sae_id=f'layer_{sae_layer}/width_16k/canonical',
    device=args.device
)


class Classifier:
    def __init__(self, features_file=args.features_file):
        """Initialize the classifier with SAE features."""
        features = pd.read_json(features_file, orient='records', lines=True)
        features = features[features['score'] >= args.min_score]
        
        features_thresholds = features.set_index('idx')['thresholds'].to_dict()
        features_scores = features.set_index('idx')['score'].to_dict()

        max_feature_idx = max(features_thresholds.keys()) if features_thresholds else 0
        self.thresholds = torch.full((max_feature_idx + 1,), float('-inf')).to(args.device)
        self.window_thresholds = torch.full((max_feature_idx + 1,), float('-inf')).to(args.device)
        self.scores = torch.zeros((max_feature_idx + 1,)).to(args.device)
        
        for idx, threshold in features_thresholds.items():
            self.thresholds[idx] = threshold[args.threshold_bin]
            self.window_thresholds[idx] = 0
            self.scores[idx] = features_scores[idx]
        
        self.feature_mask = self.thresholds > float('-inf')
    
    def window_batch(self, medical, feature_counts, special_tokens_mask, window=True):
        """Apply windowing to expand medical token annotations."""
        if not window:
            return medical
            
        batch_size, seq_len = medical.shape
        windowed_medical = medical.clone()
        
        for batch_idx in range(batch_size):
            seq_medical = windowed_medical[batch_idx].clone()
            seq_feature_counts = feature_counts[batch_idx]
            seq_special_mask = special_tokens_mask[batch_idx]
            
            changed = True
            while changed:
                changed = False
                
                for i in range(seq_len):
                    if seq_medical[i] or not seq_special_mask[i]:
                        continue
                    
                    if seq_feature_counts[i] > 0:
                        left_neighbor = seq_medical[i-1] if i > 0 else False
                        right_neighbor = seq_medical[i+1] if i < seq_len - 1 else False
                        
                        if left_neighbor or right_neighbor:
                            seq_medical[i] = True
                            changed = True
            
            windowed_medical[batch_idx] = seq_medical
        
        return windowed_medical
    
    def process_batch(self, batch, input_ids, window=True):
        """Process a batch of documents with optional windowing."""
        relevant_features = batch[:, :, :self.thresholds.shape[0]]
        relevant_thresholds = self.thresholds[:relevant_features.shape[2]]
        relevant_window_thresholds = self.window_thresholds[:relevant_features.shape[2]]
        relevant_scores = self.scores[:relevant_features.shape[2]]

        exceeds_threshold = relevant_features > relevant_thresholds
        valid_exceeds = exceeds_threshold & self.feature_mask[:relevant_features.shape[2]]

        if args.weighted:
            weighted_features = valid_exceeds.float() * relevant_scores.unsqueeze(0).unsqueeze(0)
            feature_counts = weighted_features.sum(dim=2)
        else:
            feature_counts = valid_exceeds.sum(dim=2).int()
        
        exceeds_window = relevant_features > relevant_window_thresholds
        valid_exceeds_window = exceeds_window & self.feature_mask[:relevant_features.shape[2]]
        if args.weighted:
            weighted_baseline_features = valid_exceeds_window.float() * relevant_scores.unsqueeze(0).unsqueeze(0)
            baseline_feature_counts = weighted_baseline_features.sum(dim=2)
        else:
            baseline_feature_counts = valid_exceeds_window.sum(dim=2).int()
        
        special_tokens_mask = (input_ids != gemma_tokenizer.bos_token_id) & (input_ids != gemma_tokenizer.eos_token_id)
        feature_counts = feature_counts * special_tokens_mask.int()
        baseline_feature_counts = baseline_feature_counts * special_tokens_mask.int()

        medical = (feature_counts >= args.min_features)
        medical = self.window_batch(medical, baseline_feature_counts, special_tokens_mask, window)

        return medical


def retokenize_labels(text, gemma_tokens, gemma_labels):
    """
    Convert labels from Gemma token space back to tiktoken space.
    
    Args:
        text: The original text
        gemma_tokens: List of Gemma token ids
        gemma_labels: List of labels (0/1) for each Gemma token
    
    Returns:
        tiktoken_labels: List of labels for tiktoken tokens
    """
    tiktoken_tokens = tiktoken_enc.encode(text)
    tiktoken_labels = [0] * len(tiktoken_tokens)
    
    # Find character spans for Gemma tokens labeled 1
    labeled_spans = []
    current_pos = 0
    
    for i, token in enumerate(gemma_tokens):
        token_text = gemma_tokenizer.decode([token])
        token_start = text.find(token_text, current_pos)
        if token_start != -1:
            token_end = token_start + len(token_text)
            if gemma_labels[i] == 1:
                labeled_spans.append((token_start, token_end))
            current_pos = token_end
    
    # Check overlap for each tiktoken token
    current_pos = 0
    for i, token in enumerate(tiktoken_tokens):
        token_text = tiktoken_enc.decode([token])
        token_start = text.find(token_text, current_pos)
        if token_start != -1:
            token_end = token_start + len(token_text)
            
            for span_start, span_end in labeled_spans:
                if not (token_end <= span_start or token_start >= span_end):
                    tiktoken_labels[i] = 1
                    break
                    
            current_pos = token_end
    
    return tiktoken_labels


# Load tokenized data
input_file = os.path.join(args.data_path, f"{args.split}.bin")
if not os.path.exists(input_file):
    print(f"Error: input file {input_file} does not exist")
    sys.exit(1)

tokens = np.memmap(input_file, dtype=np.uint32, mode='r')
print(f"Loaded {len(tokens)} tokens from {input_file}")

# Split into documents by EOT token
eot_token = tiktoken_enc.eot_token
splits = np.where(tokens == eot_token)[0]
doc_starts = np.concatenate([[0], splits + 1])
doc_ends = np.concatenate([splits + 1, [len(tokens)]])

documents = []
for start, end in zip(doc_starts, doc_ends):
    if start < end:
        doc_tokens = tokens[start:end].tolist()
        documents.append(doc_tokens)

print(f"Found {len(documents)} documents")

# Initialize classifier
classifier = Classifier(features_file=args.features_file)

# Process documents and collect labels
all_labels = []
batch_size = args.batch_size

for i in tqdm(range(0, len(documents), batch_size), desc="Processing documents"):
    batch_docs = documents[i:i + batch_size]
    
    # Decode tiktoken tokens to text
    batch_texts = []
    for doc_tokens in batch_docs:
        # Remove EOT token for decoding if present
        if doc_tokens and doc_tokens[-1] == eot_token:
            text = tiktoken_enc.decode(doc_tokens[:-1])
        else:
            text = tiktoken_enc.decode(doc_tokens)
        batch_texts.append(text)
    
    # Encode with Gemma tokenizer
    gemma_inputs = gemma_tokenizer(
        batch_texts,
        return_tensors='pt',
        max_length=1024,
        padding=True,
        truncation=True
    ).to(args.device)
    
    with torch.no_grad():
        outputs = model(**gemma_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[sae_layer]
        
        sae_output = sae.encode(hidden_states)
        annotations = classifier.process_batch(sae_output, gemma_inputs.input_ids, window=args.window)
        
        attention_mask = gemma_inputs['attention_mask']
    
    # Process each document in batch
    for j, (doc_tokens, text) in enumerate(zip(batch_docs, batch_texts)):
        # Get Gemma tokens and labels for this document (excluding padding)
        seq_len = attention_mask[j].sum().item()
        gemma_tokens = gemma_inputs.input_ids[j, :seq_len].cpu().tolist()
        gemma_labels = annotations[j, :seq_len].cpu().int().tolist()
        
        # Skip BOS token if present
        if gemma_tokens and gemma_tokens[0] == gemma_tokenizer.bos_token_id:
            gemma_tokens = gemma_tokens[1:]
            gemma_labels = gemma_labels[1:]
        
        # Retokenize labels to tiktoken space
        tiktoken_labels = retokenize_labels(text, gemma_tokens, gemma_labels)
        
        # Add label for EOT token (always 0)
        if doc_tokens and doc_tokens[-1] == eot_token:
            tiktoken_labels.append(0)
        
        # Verify length matches
        if len(tiktoken_labels) != len(doc_tokens):
            print(f"Warning: label length mismatch for doc {i+j}: {len(tiktoken_labels)} vs {len(doc_tokens)} tokens")
            # Pad or truncate to match
            if len(tiktoken_labels) < len(doc_tokens):
                tiktoken_labels.extend([0] * (len(doc_tokens) - len(tiktoken_labels)))
            else:
                tiktoken_labels = tiktoken_labels[:len(doc_tokens)]
        
        all_labels.extend(tiktoken_labels)

# Save labels
all_labels = np.array(all_labels, dtype=bool)
output_file = os.path.join(args.data_path, f"{args.split}_labels.bin")

labels_memmap = np.memmap(output_file, dtype=bool, mode='w+', shape=(len(all_labels),))
labels_memmap[:] = all_labels
labels_memmap.flush()

print(f"Saved {len(all_labels)} labels to {output_file}")
print(f"Positive labels: {all_labels.sum()} ({100 * all_labels.mean():.2f}%)")
print("Done")

