"""
reads a numpy array of tokens and labels and re-tokenizes / labels them
"""

import tiktoken
from transformers import AutoTokenizer
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd

sys.path.append('..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=DATA_PATH)
parser.add_argument('--input_tokenizer', type=str, default='answerdotai/ModernBERT-large')
parser.add_argument('--output_tokenizer', type=str, default='gpt4')
args = parser.parse_args()

if args.input_tokenizer == 'gpt4':
    input_tokenizer = tiktoken.get_encoding("cl100k_base")
    input_bot = input_tokenizer.eot_token
else:
    input_tokenizer = AutoTokenizer.from_pretrained(args.input_tokenizer)
    input_bot = input_tokenizer.bos_token_id

if args.output_tokenizer == 'gpt4':
    output_tokenizer = tiktoken.get_encoding("cl100k_base")
    output_bot = output_tokenizer.eot_token
else:
    output_tokenizer = AutoTokenizer.from_pretrained(args.output_tokenizer)
    output_bot = output_tokenizer.bos_token_id

thresholds = pd.read_csv('../config/accuracy-sweep-thresholds.csv')
# threshold = list(thresholds[thresholds['model'] == args.input_tokenizer.split('/')[-1]][thresholds['dataset'] == 'test_tokens']['threshold'])[0]
threshold = list(thresholds[thresholds['model'] == 'ModernBERT-large']['threshold'])[0]

print(f'threshold: {threshold}')

tokens = np.memmap(os.path.join(args.data_path, 'filtered-bert', 'val_bert.bin'), dtype=np.uint32, mode='r')
labels = np.memmap(os.path.join(args.data_path, 'filtered-bert', 'val_filter_bert.bin'), dtype=np.float16, mode='r')
doc_lens = np.memmap(os.path.join(args.data_path, 'filtered-bert', 'val_lens.bin'), dtype=np.uint32, mode='r')

# split into documents by BOT token
def chunkify(tokens, labels, doc_lens):

    # splits = np.where(tokens == input_bot)[0]
    # indices = np.concatenate([[1], splits])

    indices = np.cumsum(doc_lens)
    assert indices[-1] == len(tokens)
    tokens = [tokens[indices[i]:indices[i+1]] for i in range(len(indices)-1) 
           if indices[i] < indices[i+1]]
    labels = [labels[indices[i]:indices[i+1]] for i in range(len(indices)-1) 
           if indices[i] < indices[i+1]]
    labels = [[label > threshold for label in doc] for doc in labels]

    return tokens, labels

def retokenize(input_tokens, input_labels):

    text = input_tokenizer.decode(input_tokens)
    output_tokens = output_tokenizer.encode(text, disallowed_special=())
    output_labels = [0] * len(output_tokens)

    # find character spans for input tokens labeled 1
    labeled_spans = []
    current_pos = 0
    
    for i, token in enumerate(input_tokens):
        token_text = input_tokenizer.decode([token])
        token_start = text.find(token_text, current_pos)
        if token_start != -1:
            token_end = token_start + len(token_text)
            if input_labels[i] == 1:
                labeled_spans.append((token_start, token_end))
            current_pos = token_end
    
    # check overlap for each output token
    current_pos = 0
    for i, token in enumerate(output_tokens):
        token_text = output_tokenizer.decode([token])
        token_start = text.find(token_text, current_pos)
        if token_start != -1:
            token_end = token_start + len(token_text)
            
            # check if this output token overlaps with any labeled input token
            for span_start, span_end in labeled_spans:
                if not (token_end <= span_start or token_start >= span_end):
                    output_labels[i] = 1
                    break
                    
            current_pos = token_end

    return output_tokens, output_labels

tokens, labels = chunkify(tokens, labels, doc_lens)
print(f'retokenizing {len(tokens)}, {len(labels)} documents')
assert len(tokens) == len(labels)
all_tokens = []
all_labels = []

for i in tqdm(range(len(tokens)), desc='retokenizing'):
    output_tokens, output_labels = retokenize(tokens[i][1:], labels[i][1:])
    all_tokens.extend(output_tokens + [output_bot]) # add eot token between docs
    all_labels.extend(output_labels + [False])

all_tokens = np.array(all_tokens, dtype=np.uint32)
all_labels = np.array(all_labels, dtype=np.float16)

print(f'saving to {os.path.join(args.data_path, "filtered-bert", "val_retokenized.bin")}')
tokens_file = np.memmap(os.path.join(args.data_path, "filtered-bert", "val_retokenized.bin"), dtype=np.uint32, mode='w+', shape=(len(all_tokens),))
labels_file = np.memmap(os.path.join(args.data_path, "filtered-bert", "val_filter_retokenized.bin"), dtype=np.float16, mode='w+', shape=(len(all_labels),))

tokens_file[:] = all_tokens
labels_file[:] = all_labels

tokens_file.flush()
labels_file.flush()