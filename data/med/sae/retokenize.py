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

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../../../../../workspace/med/data/')
parser.add_argument('--input_tokenizer', type=str, default='google/gemma-2-9b')
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

tokens = np.memmap(os.path.join(args.data_path, 'probe', 'token', 'gemma_tokens.bin'), dtype=np.uint32, mode='r')
labels = np.memmap(os.path.join(args.data_path, 'probe', 'token', 'gemma_labels.bin'), dtype=bool, mode='r')
sources = np.memmap(os.path.join(args.data_path, 'probe', 'token', 'gemma_sources.bin'), dtype=np.uint32, mode='r')
doc_ids = np.memmap(os.path.join(args.data_path, 'probe', 'token', 'gemma_doc_ids.bin'), dtype=np.uint32, mode='r')

# split into documents by BOT token
def chunkify(tokens, labels, sources, doc_ids):

    splits = np.where(tokens == input_bot)[0]
    indices = np.concatenate([[1], splits])

    tokens = [tokens[indices[i]:indices[i+1]] for i in range(len(indices)-1) 
           if indices[i] < indices[i+1]]
    labels = [labels[indices[i]:indices[i+1]] for i in range(len(indices)-1) 
           if indices[i] < indices[i+1]]
    sources = [sources[indices[i]:indices[i+1]] for i in range(len(indices)-1) 
           if indices[i] < indices[i+1]]
    doc_ids = [doc_ids[indices[i]:indices[i+1]] for i in range(len(indices)-1) 
           if indices[i] < indices[i+1]]

    return tokens, labels, sources, doc_ids

def retokenize(input_tokens, input_labels, input_sources, input_doc_ids):

    text = input_tokenizer.decode(input_tokens)
    output_tokens = output_tokenizer.encode(text)
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
    
    output_sources = input_sources[0] * np.ones(len(output_tokens))
    output_doc_ids = input_doc_ids[0] * np.ones(len(output_tokens))

    return output_tokens, output_labels, output_sources, output_doc_ids

tokens, labels, sources, doc_ids = chunkify(tokens, labels, sources, doc_ids)
all_tokens = []
all_labels = []
all_sources = []
all_doc_ids = []

for i in tqdm(range(len(tokens)), desc='retokenizing'):
    output_tokens, output_labels, output_sources, output_doc_ids = retokenize(tokens[i][1:], labels[i][1:], sources[i][1:], doc_ids[i][1:])
    all_tokens.extend(output_tokens)
    all_labels.extend(output_labels)
    all_sources.extend(output_sources)
    all_doc_ids.extend(output_doc_ids)

all_tokens = np.array(all_tokens, dtype=np.uint32)
all_labels = np.array(all_labels, dtype=bool)
all_sources = np.array(all_sources, dtype=np.uint32)
all_doc_ids = np.array(all_doc_ids, dtype=np.uint32)

if args.output_tokenizer == 'gpt4':
    probe_folder = 'probe'
else:
    probe_folder = args.output_tokenizer.split('/')[-1]

os.makedirs(os.path.join(args.data_path, probe_folder, 'token'), exist_ok=True)
tokens_file = np.memmap(os.path.join(args.data_path, probe_folder, 'token', 'tokens.bin'), dtype=np.uint32, mode='w+', shape=(len(all_tokens),))
labels_file = np.memmap(os.path.join(args.data_path, probe_folder, 'token', 'labels.bin'), dtype=bool, mode='w+', shape=(len(all_labels),))
sources_file = np.memmap(os.path.join(args.data_path, probe_folder, 'token', 'sources.bin'), dtype=np.uint32, mode='w+', shape=(len(all_sources),))
doc_ids_file = np.memmap(os.path.join(args.data_path, probe_folder, 'token', 'doc_ids.bin'), dtype=np.uint32, mode='w+', shape=(len(all_doc_ids),))

tokens_file[:] = all_tokens
labels_file[:] = all_labels
sources_file[:] = all_sources
doc_ids_file[:] = all_doc_ids

tokens_file.flush()
labels_file.flush()
sources_file.flush()
doc_ids_file.flush()