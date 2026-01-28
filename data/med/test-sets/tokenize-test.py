"""
tokenize filtered test set jsonl files and save as bin files
"""

import argparse
import os
import sys
import json
import numpy as np
import tiktoken
from datasets import Dataset

sys.path.append('../../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, required=True, help='name of the filtered jsonl file (without _filtered.jsonl extension)')
parser.add_argument('--data_path', type=str, default=os.path.join(DATA_PATH, 'test'))
args = parser.parse_args()

num_proc = 8
enc = tiktoken.get_encoding("cl100k_base")
dtype = np.uint32


def process(example):
    ids = enc.encode_ordinary(example['text'])
    
    # truncate documents longer than 2^12 - 1 tokens
    length = len(ids)
    max_length = 2**12 - 2  # extra -1 for eot token
    
    if length > max_length:
        ids = ids[:max_length]
        length = max_length
    
    ids.append(enc.eot_token)
    length += 1
    
    return {'ids': ids, 'len': length}


def save_to_file(dataset, filename, dtype=np.uint32):
    tokenized = dataset.map(
        process,
        remove_columns=['text', 'source'],
        desc="tokenizing splits",
        num_proc=num_proc
    )
    
    ids = np.concatenate(tokenized['ids'])
    
    print(f'total: {len(ids)} tokens')
    
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(len(ids),))
    arr[:] = ids
    arr.flush()


# load filtered jsonl
input_file = os.path.join(args.data_path, f"{args.split}_filtered.jsonl")

if not os.path.exists(input_file):
    print(f"error: input file {input_file} does not exist")
    sys.exit(1)

documents = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        documents.append(json.loads(line))

print(f"loaded {len(documents)} documents from {input_file}")

dataset = Dataset.from_list(documents)
output_file = os.path.join(args.data_path, f"{args.split}.bin")
save_to_file(dataset, output_file, dtype=dtype)

print(f"saved tokenized data to {output_file}")
print("done")

