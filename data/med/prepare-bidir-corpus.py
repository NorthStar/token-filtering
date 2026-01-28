"""
prep data for roberta training
 - this has to be done outside of the main training script because HF runs into api limits w/ DDP
"""

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from itertools import islice
from tqdm import tqdm
import os
import numpy as np
import tiktoken
import pickle
import sys
from transformers import AutoTokenizer

import argparse

sys.path.append('../..')
from paths import DATA_PATH

# FineWeb: ~1k toks/doc, 61M param model @ 4x chinchilla -> 4.8B toks -> 4.8M docs
# PubMed + FineWeb: 5k + 1k toks/doc, 224M param model @ 4x chinchilla -> 17.9B toks -> 1.8M PM + 9M FW
parser = argparse.ArgumentParser()
parser.add_argument('--num_documents', type=int, default=11000000, help = 'number of docs for training')
parser.add_argument('--num_threads', type=int, default=10, help = 'number of threads to use')
parser.add_argument('--thread_idx', type=int, default=0, help = 'which thread are we on?')
parser.add_argument('--data_path', type=str, default=DATA_PATH, help = 'path to save data')
parser.add_argument('--tokenizer', type=str, default='gpt', help = 'tokenizer to use')
args = parser.parse_args()

num_proc = 8

if args.tokenizer == 'gpt':
    enc = tiktoken.get_encoding("cl100k_base")
else:
    enc = AutoTokenizer.from_pretrained(args.tokenizer)

workspace_base = os.path.dirname(os.path.dirname(DATA_PATH))

datasets = []
# dataset = load_dataset(f'HuggingFaceFW/fineweb-edu', name = 'sample-100BT', cache_dir=os.path.join(workspace_base, 'fineweb-100BT'), download_mode='reuse_dataset_if_exists')
# rng = np.random.RandomState(42)
# indices = rng.choice(len(dataset['train']), args.num_documents, replace=False)
# # datasets.append(dataset['train'].select(indices))
# dataset = dataset['train'].select(indices)

rng = np.random.RandomState(42)

print('loading pubmed data from huggingface...')
dataset = load_dataset('common-pile/pubmed_filtered', cache_dir=os.path.join(workspace_base, 'pubmed'), split='train', download_mode='reuse_dataset_if_exists')
indices = rng.choice(len(dataset), int(0.16 * args.num_documents), replace=False)
datasets.append(dataset.select(indices))

print('loading fineweb data...')
dataset = load_dataset(f'HuggingFaceFW/fineweb-edu', name = 'sample-100BT', cache_dir=os.path.join(workspace_base, 'fineweb-100BT'), download_mode='reuse_dataset_if_exists')
indices = rng.choice(len(dataset['train']), int(0.84 * args.num_documents), replace=False)
datasets.append(dataset['train'].select(indices))

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)

if args.num_threads > 1:
    output_dir = os.path.join(args.data_path, 'roberta-pubmed', f'{args.thread_idx}')
    os.makedirs(output_dir, exist_ok=True)
    dataset = dataset.shard(num_shards=args.num_threads, index=args.thread_idx)
else:
    output_dir = os.path.join(args.data_path, 'roberta-pubmed')
    os.makedirs(output_dir, exist_ok=True)

split_dataset = dataset.train_test_split(test_size=0.05, seed=2357, shuffle=True)

def process(example):
    if args.tokenizer == 'gpt':
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
    else:
        ids = enc.encode(example['text'], add_special_tokens=False)
        ids.append(enc.eos_token_id)
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc=f"tokenizing",
    num_proc=num_proc,
)

tokenized['train'] = tokenized['train'].shuffle(seed=42)
tokenized['test']  = tokenized['test'].shuffle(seed=42)

print('tokenized, now writing to files...')

# os.makedirs(os.path.join(args.data_path, 'roberta-edu'), exist_ok=True)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(output_dir, f'{split}.bin')
    dtype = np.uint32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    idx = 0
    write_batch_size = len(dset) // 10

    for batch_idx in tqdm(range(0, len(dset), write_batch_size), desc=f'writing {filename}'):
        batch = dset[batch_idx:min(batch_idx + write_batch_size, len(dset))]
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()