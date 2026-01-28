"""
generate proper test sets for model training
"""

import argparse
import os
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import sys

sys.path.append('../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='path to save dataset to')
parser.add_argument('--skip_documents', type=int, default=1000) # number of docs skipped per dset (used for probe training)
parser.add_argument('--num_documents', type=int, default=5000) # TOTAL number of docs for test sets (not per dset!!)
args = parser.parse_args()

num_proc = 8
enc = tiktoken.get_encoding("cl100k_base")
dtype = np.uint32

def process(example):
    
    ids = enc.encode_ordinary(example['text'])

    # truncate documents longer than 2^12 - 1 tokens
    length = len(ids)
    max_length = 2**12 - 2 # extra -1 for eot token

    if length > max_length:
        ids = ids[:max_length]
        length = max_length
    
    ids.append(enc.eot_token)
    length += 1
    
    return {'ids' : ids, 'len': length}

def save_to_file(dataset, filename, dtype=np.uint32):

    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing splits",
        num_proc=num_proc
    )

    ids = np.concatenate(tokenized['ids'])

    print(f'total: {len(ids)} tokens')
    
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(len(ids),))
    arr[:] = ids
    arr.flush()

df = pd.read_json(os.path.join(args.data_path, 'docs', f'documents.jsonl'), lines=True)

ds = Dataset.from_pandas(df[df['medical'] == False])
save_to_file(ds, os.path.join(args.data_path, 'filtered', f'target.bin'), dtype=np.uint32)

ds = Dataset.from_pandas(df[df['medical'] == True])
save_to_file(ds, os.path.join(args.data_path, 'filtered', f'ood.bin'), dtype=np.uint32)

sys.exit()

# four datasets: test_target,test_ood, test_parallel, test_parallel_hard
# test target: arxiv, philpapers, stanford encyclopedia of philosophy, 
datasets = []
num_documents = args.num_documents // 64

ds = load_dataset('neuralwork/arxiver', split='train')
ds = ds.shuffle(seed=42).select(range(34 * num_documents))
ds = ds.map(lambda x: {'text' : x['abstract']})
datasets.append(ds)

ds = load_dataset('AiresPucrs/stanford-encyclopedia-philosophy', split='train')
ds = ds.shuffle(seed=42).select(range(24 * num_documents))
datasets.append(ds)

ds = load_dataset('timaeus/pile-philpapers', split='train')
ds = ds.shuffle(seed=42).select(range(5 * num_documents)) # these are pretty long
datasets.append(ds)

ds = load_dataset('manu/project_gutenberg', split='en')
ds = ds.shuffle(seed=42).select(range(1 * num_documents)) # these are even longer
datasets.append(ds)

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)
save_to_file(dataset, os.path.join(args.data_path, 'filtered', f'test_target_true.bin'), dtype=np.uint32)

# test ood: pubmed, med textbooks
datasets = []
num_documents = args.num_documents // 2

ds = load_dataset('MedRAG/pubmed', split='train', streaming=True)
ds = Dataset.from_list(list(ds.skip(args.skip_documents).take(args.num_documents))) # we use the first 1k for probe training
ds = ds.map(lambda x: {'text' : x['content']})
datasets.append(ds)

ds = load_dataset('cogbuji/medqa_corpus_en', 'core_clinical', trust_remote_code=True)
ds = ds.shuffle(seed=42) # note random seed is the same as in get_additional.py
ds = Dataset.from_list(list(ds['train'].skip(args.skip_documents).take(args.num_documents)))
datasets.append(ds)

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)
save_to_file(dataset, os.path.join(args.data_path, 'filtered', f'test_ood_true.bin'), dtype=np.uint32)

# test parallel: biorxiv ecology, stackexchange bio
datasets = []
num_documents = args.num_documents // 6

ds = load_dataset('mteb/raw_biorxiv', split='train', streaming=True)
ds = Dataset.from_list(list(ds.skip(args.skip_documents))) # skip, then filter, then select
ds = ds.filter(lambda x: x['category'] in ['animal behavior and cognition', 'ecology', 'evolutionary biology', 'paleontology', 'plant biology', 'zoology'])
ds = ds.select(range(5 * num_documents))
ds = ds.map(lambda x: {'text' : x['abstract']})
datasets.append(ds)

ds = load_dataset('donfu/oa-stackexchange', split='train', streaming=True)
ds = Dataset.from_list(list(ds.skip(29327 + args.skip_documents).take(num_documents))) # biology starts at 29327
ds = ds.map(lambda x: {'text': x['RESPONSE']})
datasets.append(ds)

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)
save_to_file(dataset, os.path.join(args.data_path, 'filtered', f'test_parallel.bin'), dtype=np.uint32)

# test parallel hard: biorxiv cell bio, bio textbooks (cell bio, histology)
datasets = []
num_documents = args.num_documents // 3

ds = load_dataset('mteb/raw_biorxiv', split='train', streaming=True)
ds = Dataset.from_list(list(ds.skip(args.skip_documents)))
ds = ds.filter(lambda x: x['category'] in ['biochemistry', 'biophysics', 'cell biology', 'microbiology', 'molecular biology'])
ds = ds.select(range(2 * num_documents))
ds = ds.map(lambda x: {'text' : x['abstract']})
datasets.append(ds)

ds = load_dataset('cogbuji/medqa_corpus_en', 'basic_biology', trust_remote_code=True)
ds = ds.shuffle(seed=42) # again, random seed is the same as in get_additional.py
ds = ds.filter(lambda x: x['source'] in ['textbooks/en/Cell_Biology_Alberts.txt', 'textbooks/en/Histology_Ross.txt'])
ds = Dataset.from_list(list(ds['train'].skip(args.skip_documents).take(num_documents)))
datasets.append(ds)

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)
save_to_file(dataset, os.path.join(args.data_path, 'filtered', f'test_parallel_hard.bin'), dtype=np.uint32)