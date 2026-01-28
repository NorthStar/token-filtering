"""
generate test sets as jsonl files with text and source fields
"""

import argparse
import os
import sys
import json
from datasets import load_dataset, Dataset, concatenate_datasets

sys.path.append('../../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='path to save dataset to')
parser.add_argument('--num_documents', type=int, default=5000)  # TOTAL number of docs for test sets (not per dset!!)
args = parser.parse_args()

output_dir = os.path.join(args.data_path, 'test')
os.makedirs(output_dir, exist_ok=True)

def save_to_jsonl(dataset, filename):
    """save dataset to jsonl with text and source fields"""
    with open(filename, 'w', encoding='utf-8') as f:
        for row in dataset:
            line = {'text': row['text'], 'source': row.get('source', 'unknown')}
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f"saved {len(dataset)} documents to {filename}")

# test target: arxiv, gutenberg, philpapers
datasets = []

ds = load_dataset('common-pile/arxiv_abstracts_filtered', split='train')
ds = ds.shuffle(seed=42).select(range(13 * args.num_documents // 8)).select_columns(['text'])
datasets.append(ds)

ds = load_dataset('timaeus/pile-philpapers', split='train')
ds = ds.shuffle(seed=42).select(range(1 * args.num_documents // 4)).select_columns(['text'])
datasets.append(ds)

ds = load_dataset('common-pile/project_gutenberg_filtered', split='train')
ds = ds.shuffle(seed=42).select(range(args.num_documents // 8)).select_columns(['text']) # these are pretty long
datasets.append(ds)

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)
save_to_jsonl(dataset, os.path.join(output_dir, 'test_target.jsonl'))

# test_ood: pubmed
datasets = []

ds = load_dataset('timaeus/pile-pubmed_abstracts', split='train')
ds = ds.shuffle(seed=42).select(range(args.num_documents)).select_columns(['text'])
datasets.append(ds)

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)
save_to_jsonl(dataset, os.path.join(output_dir, 'test_ood.jsonl'))

# test_parallel: biorxiv ecology
datasets = []
bio_categories = ['animal behavior and cognition', 'ecology', 'evolutionary biology', 'paleontology', 'plant biology', 'zoology']

ds = load_dataset('mteb/raw_biorxiv', split='train')
ds = ds.shuffle(seed=42)
ds = ds.filter(lambda x: x['category'] in bio_categories)
ds = ds.select(range(min(args.num_documents * 5, len(ds))))
ds = ds.map(lambda x: {'text' : x['abstract']}, remove_columns=ds.column_names)
datasets.append(ds)

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)
save_to_jsonl(dataset, os.path.join(output_dir, 'test_parallel.jsonl'))

# test_parallel_hard: biorxiv biochem
datasets = []
biochem_categories = ['biochemistry', 'biophysics', 'cell biology', 'microbiology', 'molecular biology']

ds = load_dataset('mteb/raw_biorxiv', split='train')
ds = ds.shuffle(seed=42)
ds = ds.filter(lambda x: x['category'] in biochem_categories)
ds = ds.select(range(min(args.num_documents * 5, len(ds))))
ds = ds.map(lambda x: {'text' : x['abstract']}, remove_columns=ds.column_names)
datasets.append(ds)

dataset = concatenate_datasets(datasets)
dataset = dataset.shuffle(seed=42)
save_to_jsonl(dataset, os.path.join(output_dir, 'test_parallel_hard.jsonl'))

print("done generating test sets")

