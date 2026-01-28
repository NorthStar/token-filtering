"""
prepare dataset for training a probe on medical content
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

import tiktoken
from transformers import AutoTokenizer

sys.path.append('../..')
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=DATA_PATH)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--tokenizer", type=str, default='gpt')
parser.add_argument("--type", type=str, default='sentence', choices=['sentence', 'document'])
parser.add_argument("--remove_source", type=str, default=None)
args = parser.parse_args()

if args.tokenizer == 'gpt':
    enc = tiktoken.get_encoding("cl100k_base")
    eot_token = enc.eot_token
else:
    enc = AutoTokenizer.from_pretrained(args.tokenizer)
    eot_token = enc.sep_token_id

# load the dataset
if args.type == 'sentence':
    df = pd.read_json(os.path.join(args.data_path, 'docs', 'sentences.jsonl'), orient='records', lines=True)
    df.sort_values(by=['doc_id', 'sent_id'], inplace=True)
else:
    df = pd.read_json(os.path.join(args.data_path, 'docs', 'documents.jsonl'), orient='records', lines=True)
    df['doc_id'] = df.index

if args.remove_source is not None:
    df = df[df['source'] != args.remove_source]

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

ALL_DOC_LENS = []
ALL_SOURCES = []

def tokenize_batch(batch: pd.DataFrame):

    batch = batch.copy()

    # tokenize the batch
    if args.tokenizer == 'gpt':
        batch['ids'] = enc.encode_ordinary_batch(batch['text'])
    else:
        batch['ids'] = enc(batch['text'].tolist())['input_ids']
    batch['len'] = batch['ids'].apply(len)
    batch['mask'] = [[row['medical']] * row['len'] for _, row in batch.iterrows()]

    tokens = []
    masks  = []
    doc_ids = []
    sources = []

    for doc_id, doc in batch.groupby('doc_id'):

        if len(doc) <= 1024:

            tokens.append([token for ids in doc['ids'] for token in ids] + [eot_token])
            masks.append([mask for ids in doc['mask'] for mask in ids] + [False])
            doc_ids.append([doc_id % (2**32)] * len(tokens[-1])) # fit into 32 bits
            sources.append([source_to_idx[doc['source'].iloc[0]]] * len(tokens[-1]))
            
            ALL_SOURCES.append(doc['source'].iloc[0])
            ALL_DOC_LENS.append(len(tokens[-1]))
    
    return tokens, masks, doc_ids, sources

all_tokens = []
all_masks  = []
all_doc_ids = []
all_sources = []

i = 0
while i < len(df):

    # get the next at least 1000 rows
    # if the 1001st row is not in the same document, get until the end of the document

    batch_length = min(args.batch_size, len(df) - i)

    if batch_length == args.batch_size:

        last_document = df['doc_id'].iloc[i + batch_length - 1] 
        try:
            _ = df['doc_id'].iloc[i + batch_length]
        except:
            break

        while df['doc_id'].iloc[i + batch_length] == last_document:
            batch_length += 1

    # get the batch
    batch = df.iloc[i:i+batch_length]
    tokens, masks, doc_ids, sources = tokenize_batch(batch)

    all_tokens.extend(tokens)
    all_masks.extend(masks)
    all_doc_ids.extend(doc_ids)
    all_sources.extend(sources)

    i += batch_length

    print(f'tokenized [{i}:{i+batch_length}]/{len(df)}')

all_tokens = np.concatenate(all_tokens)
all_masks  = np.concatenate(all_masks)
all_doc_ids = np.concatenate(all_doc_ids)
all_sources = np.concatenate(all_sources)

if args.tokenizer == 'gpt':
    probe_folder = 'probe'
else:
    probe_folder = args.tokenizer.split('/')[-1]

os.makedirs(os.path.join(args.data_path, probe_folder, args.type), exist_ok=True)
tokens = np.memmap(os.path.join(args.data_path, probe_folder, args.type, 'tokens.bin'), dtype=np.uint32, mode='w+', shape=(len(all_tokens),))
labels = np.memmap(os.path.join(args.data_path, probe_folder, args.type, 'labels.bin'), dtype=bool, mode='w+', shape=(len(all_masks),))
doc_ids = np.memmap(os.path.join(args.data_path, probe_folder, args.type, 'doc_ids.bin'), dtype=np.uint32, mode='w+', shape=(len(all_doc_ids),))
sources = np.memmap(os.path.join(args.data_path, probe_folder, args.type, 'sources.bin'), dtype=np.uint8, mode='w+', shape=(len(all_sources),))

print('saving to files...')

# write to files in batches
for i in range(0, len(all_tokens), args.batch_size):

    batch_size = min(args.batch_size, len(all_tokens) - i)
    tokens[i:i+batch_size] = all_tokens[i:i+batch_size]
    labels[i:i+batch_size] = all_masks[i:i+batch_size]
    doc_ids[i:i+batch_size] = all_doc_ids[i:i+batch_size]
    sources[i:i+batch_size] = all_sources[i:i+batch_size]

tokens.flush()
labels.flush()
doc_ids.flush()
sources.flush()

print('medical ratio:', np.sum(labels) / len(labels))
print('average document length:', np.mean(ALL_DOC_LENS))
print('# of documents > 1024:', np.sum(np.array(ALL_DOC_LENS) > 1024))

source_df = pd.DataFrame({'source': ALL_SOURCES})
print('source distribution:')
print(source_df['source'].value_counts())