"""
Unified filter for fineweb 10bt - combines token, document, and aggressive filtering approaches
- threading: we can split the dataset into num_threads chunks and process each chunk separately
"""

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import tiktoken
from datasets import load_dataset, DatasetDict, concatenate_datasets # huggingface datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tiktokenizer import TikTokenizer
import sklearn
from sklearn.preprocessing import normalize
import pickle as pkl

import sys
sys.path.append('..')
from model import GPT, GPTConfig
from paths import DATA_PATH, MODEL_PATH

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, choices=['token', 'sent', 'doc', 'none'], required=True)
parser.add_argument('--label', type=str, default=None, choices=['token', 'sentence', 'document'], required=False)
parser.add_argument('--model_path', type=str, default=MODEL_PATH)
parser.add_argument('--model', type=str, default='pubmed-224M', help = 'filepath of model')
parser.add_argument('--probe_dir', type=str, default='bidir', help = 'filepath of probes')
parser.add_argument('--proportion', type=float, default=1.0, help = 'proportion of dataset to tokenize')
parser.add_argument('--num_threads', type=int, default=1, help = 'number of "threads" to use for processing')
parser.add_argument('--thread_idx', type=int, default=0, help = 'which thread are we on?')
parser.add_argument('--data_path', type=str, default=DATA_PATH, help = 'path to save dataset')
parser.add_argument('--hf_path', type=str, default=None, help = 'path to huggingface cache')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--device_ids', type=int, nargs='+', default=None, help='device ids to use')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for processing')
parser.add_argument('--dtype', type=str, default='float16', choices=['float32', 'float16', 'bfloat16'], help='data type for mixed precision')
parser.add_argument('--tokenizer', type=str, default='gpt', choices=['bert', 'roberta', 'gpt'], help='tokenizer type')
parser.add_argument('--output_dir', type=str, default=None, help='output directory')
args = parser.parse_args()

# Set up output directory based on filtering method
if args.tokenizer == 'bert':
    output_dir_map = {'token': f'filtered-bert', 'sent': 'sentfilter', 'doc': 'docfilter-new', 'none': 'unfiltered-new'}
elif args.model == 'roberta-edu':
    output_dir_map = {'token': f'filtered-roberta', 'sent': 'sentfilter', 'doc': 'docfilter-new', 'none': 'unfiltered-new'}
elif args.model == 'edu-61M':
    output_dir_map = {'token': f'filtered-edu', 'sent': 'sentfilter', 'doc': 'docfilter-new', 'none': 'unfiltered-new'}
else:
    output_dir_map = {'token': f'filtered-{args.model.split("-")[1]}', 'sent': 'sentfilter', 'doc': 'docfilter-new', 'none': 'unfiltered-new'}

if args.output_dir is not None:
    output_dir = args.output_dir
else:
    output_dir = output_dir_map[args.method]

if args.num_threads > 1:
    output_dir = f'{output_dir}/{args.thread_idx}'

print(f'output_dir: {output_dir}')
os.makedirs(os.path.join(args.data_path, output_dir), exist_ok=True)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
if args.method == 'none':
    num_proc = 8
else:
    num_proc = 1

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

# set device
device = torch.device(args.device)

# Setup mixed precision
device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = torch.no_grad()

# initialize tokenizer to get special token IDs
enc = tiktoken.get_encoding("cl100k_base")
special_token_ids = np.array([0, enc.eot_token])

# Skip dataset loading for aggro method since we use existing filtered data
dsets = []
for dataset_name in ['fineweb-edu']:

    if args.hf_path is not None:
        dataset = load_dataset(f'Zyphra/dclm-dedup', name = 'sample-100BT', cache_dir=args.hf_path + '/fineweb-100BT', download_mode='reuse_dataset_if_exists')
    else:
        dataset = load_dataset(f'Zyphra/dclm-dedup', name = 'sample-100BT', download_mode='reuse_dataset_if_exists')
    
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset['train']), 60_000_000, replace=False)
    dsets.append(dataset['train'].select(indices))

    # dataset = dataset.shuffle(seed=42)
    # dsets.append(dataset['train'][:30000000]) # first 30M rows

dataset = concatenate_datasets(dsets)
dataset = dataset.shuffle(seed=42)

if args.num_threads > 1:
    dataset = dataset.shard(num_shards=args.num_threads, index=args.thread_idx)

if args.proportion < 1:
    sample_half = dataset.train_test_split(test_size=(1 - args.proportion), seed=42, shuffle=True)
else:
    sample_half = DatasetDict({'train': dataset})

train_test = sample_half['train'].train_test_split(test_size=0.005, seed=42, shuffle=True)
split_dataset = DatasetDict({
    'train' : train_test['train'],
    'val' : train_test['test']
})

print('len(train):', len(split_dataset['train']))
print('len(val):', len(split_dataset['val']))

# load sklearn probe using pickle - different naming conventions for different methods
if args.method not in ['none']:

    if args.label is None:
        if args.method == 'doc':
            probe_file = os.path.join(args.model_path, f'probes/{args.model}-doc.pkl')
        elif args.method == 'token':
            if args.tokenizer == 'bert':
                probe_file = os.path.join(args.model_path, f'probes/{args.model.split("/")[-1]}-token.pkl')
            else:
                probe_file = os.path.join(args.model_path, f'probes/{args.model}-token.pkl')
        else:
            probe_file = os.path.join(args.model_path, f'probes/{args.model}-sent.pkl')
    else:
        probe_file = os.path.join(args.model_path, f'probes/{args.model}-{args.method}-{args.label}.pkl')

    saved_probe = pkl.load(open(probe_file, 'rb'))
    try: # for new refactor
        layer = saved_probe['layer']
        probe = saved_probe['probe']
    except: # for old repo
        layer = 0 if args.method == 'doc' else 2
        probe = saved_probe
else:
    # No probe needed for 'none' methods
    probe = None
    layer = 0

def load_gpt(model_file, device):
    checkpoint = torch.load(model_file, map_location=device)
    model_args = checkpoint['model_args']
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    return model

if args.method not in ['none']:
    if args.tokenizer == 'gpt':
        # Load both left and right models
        print(f'loading left-{args.model}.pt and right-{args.model}.pt')
        left_model = load_gpt(os.path.join(args.model_path, f'{args.probe_dir}/left-{args.model}.pt'), device)
        right_model = load_gpt(os.path.join(args.model_path, f'{args.probe_dir}/right-{args.model}.pt'), device)
        model = {'left': left_model, 'right': right_model}
        model['left'].to(device)
        model['right'].to(device)
        model['left'].eval()
        model['right'].eval()
        
        hidden_size = model['left'].config.n_embd * 2  # concatenated features
        max_length = model['left'].config.block_size
    else:
        if args.tokenizer == 'bert':
            model = AutoModelForMaskedLM.from_pretrained(args.model)
        else:
            model = AutoModelForMaskedLM.from_pretrained(os.path.join(args.model_path, args.model))
        model.config.output_hidden_states = True

        if args.tokenizer == 'roberta':
            model.roberta.encoder.layer = model.roberta.encoder.layer[:layer + 1]
        else:
            model.model.layers = model.model.layers[:layer + 1]

        hidden_size = model.config.hidden_size
        max_length = model.config.max_position_embeddings - 2

        if args.device_ids is not None:
            model = torch.nn.DataParallel(model, device_ids=args.device_ids)

        model.to(device)
        model.eval()

        # enable memory-efficient attention if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        model = torch.compile(model)
else:
    # No model needed for 'none' methods
    model = None
    max_length = 512  # Use a reasonable default

if args.tokenizer == 'bert':
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Convert special token strings to IDs
    special_token_ids = set(tokenizer.all_special_ids)
else:
    tokenizer = TikTokenizer()
    special_token_ids = {0, enc.eot_token}  

dtype = np.uint32

### convert the sklearn probe to a torch layer...
if probe is not None:
    weights = torch.tensor(probe.coef_,      dtype=torch.float32)
    bias    = torch.tensor(probe.intercept_, dtype=torch.float32)

    torch_probe = nn.Linear(weights.shape[1], weights.shape[0])
    with torch.no_grad():
        torch_probe.weight.copy_(weights)
        torch_probe.bias.copy_(bias)

    torch_probe.to(device)
    torch_probe.eval()
else:
    torch_probe = None

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

def prep_tokens(token_ids, pad_token_id=0):
    return {'input_ids' : token_ids, 'attention_mask' : (token_ids != pad_token_id).long()}

# token level filtering
def process_token(examples):        

    # use larger max_length for tokenization to match GPT/BERT
    tokenize_max_length = 1024 # if args.tokenizer == 'roberta' else max_length
    
    inputs = tokenizer(examples['text'], padding = True, truncation = True, max_length = tokenize_max_length, return_tensors = 'pt')
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    
    attention_mask = inputs['attention_mask'].bool()
    batch_lens = attention_mask.sum(dim=1)
    flat_input_ids = inputs['input_ids'][attention_mask]
    
    with ctx:
        if args.tokenizer == 'gpt':
            # Handle dual GPT models
            if tokenize_max_length > max_length:
                # GPT with chunking: split sequences into chunks along sequence dimension
                # this is a hack to get around the fact that some GPT models have limited context lengths
                # (e.g., 256 or 128) while we tokenize with max_length=1024
                num_chunks = tokenize_max_length // max_length
                chunks = []
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * max_length
                    end_idx = min(start_idx + max_length, tokenize_max_length)
                    
                    chunk_input_ids = inputs['input_ids'][:, start_idx:end_idx]
                    chunk_features = get_gpt_hidden_states(model['left'], model['right'], chunk_input_ids, layer)
                    chunks.append(chunk_features)
                
                # Concatenate all chunks along sequence dimension
                combined_features = torch.cat(chunks, dim=1)
                hidden_states = combined_features[attention_mask]
            else:
                features = get_gpt_hidden_states(model['left'], model['right'], inputs['input_ids'], layer)
                hidden_states = features[attention_mask]
        elif args.tokenizer == 'roberta' and tokenize_max_length > max_length:
            # RoBERTa with chunking: split sequences in half along sequence dimension
            # this is a hack to get around the fact that roberta has a max length of 512
            # so we split the sequence in half and process each half separately

            first_half = {'input_ids': inputs['input_ids'][:, :max_length], 
                         'attention_mask': inputs['attention_mask'][:, :max_length]}
            second_half = {'input_ids': inputs['input_ids'][:, max_length:], 
                          'attention_mask': inputs['attention_mask'][:, max_length:]}
            
            # Process both halves
            outputs_first = model(**first_half)
            outputs_second = model(**second_half)
            
            # Concatenate hidden states along sequence dimension, then flatten by attention mask
            combined_hidden = torch.cat([outputs_first.hidden_states[layer], 
                                        outputs_second.hidden_states[layer]], dim=1)
            hidden_states = combined_hidden[attention_mask]
        else:
            # forward pass through huggingface model
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[layer][attention_mask]

        probs = torch.sigmoid(torch_probe(hidden_states))
        predictions = torch.cat([1 - probs, probs], dim=-1)[:, 1]
    
    batch_ids = []
    # filtered_masks = []
    all_predictions = []
    
    cumsum_lens = torch.nn.functional.pad(batch_lens[:-1].cumsum(0), (1, 0))
    
    for i, (start_idx, length) in enumerate(zip(cumsum_lens, batch_lens)):
        end_idx = start_idx + length
        batch_ids.append(flat_input_ids[start_idx:end_idx])
        # filtered_masks.append(masks[start_idx:end_idx])
        all_predictions.append(predictions[start_idx:end_idx])
    
    return {'ids': batch_ids, 'len': batch_lens, 'filtered': all_predictions}

def process_doc(examples):
    """Document-level filtering process"""
    inputs = tokenizer(examples['text'], padding = True, truncation = True, max_length = max_length, return_tensors = 'pt')

    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    
    attention_mask = inputs['attention_mask'].bool()
    batch_lens = attention_mask.sum(dim=1)
    flat_input_ids = inputs['input_ids'][attention_mask]
    
    with ctx:
        if args.tokenizer == 'gpt':
            # Handle dual GPT models
            features = get_gpt_hidden_states(model['left'], model['right'], inputs['input_ids'], layer)
            hidden_states = features * attention_mask.unsqueeze(-1)
        else:
            # forward pass through huggingface model
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[layer] * attention_mask.unsqueeze(-1)
        
        # convert to numpy for sklearn probe and normalize
        features = hidden_states.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
        features = features.detach().cpu().numpy()
        
        # use sklearn probe to predict
        predictions = probe.predict(features)
        
        # convert back to torch tensor
        predictions = torch.tensor(predictions)
    
    batch_ids = []
    filtered  = []
    
    for i in range(len(inputs['input_ids'])):
        batch_ids.append(inputs['input_ids'][i][attention_mask[i]])
        filtered.append(predictions[i])

    return {'ids': batch_ids, 'filtered': filtered, 'len': batch_lens}

def process_none(examples):
    # no filtering just tokenize
    inputs = tokenizer(examples['text'], padding = True, truncation = True, max_length = max_length, return_tensors = 'pt')
    
    batch_ids = []
    batch_lens = []
    
    for i in range(len(inputs['input_ids'])):
        # Get actual length (non-padded)
        actual_length = (inputs['input_ids'][i] != 0).sum()
        batch_ids.append(inputs['input_ids'][i][:actual_length])
        batch_lens.append(actual_length)
    
    batch_lens = torch.tensor(batch_lens)
    
    return {'ids': batch_ids, 'len': batch_lens}

# Choose processing function based on method
process_func = {'token': process_token, 'sent': process_token, 'doc': process_doc, 'none': process_none}[args.method]

# tokenize the dataset with larger batch size and optimizations
remove_columns = ['text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count'] if args.method == 'doc' else ['text']

tokenized = split_dataset.map(
    process_func,
    batched=True,
    batch_size=args.batch_size,
    remove_columns=[col for col in split_dataset['train'].column_names if col not in ['ids', 'len', 'keep', 'filtered']],
    desc='tokenizing the splits',
    num_proc=num_proc  # Keep single process to avoid CUDA context issues
)

if args.method == 'doc':
    
    num_removed = (np.array(tokenized['train']['filtered']) == 1).sum()
    total_documents = len(tokenized['train']['filtered'])
    print(f'removed {num_removed}/{total_documents} documents')

    tokenized = tokenized.filter(
        lambda x: x['filtered'] == 0,
        desc='filtering documents',
        num_proc=num_proc
    )

sep_token_id = enc.eot_token

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():

    if split == 'test':
        continue

    arr_len = np.sum(dset['len'], dtype=np.uint64)

    # save token IDs
    filename = os.path.join(args.data_path, output_dir, f'{split}.bin')
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    print(f'saving as {dtype} with shape {arr.shape}')

    # for token filtering, also save filter masks
    if args.method == 'token' or args.method == 'sent':
        filtered_filename = os.path.join(args.data_path, output_dir, f'{split}_filter.bin')
        filtered_arr = np.memmap(filtered_filename, dtype=np.float16, mode='w+', shape=(arr_len,))
    
    total_batches = 1024
    idx = 0
    
    # for batch_idx in tqdm(range(total_batches), desc=f'writing {split}.bin'):
    write_batch_size = len(dset) // 1024
    for batch_idx in tqdm(range(0, len(dset), write_batch_size), desc=f'writing {split}.bin'):

        # batch together samples for faster write
        # batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        batch = dset[batch_idx:min(batch_idx + write_batch_size, len(dset))]
        
        arr_batch = np.concatenate(batch['ids'])
        
        if args.method == 'token' or args.method == 'sent':
            filtered_batch = np.concatenate(batch['filtered'])
            # special_token_mask = torch.isin(torch.tensor(arr_batch), special_token_ids)        
            special_token_mask = np.isin(arr_batch, special_token_ids)

            filtered_batch[special_token_mask] = 0
            
            # write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            filtered_arr[idx : idx + len(filtered_batch)] = filtered_batch
            idx += len(arr_batch)
        else:
            # write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
    
    arr.flush()
    if args.method == 'token' or args.method == 'sent':
        filtered_arr.flush()
    
    lens_filename = os.path.join(args.data_path, output_dir, f'{split}_lens.bin')

    # Handle the fact that dset['len'] is a list of scalars
    all_lens_list = []
    for batch_lens in dset['len']:
        if isinstance(batch_lens, (list, tuple)):
            all_lens_list.extend(batch_lens)
        else:
            # It's a scalar integer
            all_lens_list.append(batch_lens)
    all_lens = np.array(all_lens_list, dtype=np.uint32)
    lens_arr = np.memmap(lens_filename, dtype=np.uint32, mode='w+', shape=(len(all_lens),))
    lens_arr[:] = all_lens
    lens_arr.flush()
    print(f"Saved {len(all_lens)} document lengths to {lens_filename}") 