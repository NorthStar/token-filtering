"""
Pipeline for supervised fine-tuning (SFT) on chat/conversation data
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
import wandb

import tiktoken
from data.tiktokenizer import TikTokenizer
from model import GPTConfig, GPT
from datasets import load_dataset

from paths import DATA_PATH, MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--model',      type=str, default='mask-51M.pt', help='path to pretrained model')
parser.add_argument('--device',     type=str, default='cuda', help='device to use for computation')
parser.add_argument('--dtype',      type=str, default='bfloat16', help='dtype to use for computation')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'), help='path to pretrained models')
parser.add_argument('--save_path',  type=str, default=os.path.join(MODEL_PATH, 'chat'), help='path to save model')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_batch',    type=int, default=None, help='number of batches to train for')
parser.add_argument('--lr',         type=float, default=1e-6)
parser.add_argument('--dropout',    type=float, default=0.1)
parser.add_argument('--w_decay',    type=float, default=0.01)
parser.add_argument('--beta1',      type=float, default=0.9)
parser.add_argument('--beta2',      type=float, default=0.95)
parser.add_argument('--epochs',     type=int, default=1)
parser.add_argument('--eval_iters', type=int, default=100)
parser.add_argument('--log_int',    type=int, default=100)
parser.add_argument('--max_length', type=int, default=2048, help='maximum sequence length')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='chat-sft', help='wandb project name')
args = parser.parse_args()

# Initialize DDP
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = args.device

model_path = args.model_path
save_path = args.save_path
batch_size = args.batch_size
lr = args.lr
w_decay = args.w_decay
beta1 = args.beta1
beta2 = args.beta2
epochs = args.epochs
eval_iters = args.eval_iters
dtype = args.dtype
max_length = args.max_length

# Initialize wandb (only on master process)
if not args.no_wandb and master_process:
    name = args.model.split('.')[0] + '-chat-sft' + '-lr' + str(lr)
    
    wandb.init(
        project=args.wandb_project,
        name=name,
        config={
            'model': args.model,
            'batch_size': batch_size,
            'learning_rate': lr,
            'weight_decay': w_decay,
            'beta1': beta1,
            'beta2': beta2,
            'epochs': epochs,
            'dtype': dtype,
            'device': device,
            'eval_iters': eval_iters,
            'log_interval': args.log_int,
            'ddp': ddp,
            'world_size': ddp_world_size,
            'max_length': max_length
        }
    )

if master_process and not os.path.exists(save_path):
    os.makedirs(save_path)

torch.manual_seed(1337 + seed_offset)
np.random.seed(1337 + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
def load_model(model_file):
    checkpoint = torch.load(model_file, map_location=device)
    model_args = checkpoint['model_args']
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    return model, model_args

model_file = os.path.join(model_path, args.model)
model, model_args = load_model(model_file)

model.eval()
model.to(device)

# Wrap model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

tokenizer = TikTokenizer('cl100k_base')
cl100k_base = tiktoken.get_encoding('cl100k_base')

enc = tiktoken.Encoding(
    name="cl100k_mask",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        '<|im_start|>': 100261,
        '<|im_end|>': 100262,
        "<|mask|>": 100277
    }
)

def format_chat_messages(messages):
    """
    Format chat messages using ChatML format and track which tokens are assistant content.
    
    Args:
        messages: List of dicts with 'role' and 'content' keys
    
    Returns:
        tokens: List of token IDs
        assistant_mask: Boolean array indicating which tokens are assistant content
    """
    all_tokens = []
    all_masks = []
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        # Format message with ChatML template
        role_start = f"<|im_start|>{role}\n"
        message_content = content
        
        # Tokenize each part separately
        start_tokens = enc.encode(role_start, disallowed_special=())
        content_tokens = enc.encode(message_content, disallowed_special=())
        end_token = enc.encode("<|im_end|>", disallowed_special=())
        
        # Combine tokens for this message
        message_tokens = start_tokens + content_tokens + end_token
        
        # Create mask: only content tokens from assistant messages should have loss computed
        if role == 'assistant':
            # Compute loss on content tokens and end token (<|im_end|>)
            message_mask = [False] * len(start_tokens) + [True] * len(content_tokens) + [True] * len(end_token)
        else:
            # Never compute loss on non-assistant messages
            message_mask = [False] * len(message_tokens)
        
        all_tokens.extend(message_tokens)
        all_masks.extend(message_mask)
    
    return all_tokens, np.array(all_masks, dtype=bool)

class ChatDataset:
    def __init__(self, dataset, shuffle=True, max_length=2048):
        """
        Dataset for chat/conversation data.
        
        Args:
            dataset: HuggingFace dataset with 'messages' column
            shuffle: Whether to shuffle the data
            max_length: Maximum sequence length
        """
        self.data = []
        self.max_length = max_length
        
        if shuffle:
            dataset = dataset.shuffle(seed=42)
        
        # Process and format all examples
        for example in tqdm(dataset, desc="Processing chat data"):
            messages = example['messages']
            
            # Format and tokenize messages, getting tokens and mask
            tokens, assistant_mask = format_chat_messages(messages)
            
            # Skip if too long
            if len(tokens) > max_length:
                continue
            
            self.data.append({
                'tokens': tokens,
                'assistant_mask': assistant_mask
            })
    
    def __len__(self):
        return len(self.data)
    
    def get_batch(self, idx, batch_size):
        """
        Get a batch of data.
        
        Returns:
            inputs: Token IDs (batch_size, seq_len)
            labels: Target tokens with -1 for non-assistant positions (batch_size, seq_len)
        """
        batch_data = self.data[idx:idx+batch_size]
        
        # Pad all sequences to max_length
        inputs_list = []
        labels_list = []
        
        for item in batch_data:
            tokens = item['tokens']
            assistant_mask = item['assistant_mask']
            
            # Pad tokens
            padded_tokens = tokens + [0] * (self.max_length - len(tokens))
            padded_tokens = padded_tokens[:self.max_length]
            
            # Pad mask
            padded_mask = np.concatenate([assistant_mask, np.zeros(self.max_length - len(assistant_mask), dtype=bool)])
            padded_mask = padded_mask[:self.max_length]
            
            # Create labels: copy tokens where mask is True, otherwise -1
            labels = np.where(padded_mask, padded_tokens, -1)
            
            inputs_list.append(padded_tokens)
            labels_list.append(labels)
        
        inputs = torch.tensor(inputs_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Shift for next-token prediction
        # inputs: [0, 1, 2, 3, 4] -> labels: [1, 2, 3, 4, ?]
        inputs = inputs[:, :-1]
        labels = labels[:, 1:]
        
        return inputs, labels

@torch.no_grad()
def estimate_loss(dataset, eval_model=None):
    """Estimate loss on a dataset"""
    if eval_model is None:
        eval_model = model
    
    eval_model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        idx = np.random.randint(0, max(0, len(dataset) - batch_size))
        X, Y = dataset.get_batch(idx, batch_size)
        
        X = X.to(device)
        Y = Y.to(device)
        
        with ctx:
            _, loss = eval_model(X, idx_filter=None, targets=Y, targets_filter=None)
        
        losses[k] = loss.mean().item()
    
    eval_model.train()
    return losses.mean()

# Load smoltalk dataset
if master_process:
    print("Loading smoltalk dataset...")

smoltalk = load_dataset("HuggingFaceTB/smol-smoltalk")
train_dataset = ChatDataset(smoltalk['train'], shuffle=True, max_length=max_length)
test_dataset = ChatDataset(smoltalk['test'], shuffle=False, max_length=max_length)

if master_process:
    print(f"Loaded {len(train_dataset)} training examples")
    print(f"Loaded {len(test_dataset)} test examples")

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# Get raw model for optimizer configuration and evaluation
raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(w_decay, lr, 0.6, 0.04, (beta1, beta2), device_type, optimizer_type='adamw')

results = []

# Initial evaluation
if master_process:
    print("Initial evaluation...")
    train_loss = estimate_loss(train_dataset, raw_model)
    test_loss = estimate_loss(test_dataset, raw_model)
    
    print(f"Initial train loss: {train_loss:.4f}")
    print(f"Initial test loss: {test_loss:.4f}")
    
    if not args.no_wandb:
        wandb.log({
            'train/loss': train_loss,
            'test/loss': test_loss,
            'iteration': 0
        })
    
    results.append({
        'samples_seen': 0,
        'train_loss': train_loss,
        'test_loss': test_loss
    })

# Synchronize all processes before starting training
if ddp:
    dist.barrier()

model.train()
total_loss = 0
num_batches = 0

if args.n_batch is None:
    n_batch = len(train_dataset) // batch_size
else:
    n_batch = args.n_batch

if master_process:
    print(f"Training for {n_batch} batches...")

for batch_idx in range(0, n_batch * batch_size, batch_size):
    
    X, Y = train_dataset.get_batch(batch_idx, batch_size)
    X = X.to(device)
    Y = Y.to(device)
    
    with ctx:
        _, loss = model(X, idx_filter=None, targets=Y, targets_filter=None)
    
    total_loss += loss.item()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    num_batches += 1
    
    # Log progress
    if num_batches % args.log_int == 0:
        
        # Synchronize before evaluation
        if ddp:
            dist.barrier()
        
        if master_process:
            test_loss = estimate_loss(test_dataset, raw_model)
            avg_train_loss = total_loss / args.log_int
            total_loss = 0
            
            if not args.no_wandb:
                wandb.log({
                    'train/loss': avg_train_loss,
                    'test/loss': test_loss,
                    'iteration': batch_idx,
                    'samples_seen': batch_idx
                })
            
            print(f"batch {num_batches} | train_loss {avg_train_loss:.4f} | test_loss {test_loss:.4f}")
            
            results.append({
                'samples_seen': batch_idx,
                'train_loss': avg_train_loss,
                'test_loss': test_loss
            })
        
        # Synchronize after evaluation
        if ddp:
            dist.barrier()

# Save results and checkpoint
if master_process:
    # Save results
    if not os.path.exists('analysis/results/'):
        os.makedirs('analysis/results/')
    
    df = pd.DataFrame(results)
    df.to_csv('analysis/results/chat-sft.csv', index=False)
    
    # Save checkpoint
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
    }
    
    model_name = args.model.split('.')[0] + '-chat-sft'
    print(f"Completed training, saving checkpoint as {model_name}.pt")
    torch.save(checkpoint, os.path.join(save_path, f'{model_name}.pt'))
    
    if not args.no_wandb:
        wandb.finish()

# Cleanup DDP
if ddp:
    dist.destroy_process_group()

