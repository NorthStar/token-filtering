"""
Adversarial finetuning script that trains on pubmed_medical.bin.
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model filename (e.g. model.pt)')
parser.add_argument('--data_path', type=str, default=os.path.join(DATA_PATH, 'finetune'), help='path to data directory with pubmed_medical.bin')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'), help='path to pretrained models')
parser.add_argument('--save_path',  type=str, default=os.path.join(MODEL_PATH, 'pubmed-adversarial-finetune'), help='path to save model')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help='number of gradient accumulation steps')
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.95)
parser.add_argument('--eval_int', type=int, default=2, help='compute loss, save checkpoint, log to wandb every N steps')
parser.add_argument('--eval_iters', type=int, default=100, help='number of batches for loss evaluation')
parser.add_argument('--min_iters', type=int, default=250, help='stop training after this many iterations')
parser.add_argument('--min_val_loss', type=float, default=2.60, help='stop training if val loss falls below this')
parser.add_argument('--wandb_project', type=str, default='medical-adversarial-finetune-pubmed')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--block_size', type=int, default=2048, help='sequence length')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--grad_clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--results_dir', type=str, default=os.path.join('analysis', 'adversarial-finetune'), help='directory for results CSV')
args = parser.parse_args()

# Initialize DDP if available
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
    ddp_world_size = 1
    device = args.device

torch.manual_seed(42 + seed_offset)
np.random.seed(42 + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Initialize wandb
if not args.no_wandb and master_process:
    import wandb
    run_name = f"{args.model.split('.')[0]}-pubmed-ft-lr{args.learning_rate}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )

# Load model
def load_model(model_file):
    checkpoint = torch.load(model_file, map_location=device)
    model_args = checkpoint['model_args']
    
    # Override dropout if specified
    if args.dropout != model_args.get('dropout', 0.0):
        model_args['dropout'] = args.dropout
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    return model, model_args

model_file = os.path.join(args.model_path, args.model)
if master_process:
    print(f"Loading model from {model_file}")
model, model_args = load_model(model_file)
block_size = model_args.get('block_size', args.block_size)

model.to(device)

# Wrap in DDP if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

if master_process:
    print(f"Model loaded with {raw_model.get_num_params()} parameters")
    print(f"Block size: {block_size}")

# Load data
train_data = np.memmap(os.path.join(args.data_path, 'pubmed_train.bin'), dtype=np.uint32, mode='r')
test_data = np.memmap(os.path.join(args.data_path, 'pubmed_test.bin'), dtype=np.uint32, mode='r')

if master_process:
    print(f"Train tokens (pubmed_train): {len(train_data):,}")
    print(f"Test tokens (pubmed_test): {len(test_data):,}")

def get_batch(split):
    """Get a random batch from the data."""
    if split == 'train':
        data = train_data
    elif split == 'test':
        data = test_data
    else:
        raise ValueError(f"Unknown split: {split}")
    
    max_start_idx = len(data) - block_size - 1
    ix = torch.randint(0, max_start_idx, (args.batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    
    return x, y

def count_tokens_in_batch(x):
    """Count non-padding tokens."""
    return (x != 0).sum().item()

@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and test sets."""
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, idx_filter=None, targets=y, targets_filter=None)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Learning rate scheduler with warmup
def get_lr(it):
    return args.learning_rate

# Configure optimizer
optimizer = raw_model.configure_optimizers(
    args.weight_decay, 
    args.learning_rate, 
    args.learning_rate,  # embed_learning_rate 
    args.learning_rate,  # scalar_learning_rate
    (args.beta1, args.beta2), 
    device_type, 
    optimizer_type='adamw'
)

# Set initial learning rates
for param_group in optimizer.param_groups:
    param_group["initial_lr"] = param_group["lr"]

scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

# Training loop
results = []
total_tokens_seen = 0
iter_num = 0
best_val_loss = float('inf')

# Ensure output directories exist
if master_process:
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * block_size
if master_process:
    print(f"\nStarting training (min_iters={args.min_iters}, min_val_loss={args.min_val_loss})...")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.gradient_accumulation_steps * ddp_world_size * args.batch_size}")
    print(f"Tokens per iteration (all GPUs): {tokens_per_iter:,}")
    print(f"Loss/checkpoint every {args.eval_int} steps ({args.eval_iters} batches)")

# Initial evaluation before training (all ranks must participate for DDP sync)
if master_process:
    print("\nInitial loss evaluation...")
losses = estimate_loss()
if master_process:
    print(f"Initial: train {losses['train']:.4f} | test {losses['test']:.4f}")
    
    results.append({
        'step': 0,
        'tokens_seen': 0,
        'train_loss': losses['train'],
        'test_loss': losses['test'],
        'lr': args.learning_rate,
    })
    
    if not args.no_wandb:
        wandb.log({
            'iter': 0,
            'tokens_seen': 0,
            'train/loss': losses['train'],
            'test/loss': losses['test'],
            'lr': args.learning_rate,
        })

model.train()
t0 = time.time()

while True:
    
    # Update learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
    
    # Checkpoint evaluation: compute loss, save model, log to wandb
    # All ranks must call estimate_loss() together for DDP sync
    should_stop = False
    if iter_num % args.eval_int == 0:
        losses = estimate_loss()
        
        if master_process:
            print(f"step {iter_num} | tokens {total_tokens_seen:,} | train {losses['train']:.4f} | test {losses['test']:.4f}")
            
            # Log to wandb
            if not args.no_wandb:
                wandb.log({
                    'iter': iter_num,
                    'tokens_seen': total_tokens_seen,
                    'train/loss': losses['train'],
                    'test/loss': losses['test'],
                    'lr': lr,
                })
            
            # Save results to CSV
            results.append({
                'step': iter_num,
                'tokens_seen': total_tokens_seen,
                'train_loss': losses['train'],
                'test_loss': losses['test'],
                'lr': lr,
            })
            
            # Save checkpoint (overwrite same file)
            if losses['test'] < best_val_loss:
                best_val_loss = losses['test']
            
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'tokens_seen': total_tokens_seen,
                'best_val_loss': best_val_loss,
                'config': vars(args),
            }
            
            model_name = args.model.split('.')[0] + '-finetuned.pt'
            torch.save(checkpoint, os.path.join(args.save_path, model_name))
            
            # Stop if BOTH min_iters reached AND test loss below threshold
            if iter_num >= args.min_iters and losses['test'] < args.min_val_loss:
                print(f"Stopping: iter {iter_num} >= {args.min_iters} and test loss {losses['test']:.4f} < {args.min_val_loss}")
                should_stop = True
        
        # Broadcast stop decision to all ranks
        if ddp:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1
        
        if should_stop:
            break
    
    # Forward backward update with gradient accumulation
    for micro_step in range(args.gradient_accumulation_steps):
        if ddp:
            # Only sync gradients at the last micro step
            model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
        
        # Get batch
        x, y = get_batch('train')
        
        # Count tokens (multiply by world_size to count all GPUs)
        batch_tokens = count_tokens_in_batch(x) * ddp_world_size
        total_tokens_seen += batch_tokens
        
        # Forward pass
        with ctx:
            _, loss = model(x, idx_filter=None, targets=y, targets_filter=None)
            loss = loss / args.gradient_accumulation_steps  # Scale loss for accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
    
    # Gradient clipping
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    iter_num += 1

# Final evaluation and save (all ranks must participate for DDP sync)
losses = estimate_loss()
if master_process:
    print(f"\nFinal: step {iter_num} | tokens {total_tokens_seen:,} | train {losses['train']:.4f} | test {losses['test']:.4f}")
    
    results.append({
        'step': iter_num,
        'tokens_seen': total_tokens_seen,
        'train_loss': losses['train'],
        'test_loss': losses['test'],
        'lr': get_lr(iter_num),
    })
    
    # Save final checkpoint
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'tokens_seen': total_tokens_seen,
        'best_val_loss': best_val_loss,
        'config': vars(args),
    }
    model_name = args.model.split('.')[0] + '-finetuned.pt'
    torch.save(checkpoint, os.path.join(args.save_path, model_name))
    print(f"Saved final checkpoint to {model_name}")
    
    # Save results CSV (include LR in filename for sweep runs)
    lr_str = f"{args.learning_rate:.0e}".replace('-', '').replace('+', '')
    csv_name = f"{args.model.split('.')[0]}-lr{lr_str}-pubmed.csv"
    csv_path = os.path.join(args.results_dir, csv_name)
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    if not args.no_wandb:
        wandb.finish()

# Cleanup DDP
if ddp:
    dist.destroy_process_group()

