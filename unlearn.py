"""
Representation-based unlearning for open-ended text generation.

Takes two .bin files (forget and retain) and applies representation unlearning,
computing loss only after the first 1/32nd of tokens (context).
"""

import os
import sys
import copy
import argparse
import numpy as np
import torch
from contextlib import nullcontext
from functools import partial

import wandb
import minlora

from model import GPTConfig, GPT
from paths import DATA_PATH, MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model filename (e.g. model.pt)')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'), help='path to pretrained models')
parser.add_argument('--save_path', type=str, default=os.path.join(MODEL_PATH, 'unlearn'), help='path to save unlearned model')
parser.add_argument('--forget_data', type=str, default=os.path.join(DATA_PATH, 'test', 'test_ood.bin'), help='path to forget .bin file')
parser.add_argument('--retain_data', type=str, default=os.path.join(DATA_PATH, 'test', 'test_target.bin'), help='path to retain .bin file')
parser.add_argument('--test_forget', type=str, default=os.path.join(DATA_PATH, 'test', 'test_ood.bin'), help='path to test forget .bin file (for eval)')
parser.add_argument('--test_retain', type=str, default=os.path.join(DATA_PATH, 'test', 'test_target.bin'), help='path to test retain .bin file (for eval)')
parser.add_argument('--alpha', type=float, default=100.0, help='scaling factor for loss coefficients')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--lora_rank', type=int, default=4)
parser.add_argument('--lora_alpha', type=float, default=8)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--block_size', type=int, default=2048, help='sequence length')
parser.add_argument('--context_fraction', type=float, default=1/32, help='fraction of tokens to use as context (default 1/32)')
parser.add_argument('--max_iters', type=int, default=1000, help='max training iterations')
parser.add_argument('--wandb_project', type=str, default='med-unlearn-base')
parser.add_argument('--log_int', type=int, default=10, help='logging interval')
parser.add_argument('--eval_iters', type=int, default=20, help='number of eval batches')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--target_layers', type=int, nargs='+', default=[0, 4, 8, 12, 16, 20, 24, 28, 31], help='layers to target for unlearning')
parser.add_argument('--lora', action='store_true', help='use LoRA for parameter-efficient finetuning')
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping')
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
args = parser.parse_args()

# Setup
torch.manual_seed(1337)
np.random.seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in args.device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
def load_model(model_file):
    checkpoint = torch.load(model_file, map_location=args.device)
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

print(f"Loading model from {os.path.join(args.model_path, args.model)}")
model, model_args = load_model(os.path.join(args.model_path, args.model))
learned_model = copy.deepcopy(model)
num_layers = model_args['n_layer']

# Filter target layers to valid range
args.target_layers = [l for l in args.target_layers if l < num_layers]
print(f"Using target layers: {args.target_layers} (model has {num_layers} layers)")

# LoRA setup (conditional)
if args.lora:
    lora_config = {
        torch.nn.Embedding: {
            "weight": partial(minlora.LoRAParametrization.from_embedding, rank=args.lora_rank, lora_alpha=args.lora_alpha),
        },
        torch.nn.Linear: {
            "weight": partial(minlora.LoRAParametrization.from_linear, rank=args.lora_rank, lora_alpha=args.lora_alpha),
        },
    }
    minlora.add_lora(learned_model, lora_config=lora_config)
    minlora.tie_weights(linear=learned_model.lm_head, embedding=learned_model.transformer.wte)
    print(f"LoRA enabled with rank={args.lora_rank}, alpha={args.lora_alpha}")

model.to(args.device)
learned_model.to(args.device)

print(f"Model loaded with {model.get_num_params()} parameters")

# Load data
forget_data = np.memmap(args.forget_data, dtype=np.uint32, mode='r')
retain_data = np.memmap(args.retain_data, dtype=np.uint32, mode='r')

print(f"Forget data: {len(forget_data):,} tokens")
print(f"Retain data: {len(retain_data):,} tokens")

# Optional test data for evaluation
if args.test_forget is not None:
    test_forget_data = np.memmap(args.test_forget, dtype=np.uint32, mode='r')
    print(f"Test forget data: {len(test_forget_data):,} tokens")
else:
    test_forget_data = None

if args.test_retain is not None:
    test_retain_data = np.memmap(args.test_retain, dtype=np.uint32, mode='r')
    print(f"Test retain data: {len(test_retain_data):,} tokens")
else:
    test_retain_data = None

def get_batch(data, batch_size, block_size):
    """Get a random batch from data."""
    max_start_idx = len(data) - block_size - 1
    ix = torch.randint(0, max_start_idx, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x = x.pin_memory().to(args.device, non_blocking=True)
    else:
        x = x.to(args.device)
    
    return x

def get_context_mask(batch_size, block_size, context_fraction, device):
    """
    Create a mask where 1 = position to compute loss on (after context).
    The first context_fraction of tokens are masked out (0).
    """
    context_len = int(block_size * context_fraction)
    mask = torch.ones(batch_size, block_size, device=device)
    mask[:, :context_len] = 0
    return mask

def get_reps(model, input_ids, attn_mask, n_layers):
    """Get representations from each layer."""
    reps = []
    masks = []

    tok_emb = model.transformer.wte(input_ids)
    h = model.transformer.drop(tok_emb)
    reps.append(h)
    masks.append(attn_mask.unsqueeze(-1))
    
    for i in range(n_layers - 1):
        h = model.transformer.h[i](h)
        reps.append(h)
        masks.append(attn_mask.unsqueeze(-1))
    
    return torch.stack(reps), torch.stack(masks)  # [n_layer, batch_size, seq_len, n_embd]

def get_coeffs(cur_idx, total_idx, alpha):
    """Get forget/retain coefficients that shift over training."""
    return alpha * (1 - cur_idx / (2 * total_idx)), alpha * cur_idx / (2 * total_idx)

def get_loss(frozen_reps, learned_reps, c, target_layers):
    """
    Compute unlearning loss:
    - Forget loss: minimize cosine similarity between frozen and learned reps (push apart)
    - Retain loss: minimize L2 distance between frozen and learned reps (keep similar)
    """
    if len(c) != 2:
        raise ValueError('c must be a tuple of length 2 for circuit breakers')
    
    # Retain loss: L2 distance
    retain_loss = torch.norm(
        frozen_reps['retain'] * frozen_reps['retain_mask'] - learned_reps['retain'] * learned_reps['retain_mask'],
        dim=-1,
        p=2,
        dtype=torch.float
    ).sum() / (learned_reps['retain_mask'].sum() + 1e-8)
    
    # Forget loss: cosine similarity (we want to minimize this, push representations apart)
    normed_frozen = frozen_reps['forget'][target_layers] / (torch.norm(frozen_reps['forget'][target_layers], dim=-1, keepdim=True) + 1e-8)
    normed_learned = learned_reps['forget'][target_layers] / (torch.norm(learned_reps['forget'][target_layers], dim=-1, keepdim=True) + 1e-8)

    cos_sim = (normed_frozen * normed_learned).sum(dim=-1, keepdim=True)  # [n_layer, batch_size, seq_len, 1]
    cos_sim = cos_sim * learned_reps['forget_mask'][target_layers]
    mean_cos_sim = cos_sim.sum() / (learned_reps['forget_mask'][target_layers].sum() + 1e-8)
    forget_loss = torch.relu(mean_cos_sim)

    return c[0] * forget_loss + c[1] * retain_loss, forget_loss.item(), retain_loss.item()

@torch.no_grad()
def estimate_loss(model_to_eval, data, n_iters):
    """Estimate language modeling loss on data."""
    model_to_eval.eval()
    losses = torch.zeros(n_iters)
    
    for k in range(n_iters):
        x = get_batch(data, args.batch_size, args.block_size)
        y = torch.cat([x[:, 1:], x[:, :1]], dim=1)  # Shift for next-token prediction
        
        with ctx:
            _, loss = model_to_eval(x, idx_filter=None, targets=y, targets_filter=None)
        losses[k] = loss.item()
    
    model_to_eval.train()
    return losses.mean().item()

# Setup optimizer
if args.lora:
    optimizer = torch.optim.Adam(
        [{"params": list(minlora.get_lora_params(learned_model)), "weight_decay": args.weight_decay}],
        lr=args.lr
    )
else:
    optimizer = learned_model.configure_optimizers(
        args.weight_decay, args.lr, args.lr, args.lr, (0.9, 0.95), device_type, 'adamw'
    )

scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

# Setup target layers mask
target_layers = torch.zeros(num_layers, dtype=torch.bool)
for l in args.target_layers:
    target_layers[l] = True
print(f"Target layers mask: {target_layers.sum().item()} layers active")

# Calculate total iterations
total_iters = args.max_iters

# Initialize wandb
if not args.no_wandb:
    run_name = f"{args.model.split('.')[0]}-unlearn-lr{args.lr}-alpha{args.alpha}-lora_rank{args.lora_rank}-lora_alpha{args.lora_alpha}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )

# Initial evaluation
print("\nInitial evaluation...")
if test_forget_data is not None:
    init_forget_loss = estimate_loss(learned_model, test_forget_data, args.eval_iters)
    print(f"Initial forget loss: {init_forget_loss:.4f}")
else:
    init_forget_loss = None

if test_retain_data is not None:
    init_retain_loss = estimate_loss(learned_model, test_retain_data, args.eval_iters)
    print(f"Initial retain loss: {init_retain_loss:.4f}")
else:
    init_retain_loss = None

# Create output directory
os.makedirs(args.save_path, exist_ok=True)

# Training loop
print(f"\nStarting unlearning for {args.max_iters} iterations...")
print(f"Context fraction: {args.context_fraction} ({int(args.block_size * args.context_fraction)} tokens)")

model.eval()  # Frozen model stays in eval mode
learned_model.train()

for iter_num in range(args.max_iters):
    
    # Get coefficients (shift from forget to retain over training)
    c = get_coeffs(iter_num, total_iters, args.alpha)
    
    # Get batches
    forget_x = get_batch(forget_data, args.batch_size, args.block_size)
    retain_x = get_batch(retain_data, args.batch_size, args.block_size)
    
    # Create context mask (1 for positions after context, 0 for context)
    context_mask = get_context_mask(args.batch_size, args.block_size, args.context_fraction, args.device)
    
    # Get representations from both models
    reps = {}
    for m_type in ['frozen', 'learned']:
        reps[m_type] = {}
        m = model if m_type == 'frozen' else learned_model
        
        reps[m_type]['forget'], reps[m_type]['forget_mask'] = get_reps(m, forget_x, context_mask, num_layers)
        reps[m_type]['retain'], reps[m_type]['retain_mask'] = get_reps(m, retain_x, context_mask, num_layers)
    
    # Compute loss
    with ctx:
        loss, forget_loss, retain_loss = get_loss(reps['frozen'], reps['learned'], c, target_layers)
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(learned_model.parameters(), args.grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # Logging
    if iter_num % args.log_int == 0:
        learned_model.eval()
        
        log_str = f"[{iter_num}/{args.max_iters}] c_f: {c[0]:.2f} | c_r: {c[1]:.2f} | loss: {loss.item():.4f} | forget: {forget_loss:.4f} | retain: {retain_loss:.4f}"
        
        log_dict = {
            'train/unlearning_loss': loss.item(),
            'train/forget_loss': forget_loss,
            'train/retain_loss': retain_loss,
            'train/c_forget': c[0],
            'train/c_retain': c[1],
        }
        
        # Evaluate on test data if available
        if test_forget_data is not None:
            test_forget_loss = estimate_loss(learned_model, test_forget_data, args.eval_iters)
            delta_forget = init_forget_loss - test_forget_loss if init_forget_loss is not None else 0
            log_str += f" | Δ forget: {delta_forget:+.4f}"
            log_dict['test/forget_loss'] = test_forget_loss
            log_dict['test/delta_forget_loss'] = delta_forget
        
        if test_retain_data is not None:
            test_retain_loss = estimate_loss(learned_model, test_retain_data, args.eval_iters)
            delta_retain = init_retain_loss - test_retain_loss if init_retain_loss is not None else 0
            log_str += f" | Δ retain: {delta_retain:+.4f}"
            log_dict['test/retain_loss'] = test_retain_loss
            log_dict['test/delta_retain_loss'] = delta_retain
        
        print(log_str)
        
        if not args.no_wandb:
            wandb.log(log_dict, step=iter_num)
        
        learned_model.train()

# Final save
print("\nSaving final model...")
checkpoint = {
    'model': learned_model.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'config': vars(args),
}

model_name = args.model.replace('.pt', '-unlearned.pt')
save_file = os.path.join(args.save_path, model_name)
torch.save(checkpoint, save_file)
print(f"Training complete. Model saved to: {save_file}")

if not args.no_wandb:
    wandb.finish()
