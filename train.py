"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from omegaconf import OmegaConf
from paths import DATA_PATH, MODEL_PATH

# evil config magic
cfg_file = 'config/gpt-51M.yaml'
for i, arg in enumerate(sys.argv):
    if arg[:3] == 'cfg':
        cfg_file = arg.split('=')[1]
        sys.argv.pop(i)

cfg = OmegaConf.load(cfg_file)
cfg.update(OmegaConf.from_cli())

for key in cfg:
    try:
        exec(key + '=' + str(cfg[key]))
    except (NameError, SyntaxError) as e:
        exec(key + '="' + cfg[key] + '"')

cfg = OmegaConf.to_container(cfg)

# compute max_tokens from max_iters (before gradient_accumulation_steps is divided by world_size)
max_tokens = max_iters * batch_size * block_size * gradient_accumulation_steps

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # set per-rank triton cache to avoid race conditions with torch.compile
    triton_cache = os.environ.get('TRITON_CACHE_DIR', '')
    if triton_cache:
        # fix the rank suffix to use actual local rank (bash script can't know it)
        if '/rank_' in triton_cache:
            triton_cache = triton_cache.rsplit('/rank_', 1)[0]
        os.environ['TRITON_CACHE_DIR'] = f"{triton_cache}/rank_{ddp_local_rank}"
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = data_path # os.path.join(data_path, dataset)
loader_dtype = np.uint32
def get_batch(split, it, mask_threshold=0.5, noise_level=0.0, reverse=False):
    """
    randomly get a batch of data from the dataset

    Args:
        split: 'train' or 'val'
        mask_threshold: threshold for loss masking / removing
        noise_level: random noising for ablating classifier acc
    """

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=loader_dtype, mode='r')
        if mask:
            filter_data = np.memmap(os.path.join(data_dir, 'train_filter.bin'), dtype=np.float16, mode='r')
        # limit to first train_tokens tokens
        max_start_idx = min(len(data), int(train_tokens)) - block_size
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=loader_dtype, mode='r')
        if mask:
            filter_data = np.memmap(os.path.join(data_dir, 'val_filter.bin'), dtype=np.float16, mode='r')
        max_start_idx = len(data) - block_size
    
    
    if reverse: # for training right-to-left halves of bidirs
        ix = torch.randint(1, max_start_idx, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size][::-1]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i-1:i+block_size-1][::-1]).astype(np.int64)) for i in ix])
    else:
        ix = torch.randint(0, max_start_idx, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if mask and it >= begin_filter_step:
        # print(f'data.shape: {data.shape}')
        # print(f'filter_data.shape: {filter_data.shape}')
        # print(f'x.shape: {x.shape}')
    
        x_filter = torch.stack([torch.from_numpy((filter_data[i:i+block_size] >= mask_threshold).astype(bool)) for i in ix])
        y_filter = torch.stack([torch.from_numpy((filter_data[i+1:i+1+block_size] >= mask_threshold).astype(bool)) for i in ix])

        # x_filter = torch.stack([torch.from_numpy((filter_data[i:i+block_size]).astype(bool)) for i in ix])
        # y_filter = torch.stack([torch.from_numpy((filter_data[i+1:i+1+block_size]).astype(bool)) for i in ix])

        if noise_level > 0:
            x_filter = x_filter ^ (torch.rand(x_filter.shape, device=x_filter.device) < noise_level)
            y_filter = y_filter ^ (torch.rand(y_filter.shape, device=y_filter.device) < noise_level)
    
    else:
        # return all zeros (no masking) when mask=False
        x_filter = torch.zeros_like(x, dtype=torch.bool)
        y_filter = torch.zeros_like(y, dtype=torch.bool)
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        x_filter, y_filter = x_filter.pin_memory().to(device, non_blocking=True), y_filter.pin_memory().to(device, non_blocking=True)
    
    else:
        x, y = x.to(device), y.to(device)
        x_filter, y_filter = x_filter.to(device), y_filter.to(device)
    
    return x, x_filter, y, y_filter

def get_test_batch():

    test_target = np.memmap(os.path.join(data_dir, 'test_target_true.bin'), dtype=loader_dtype, mode='r')
    test_ood = np.memmap(os.path.join(data_dir, 'test_ood_true.bin'), dtype=loader_dtype, mode='r')
    test_parallel = np.memmap(os.path.join(data_dir, 'test_parallel.bin'), dtype=loader_dtype, mode='r')
    test_parallel_hard = np.memmap(os.path.join(data_dir, 'test_parallel_hard.bin'), dtype=loader_dtype, mode='r')

    ix = torch.randint(len(test_target) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((test_target[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((test_target[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_ood) - block_size, (batch_size,))
    x_ood = torch.stack([torch.from_numpy((test_ood[i:i+block_size]).astype(np.int64)) for i in ix])
    y_ood = torch.stack([torch.from_numpy((test_ood[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_parallel) - block_size, (batch_size,))
    x_parallel = torch.stack([torch.from_numpy((test_parallel[i:i+block_size]).astype(np.int64)) for i in ix])
    y_parallel = torch.stack([torch.from_numpy((test_parallel[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    ix = torch.randint(len(test_parallel_hard) - block_size, (batch_size,))
    x_parallel_hard = torch.stack([torch.from_numpy((test_parallel_hard[i:i+block_size]).astype(np.int64)) for i in ix])
    y_parallel_hard = torch.stack([torch.from_numpy((test_parallel_hard[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        x_ood, y_ood = x_ood.pin_memory().to(device, non_blocking=True), y_ood.pin_memory().to(device, non_blocking=True)
        x_parallel, y_parallel = x_parallel.pin_memory().to(device, non_blocking=True), y_parallel.pin_memory().to(device, non_blocking=True)
        x_parallel_hard, y_parallel_hard = x_parallel_hard.pin_memory().to(device, non_blocking=True), y_parallel_hard.pin_memory().to(device, non_blocking=True)

    else:
        x, y = x.to(device), y.to(device)
        x_ood, y_ood = x_ood.to(device), y_ood.to(device)
        x_parallel, y_parallel = x_parallel.to(device), y_parallel.to(device)
        x_parallel_hard, y_parallel_hard = x_parallel_hard.to(device), y_parallel_hard.to(device)

    return x, x_ood, y, y_ood, x_parallel, y_parallel, x_parallel_hard, y_parallel_hard

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
tokens_seen = 0

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    # print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
mup_width_multiplier = n_embd / mup_base_width
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, mup_width_multiplier=mup_width_multiplier,
                  hidden_learning_rate=hidden_learning_rate, embed_learning_rate=embed_learning_rate, scalar_learning_rate=scalar_learning_rate) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    if master_process:
        print("initializing a new model from scratch...")
        print(f"base lr: {hidden_learning_rate} | mup lr: {hidden_learning_rate / mup_width_multiplier}")
    # determine the vocab size we'll use for from-scratch training
    # if meta_vocab_size is None:
        # print("defaulting to vocab_size of GPT-4 to 100256")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 100288 # slightly larger for divby64

    gpt_conf = GPTConfig(**model_args)
    model = GPT(gpt_conf)
    if master_process:
        print(f'training model with {model.get_num_params()} parameters')

elif init_from == 'resume':
    if master_process:
        print(f"resuming training from {out_dir}...")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, f'{wandb_run_name}.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    tokens_seen = checkpoint.get('tokens_seen', 0)
elif init_from.startswith('gpt2'):
    if master_process:
        print(f"initializing from OpenAI GPT-2 weights: {init_from}...")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
# scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, hidden_learning_rate, embed_learning_rate, scalar_learning_rate, (beta1, beta2), device_type, optimizer_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

for param_group in optimizer.param_groups:
    param_group["initial_lr"] = param_group["lr"]

# compile the model
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(it):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, X_filter, Y, Y_filter = get_batch(split, it, mask_threshold=mask_threshold, noise_level=noise_level)
            with ctx:
                logits, loss = model(X, X_filter, Y, Y_filter, remove=remove_tokens)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_test_loss():

    out = {'target' : torch.zeros(eval_iters), 'ood' : torch.zeros(eval_iters), 'parallel' : torch.zeros(eval_iters), 'parallel_hard' : torch.zeros(eval_iters)}
    model.eval()

    for k in range(eval_iters):

        X, X_ood, Y, Y_ood, X_parallel, Y_parallel, X_parallel_hard, Y_parallel_hard = get_test_batch()
        
        with ctx:

            _, loss = model(X, idx_filter=None, targets=Y, targets_filter=None)
            out['target'][k] = loss.item()
        
            _, loss = model(X_ood, idx_filter=None, targets=Y_ood, targets_filter=None)
            out['ood'][k] = loss.item()

            _, loss = model(X_parallel, idx_filter=None, targets=Y_parallel, targets_filter=None)
            out['parallel'][k] = loss.item()

            _, loss = model(X_parallel_hard, idx_filter=None, targets=Y_parallel_hard, targets_filter=None)
            out['parallel_hard'][k] = loss.item()

    model.train()
    return {'target' : out['target'].mean(), 'ood' : out['ood'].mean(), 'parallel' : out['parallel'].mean(), 'parallel_hard' : out['parallel_hard'].mean()}


# learning rate decay scheduler (cosine with warmup)
# since each param group has a diff initial lr, we need to pass it in
def get_lr(init_lr, it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return init_lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (init_lr - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=cfg)

# training loop
X, X_filter, Y, Y_filter = get_batch('train', iter_num, mask_threshold=mask_threshold, noise_level=noise_level) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    # lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        # each param group has a different initial learning rate
        param_group['lr'] = get_lr(param_group['initial_lr'], iter_num) * param_group.get('lr_scale', 1.0) # mup scaling
        # muon momentum + warmup
        if param_group.get('use_muon', False):
            frac = min(iter_num / 300, 1)
            param_group['momentum'] = (1 - frac) * 0.85 + frac * 0.95

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:

        print(f"step {iter_num}: estimating loss on val...")

        losses = estimate_loss(iter_num)

        if eval_test:
            print(f"step {iter_num}: estimating loss on test...")
            test_loss = estimate_test_loss()
        else:
            test_loss = {'target' : 0.0, 'ood' : 0.0, 'parallel' : 0.0, 'parallel_hard' : 0.0}
        
        print('-' * 100)
        print(f"step {iter_num} | train {losses['train']:.4f} | val {losses['val']:.4f} | target {test_loss['target']:.4f} | ood {test_loss['ood']:.4f} | parallel {test_loss['parallel']:.4f} | hard {test_loss['parallel_hard']:.4f}")
        print('-' * 100)
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "tokens_seen": tokens_seen,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "test/target_loss": test_loss['target'],
                "test/ood_loss": test_loss['ood'],
                "test/parallel_loss": test_loss['parallel'],
                "test/parallel_hard_loss": test_loss['parallel_hard'],
                "lr": get_lr(hidden_learning_rate, iter_num),
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'tokens_seen': tokens_seen,
                    'config': cfg,
                }
                print(f"saving checkpoint to {out_dir} ({tokens_seen:,} tokens seen)")
                try:
                    torch.save(checkpoint, os.path.join(out_dir, f'{wandb_run_name}.pt'))
                except Exception as e:
                    print(f"error saving checkpoint: {e}")
                    continue
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, X_filter, Y, Y_filter, remove=remove_tokens)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # count effective tokens (non-masked tokens contribute to loss)
        if mask and iter_num >= begin_filter_step:
            micro_effective = (~Y_filter).sum().item()
        else:
            micro_effective = batch_size * block_size
        tokens_seen += micro_effective * ddp_world_size
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, X_filter, Y, Y_filter = get_batch('train', iter_num, mask_threshold=mask_threshold, noise_level=noise_level)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num} | loss {lossf:.4f} | tokens {tokens_seen:,} | time {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination condition: sync stop decision across ranks to avoid DDP deadlock
    should_stop = torch.tensor([tokens_seen > max_tokens], dtype=torch.int, device=device)
    if ddp:
        torch.distributed.all_reduce(should_stop, op=torch.distributed.ReduceOp.MAX)
    if should_stop.item():
        if master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'tokens_seen': tokens_seen,
                'config': cfg,
            }
            print(f"completed training at {tokens_seen:,} tokens, saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'{wandb_run_name}.pt'))
        break

if ddp:
    destroy_process_group()