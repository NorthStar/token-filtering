"""
train RoBERTa on documents from English and French fineweb
  - everything here is setup w/ huggingface infra bc that's easy
  - this just saves a slice of fineweb to your machine, no streaming (otherwise ddp breaks)
"""
# uv run torchrun --standalone --nproc_per_node=4 roberta.py

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import math

import os
import glob
import requests
import tiktoken
import numpy as np
import pathlib
import sys
import tiktoken
from tiktokenizer import TikTokenizer
from contextlib import nullcontext
from omegaconf import OmegaConf

from muon import MuonWithAuxAdam

# evil config magic
cfg_file = 'roberta.yaml'
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

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    gradient_accumulation_steps *= ddp_world_size
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size

else:
    # if not ddp, we are running on a single gpu, and one process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in str(device) else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data configuration
data_dir = os.path.join(data_path, dataset)
loader_dtype = np.uint32

enc = tiktoken.get_encoding('cl100k_base')
mask_token_id = enc.eot_token

def get_batch(split):

    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=loader_dtype, mode='r')
    max_start_idx = len(data) - block_size
        
    ix = torch.randint(max_start_idx, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    # create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (x != 0).long()
    
    # apply MLM masking
    input_ids = x.clone()
    labels = x.clone()
    
    # create mask for 15% of tokens (excluding pad tokens)
    probability_matrix = torch.full(input_ids.shape, 0.15)
    mask_positions = torch.bernoulli(probability_matrix).bool() & (input_ids != 0)
    
    # replace masked tokens with mask token
    input_ids[mask_positions] = mask_token_id
    
    # set labels to -100 for non-masked tokens (ignored in loss)
    labels[~mask_positions] = -100
    
    if device_type == 'cuda':
        # pin arrays for async GPU transfer
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
    else:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# initialize wandb
if master_process:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            'block_size': block_size,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'lr': hidden_learning_rate,
            'max_iters': max_iters,
            'batch_size': batch_size,
            'model_type': 'roberta',
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'ddp_world_size': ddp_world_size,
            'dtype': dtype,
            'max_iters': max_iters,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'grad_clip': grad_clip,
        }
    )

# setup model
enc = tiktoken.get_encoding('cl100k_base')
config = RobertaConfig(
    vocab_size=enc.n_vocab,
    max_position_embeddings=block_size + 2, 
    num_attention_heads=num_heads,
    num_hidden_layers=num_layers,
    hidden_size=hidden_size,
    type_vocab_size=1,
    pad_token_id=0,
    bos_token_id=enc.eot_token,
    eos_token_id=enc.eot_token,
    mask_token_id=enc.eot_token
)

if resume_from_checkpoint:
    model = RobertaForMaskedLM.from_pretrained(os.path.join(model_path, resume_from_checkpoint))
    iter_num = last_iter

else:
    model = RobertaForMaskedLM(config)
    iter_num = 0

model.to(device)
tokenizer = TikTokenizer()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

print(f'loading data from {data_dir}...')

# setup training
def configure_optimizers(model, weight_decay, hidden_learning_rate, embed_learning_rate, scalar_learning_rate, betas, device_type, optimizer_type='adamw'):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    mup_decay_params = []
    decay_params = []
    nodecay_params = []
    embed_params = []
    for n, p in param_dict.items():
        if p.dim() >= 2 and 'embed' not in n:
            if n.endswith('c_attn.weight') or n.endswith('c_fc.weight') or n.endswith('c_proj.weight'):
                mup_decay_params.append(p)
            else:
                decay_params.append(p)
        elif p.dim() >= 2 and 'embed' in n:
            if optimizer_type == 'muon':
                embed_params.append(p)
            else:
                decay_params.append(p)
        else:
            nodecay_params.append(p)
    
    if optimizer_type == 'adamw':

        optim_groups = [
            {'params': mup_decay_params, 'weight_decay': weight_decay, 'lr': hidden_learning_rate, 'lr_scale': 1},
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': hidden_learning_rate, 'lr_scale': 1},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': hidden_learning_rate, 'lr_scale': 1}
        ]

        use_fused = device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=hidden_learning_rate, betas=betas, **extra_args)
    elif optimizer_type == 'soap':

        optim_groups = [
            {'params': mup_decay_params, 'weight_decay': weight_decay, 'lr': hidden_learning_rate, 'lr_scale': 1},
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': hidden_learning_rate, 'lr_scale': 1},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': hidden_learning_rate, 'lr_scale': 1}
        ]

        from soap import SOAP
        optimizer = SOAP(optim_groups, lr=hidden_learning_rate, betas=betas)
    elif 'muon' in optimizer_type:

        # we do not mup scale!
        optim_groups = [
            {'params': mup_decay_params + decay_params, 'weight_decay': weight_decay, 'use_muon': True, 'lr': hidden_learning_rate, 'lr_scale': 1},
            {'params': embed_params, 'weight_decay': weight_decay, 'use_muon': False, 'lr': embed_learning_rate, 'betas': betas, 'lr_scale': 1},
            {'params': nodecay_params, 'weight_decay': 0.0, 'use_muon': False, 'lr': scalar_learning_rate, 'betas': betas, 'lr_scale': 1}
        ]

        if optimizer_type == 'muon':

            from muon import MuonWithAuxAdam
            optimizer = MuonWithAuxAdam(optim_groups)
        
        if optimizer_type == 'muonsingle':
            
            from muon import SingleDeviceMuonWithAuxAdam
            optimizer = SingleDeviceMuonWithAuxAdam(optim_groups)
        
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")
    # print(f"using fused AdamW: {use_fused}")

    return optimizer
# optimizer = SOAP(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

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

# unwrap model for saving checkpoints
raw_model = model.module if ddp else model

# configure optimizer
optimizer = configure_optimizers(model, weight_decay, hidden_learning_rate, embed_learning_rate, scalar_learning_rate, (beta1, beta2), device_type, optimizer_type)
for param_group in optimizer.param_groups:
    param_group["initial_lr"] = param_group["lr"]

# training parameters
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

print(f'training...')
model.train()

# training loop
while iter_num < max_iters:

    for param_group in optimizer.param_groups:
        # each param group has a different initial learning rate
        param_group['lr'] = get_lr(param_group['initial_lr'], iter_num) * param_group.get('lr_scale', 1.0) # mup scaling
        # muon momentum + warmup
        if param_group.get('use_muon', False):
            frac = min(iter_num / 300, 1)
            param_group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    
    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        batch = get_batch('train')
        
        with ctx:
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        
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
    
    iter_num += 1

    if master_process:
        # log training metrics        
        # periodic evaluation during training

        if iter_num % 10 == 0:

            wandb.log({
                'train/loss': loss.item() * gradient_accumulation_steps,
                'iter': iter_num
            })

        if iter_num % eval_interval == 0:
            
            model.eval()
            eval_loss = 0

            with torch.no_grad():
                for _ in range(eval_iters):  # evaluate on 100 batches
                    eval_batch = get_batch('test')
                    with ctx:
                        outputs = model(**eval_batch)
                    eval_loss += outputs.loss.item()
            
            avg_eval_loss = eval_loss / eval_iters
            wandb.log({
                'val/loss': avg_eval_loss,
                'iter': iter_num
            })
            
            print(f'step {iter_num}: train loss: {loss.item() * gradient_accumulation_steps:.3f} | val loss: {avg_eval_loss:.3f}')
            model.train()
        
        # save model periodically
        if iter_num % 1000 == 0:
            os.makedirs(model_path, exist_ok=True)
            raw_model.save_pretrained(os.path.join(model_path, 'roberta-pubmed'))

# final model save
if master_process:
    os.makedirs(model_path, exist_ok=True)
    raw_model.save_pretrained(os.path.join(model_path, 'roberta-pubmed'))

if ddp:
    destroy_process_group()
    
# finish wandb run
if master_process:
    wandb.finish()