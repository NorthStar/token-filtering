"""
pipeline for instruct tuning pretrained models
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
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

sys.path.append('analysis')
from eval_utils import predict_mcq, load_model, build_fewshot_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--model',      type=str, default='mask-51M.pt', help='path to pretrained models')
parser.add_argument('--dataset',    type=str, default='mcq_train_final.jsonl', help='dataset to use')
parser.add_argument('--device',     type=str, default='cuda', help='device to use for computation')
parser.add_argument('--dtype',      type=str, default='bfloat16', help='dtype to use for computation')
parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_PATH, 'gpt'), help='path to pretrained models')
parser.add_argument('--save_path',  type=str, default=os.path.join(MODEL_PATH, 'instruct'), help='path to save model')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_batch',    type=int, default=None)
parser.add_argument('--lr',         type=float, default=1e-6)
parser.add_argument('--dropout',    type=float, default=0.1)
parser.add_argument('--w_decay',    type=float, default=0.01)
parser.add_argument('--beta1',      type=float, default=0.9)
parser.add_argument('--beta2',      type=float, default=0.95)
parser.add_argument('--epochs',     type=int, default=1)
parser.add_argument('--eval_iters', type=int, default=100)
parser.add_argument('--log_int',    type=int, default=100)
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='medical-instruct', help='wandb project name')
args = parser.parse_args()

# Initialize DDP
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
else:
    # vanilla, non-DDP run
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

# Initialize wandb (only on master process)
if not args.no_wandb and master_process:

    name = args.model.split('.')[0] + '-' + str(args.dataset) + '-' + str(args.lr) + '-' + str(args.w_decay)

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
            'world_size': ddp_world_size
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

# load model
def load_model(model_file):
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

if 'remove' in args.model or 'collapse' in args.model:

    enc = tiktoken.Encoding(
        name="cl100k_mask",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|mask|>": 100277
        }
    )

else:
    enc = cl100k_base

def load_mcq_data(data_path, n_fewshot, max_test_samples=None):
    """Load MCQ data from JSONL file"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sample few-shot examples
    fewshot_examples = df.sample(n_fewshot, random_state=42)
    
    # Get test data (excluding few-shot examples)
    test_data = df.drop(fewshot_examples.index.tolist()).reset_index(drop=True)
    
    if max_test_samples:
        total_test_data = []
        for eval_type in test_data['eval'].unique():
            eval_data = test_data[test_data['eval'] == eval_type]
            try:
                eval_data = eval_data.sample(max_test_samples, random_state=42)
            except:
                print(f'not enough samples for {eval_type}')
            total_test_data.append(eval_data)
        test_data = pd.concat(total_test_data)
    
    return test_data, fewshot_examples

def split_question(text):

    question = text.split('Question: ')[1].split('Choices:')[0].strip()
    choices = []
    letters = ['A', 'B', 'C', 'D', 'E']

    for i, letter in enumerate(letters):

        if f' = {letter}' in text:
            if i == 0:
                choice = text.split(f' = {letter}')[0].split('Choice: ')[1].strip()
            else:
                choice = text.split(f' = {letter}')[0].split(f'= {letters[i-1]}\nChoice: ')[1].strip()
            choices.append(torch.tensor(enc.encode(f'Question: {question}\nAnswer: {choice}')).unsqueeze(0).to(device))
    
    return choices

def format_medmcqa_question(question):
    formatted_choices = ""
    for i, option in enumerate(['opa', 'opb', 'opc', 'opd']):
        formatted_choices += f"Choice: {question[option]} = {chr(65 + i)}\n"
    
    return f"""
    Question: {question['question']}\n
    Choices:
    {formatted_choices}
    Answer:""", chr(65 + question['cop']) # 0 -> A, 1 -> B, etc.

def format_medqa_question(question):
    formatted_choices = ""
    for letter in question['options']:
        formatted_choices += f"Choice: {question['options'][letter]} = {letter}\n"
    
    return f"""
    Question: {question['question']}\n
    Choices:
    {formatted_choices}
    Answer:""", question['answer_idx']

pubmed_map = {
    'yes': 'A',
    'maybe': 'B',
    'no': 'C'
}

def format_pubmedqa_question(question):
    formatted_choices = ""
    for letter, choice in zip(['A', 'B', 'C'], ['yes', 'maybe', 'no']):
        formatted_choices += f"Choice: {choice} = {letter}\n"
    
    return f"""
    Context: {' '.join(question['context']['contexts'])}\n
    Question: {question['question']}\n
    Choices:
    {formatted_choices}
    Answer:""", pubmed_map[question['final_decision']]

def load_med_data(val, test, n_fewshot, format_func):
    val = val.shuffle(seed = 42)
    val = val.select(range(min(n_fewshot, len(val))))

    fewshot = {'input': [], 'output': []}
    dset    = {'input': [], 'output': []}

    for i, row in enumerate(val):
        inp, out = format_func(row)
        fewshot['input'].append(inp)
        fewshot['output'].append(out)

    for i, row in enumerate(test):
        inp, out = format_func(row)
        dset['input'].append(inp)
        dset['output'].append(out)

    return dset, fewshot

def load_ade_data(data_path, n_fewshot):
    """Load ADE data from local JSONL file (already formatted)"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Shuffle and split into fewshot and test
    np.random.shuffle(data)
    fewshot_data = data[:n_fewshot]
    test_data = data[n_fewshot:]
    
    fewshot = {'input': [], 'output': []}
    dset = {'input': [], 'output': []}
    
    for item in fewshot_data:
        fewshot['input'].append(item['input'])
        fewshot['output'].append(item['output'])
    
    for item in test_data:
        dset['input'].append(item['input'])
        dset['output'].append(item['output'])
    
    return dset, fewshot

class InstructDataset:
    def __init__(self, data_path, shuffle=True):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        if shuffle:
            np.random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def get_batch(self, idx, batch_size):

        batch = self.data[idx:idx+batch_size]
        batch = [item['input'] + ' ' + item['output'] for item in batch]

        tokens = tokenizer(batch, max_length=2048)

        inputs = tokens['input_ids']
        label_indices = (inputs != 0).sum(dim = 1) - 1

        labels = torch.full_like(inputs, -1)
        batch_indices = torch.arange(len(labels))
        labels[batch_indices, label_indices] = inputs[batch_indices, label_indices]

        inputs = inputs[:, :-1]
        labels = labels[:, 1:]
        
        return inputs, labels

@torch.no_grad()
def estimate_loss(dataset, eval_model=None):
    if eval_model is None:
        eval_model = model
    
    eval_model.eval()
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):

        idx = np.random.randint(0, len(dataset) - batch_size)
        X, Y = dataset.get_batch(idx, batch_size)

        X = X.to(device)
        Y = Y.to(device)

        with ctx:
            _, loss = eval_model(X, idx_filter=None, targets=Y, targets_filter=None)
        
        losses[k] = loss.mean().item()
    
    eval_model.train()
    return losses.mean()

@torch.no_grad()
def compute_train_accuracy(X, Y, eval_model=None):
    """Compute accuracy on the current training batch"""
    if eval_model is None:
        eval_model = model
    
    eval_model.eval()
    
    with ctx:
        logits, _ = eval_model(X, idx_filter=None, targets=None, targets_filter=None)
    
    # Get predictions for positions where we have labels
    mask = Y != -1
    if mask.sum() == 0:
        eval_model.train()
        return 0.0
    
    predictions = logits.argmax(dim=-1)
    correct = (predictions == Y) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    eval_model.train()
    return accuracy.item()

@torch.no_grad()
def evaluate_accuracy(model, test_data, fewshot_examples):
    """Evaluate model accuracy by checking if output token matches expected answer"""
    results = {}
    all_answers = []
    
    # Group test data by eval type
    eval_groups = test_data.groupby('eval')
    
    for eval_type, eval_data in eval_groups:
        # Use few-shot examples from the same eval type
        eval_fewshot = fewshot_examples[fewshot_examples['eval'] == eval_type]
        if len(eval_fewshot) == 0:
            # Fallback to any few-shot examples if none available for this eval type
            eval_fewshot = fewshot_examples.head(min(4, len(fewshot_examples)))
        
        fewshot_prompt = build_fewshot_prompt(eval_fewshot.to_dict(orient='list'))
        
        correct = 0
        total = 0
        eval_answers = []
        
        for _, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc=f"evaluating {eval_type}", leave=False):
                        
            try:
                predicted_letter, _ = predict_mcq(model, row['input'], fewshot_prompt, enc, device, ctx)
            except Exception as e:
                print(f"error predicting mcq for {row['input']}: {e}")
                continue
            
            # check if correct
            expected_answer = row['output'].strip()
            if predicted_letter == expected_answer:
                correct += 1
            
            total += 1
            eval_answers.append(predicted_letter)
            all_answers.append({'eval': eval_type, 'output': predicted_letter})
        
        if total < 10:
            print('too few samples for benchmark', eval_type)
            continue

        accuracy = correct / total if total > 0 else 0
        results[eval_type] = accuracy
    
    return results, all_answers

@torch.no_grad()
def evaluate_string_accuracy(model, test_data, fewshot_examples):
    """Evaluate model accuracy by checking if logprobs of expected answer are higher than other options"""
    model.eval()

    results = {}
    all_answers = []
    
    # Group test data by eval type
    eval_groups = test_data.groupby('eval')
    letters = ['A', 'B', 'C', 'D', 'E']
    
    for eval_type, eval_data in eval_groups:
        if eval_type != 'arc easy':
            continue

        # Use few-shot examples from the same eval type
        eval_fewshot = fewshot_examples[fewshot_examples['eval'] == eval_type]
        if len(eval_fewshot) == 0:
            # Fallback to any few-shot examples if none available for this eval type
            eval_fewshot = fewshot_examples.head(min(4, len(fewshot_examples)))
        
        fewshot_prompt = build_fewshot_prompt(eval_fewshot.to_dict(orient='list'))
        
        correct = 0
        total = 0
        eval_answers = []
        
        for i, row in tqdm(eval_data.iterrows(), total=len(eval_data), desc=f"evaluating {eval_type}", leave=False):
                        
            choices = split_question(row['input'])

            # if i % 10 == 0:
            #     print(enc.decode(model.generate(choices[0], 10).detach().cpu().tolist()[0]))

            losses = dict()

            with ctx:

                for letter, choice in zip(letters[:len(choices)], choices):
                    total_logprob = 0.0
                    choice_tokens = choice.squeeze(0)  # Remove batch dimension
                    
                    # Compute log probability for the entire choice sequence
                    # Get logits for the entire sequence at once
                    logits, _ = model(choice_tokens.unsqueeze(0), idx_filter=None, targets=choice_tokens.unsqueeze(0), targets_filter=None)
                    logits = logits.squeeze(0)

                    # Compute log probabilities for each token position (except the first)
                    for i in range(1, len(choice_tokens)):
                        token_logits = logits[i-1, :]  # Logits at position i-1 predict token at position i
                        log_probs = F.log_softmax(token_logits, dim=-1)
                        token_id = choice_tokens[i].item()
                        total_logprob += log_probs[token_id].item()
                    
                    # Store negative log probability (lower is better, like loss)
                    losses[letter] = -total_logprob
            
            predicted_letter = min(losses, key=losses.get)
            expected_answer = row['output'].strip()
            if predicted_letter == expected_answer:
                correct += 1
            
            total += 1
            eval_answers.append(predicted_letter)
            all_answers.append({'eval': eval_type, 'output': predicted_letter})
        
        if total < 10:
            print('too few samples for benchmark', eval_type)
            continue
        
        accuracy = correct / total if total > 0 else 0
        results[eval_type] = accuracy
    
    return results, all_answers

@torch.no_grad()
def evaluate_string_accuracy_medical(model, dataset_name, data, fewshot, n_samples=100):
    """Evaluate model string accuracy on medical datasets by comparing log probabilities"""
    model.eval()
    
    # Sample random subset
    indices = np.random.choice(len(data['input']), min(n_samples, len(data['input'])), replace=False)
    
    # Build fewshot prompt
    fewshot_prompt = build_fewshot_prompt(fewshot)
    
    correct = 0
    total = 0
    
    for idx in tqdm(indices, desc=f"String accuracy for {dataset_name}", leave=False):
        question_text = data['input'][idx]
        expected_answer = data['output'][idx]
        
        # Extract choices from the question format
        # Format is: "Question: ... Choices: Choice: ... = A\n Choice: ... = B\n ... Answer:"
        choices_section = question_text.split('Choices:')[1].split('Answer:')[0].strip()
        choice_lines = [line.strip() for line in choices_section.split('\n') if line.strip() and 'Choice:' in line]
        
        # Extract letters and text
        choice_map = {}
        for line in choice_lines:
            # Format: "Choice: text = X"
            parts = line.split(' = ')
            if len(parts) == 2:
                letter = parts[1].strip()
                choice_text = parts[0].replace('Choice:', '').strip()
                choice_map[letter] = choice_text
        
        if not choice_map:
            continue
        
        # Compute log probabilities for each choice
        losses = {}
        
        with ctx:
            for letter, choice_text in choice_map.items():
                # Format question with this choice as the answer
                full_text = question_text.split('Answer:')[0] + f'Answer: {choice_text}'
                tokens = torch.tensor(enc.encode(full_text)).unsqueeze(0).to(device)
                
                if tokens.shape[1] > 2048:  # Skip if too long
                    continue
                
                # Compute log probability
                total_logprob = 0.0
                choice_tokens = tokens.squeeze(0)
                
                logits, _ = model(tokens, idx_filter=None, targets=tokens, targets_filter=None)
                logits = logits.squeeze(0)
                
                # Only compute log prob for the answer tokens
                # Find where "Answer:" starts to only evaluate the answer portion
                answer_start_text = question_text.split('Answer:')[0] + 'Answer:'
                answer_start_tokens = enc.encode(answer_start_text)
                start_idx = len(answer_start_tokens)
                
                for i in range(start_idx, len(choice_tokens)):
                    token_logits = logits[i-1, :]
                    log_probs = F.log_softmax(token_logits, dim=-1)
                    token_id = choice_tokens[i].item()
                    total_logprob += log_probs[token_id].item()
                
                losses[letter] = -total_logprob
        
        if not losses:
            continue
        
        # Predict the letter with lowest loss (highest log probability)
        predicted_letter = min(losses, key=losses.get)
        
        if predicted_letter == expected_answer:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

@torch.no_grad()
def evaluate_medical_datasets(model, medqa_data, medqa_fewshot, medmcqa_data, medmcqa_fewshot, pubmedqa_data, pubmedqa_fewshot, ade_data, ade_fewshot, n_samples=100):
    """Evaluate model on a random subset of medical datasets"""
    model.eval()
    
    results = {}
    
    # Sample random subsets
    medqa_indices = np.random.choice(len(medqa_data['input']), min(n_samples, len(medqa_data['input'])), replace=False)
    medmcqa_indices = np.random.choice(len(medmcqa_data['input']), min(n_samples, len(medmcqa_data['input'])), replace=False)
    pubmedqa_indices = np.random.choice(len(pubmedqa_data['input']), min(n_samples, len(pubmedqa_data['input'])), replace=False)
    ade_indices = np.random.choice(len(ade_data['input']), min(n_samples, len(ade_data['input'])), replace=False)
    
    # Build fewshot prompts
    medqa_prompt = build_fewshot_prompt(medqa_fewshot)
    medmcqa_prompt = build_fewshot_prompt(medmcqa_fewshot)
    pubmedqa_prompt = build_fewshot_prompt(pubmedqa_fewshot)
    ade_prompt = build_fewshot_prompt(ade_fewshot)
    
    # Evaluate MedQA
    correct_medqa = 0
    for idx in medqa_indices:
        try:
            predicted_letter, _ = predict_mcq(model, medqa_data['input'][idx], medqa_prompt, enc, device, ctx)
            if predicted_letter == medqa_data['output'][idx]:
                correct_medqa += 1
        except Exception as e:
            print(f"error evaluating medqa sample {idx}: {e}")
            continue
    
    results['medqa'] = correct_medqa / len(medqa_indices) if len(medqa_indices) > 0 else 0
    
    # Evaluate MedMCQA
    correct_medmcqa = 0
    for idx in medmcqa_indices:
        try:
            predicted_letter, _ = predict_mcq(model, medmcqa_data['input'][idx], medmcqa_prompt, enc, device, ctx)
            if predicted_letter == medmcqa_data['output'][idx]:
                correct_medmcqa += 1
        except Exception as e:
            print(f"error evaluating medmcqa sample {idx}: {e}")
            continue
    
    results['medmcqa'] = correct_medmcqa / len(medmcqa_indices) if len(medmcqa_indices) > 0 else 0
    
    # Evaluate PubMedQA
    correct_pubmedqa = 0
    for idx in pubmedqa_indices:
        try:
            predicted_letter, _ = predict_mcq(model, pubmedqa_data['input'][idx], pubmedqa_prompt, enc, device, ctx)
            if predicted_letter == pubmedqa_data['output'][idx]:
                correct_pubmedqa += 1
        except Exception as e:
            print(f"error evaluating pubmedqa sample {idx}: {e}")
            continue
    
    results['pubmedqa'] = correct_pubmedqa / len(pubmedqa_indices) if len(pubmedqa_indices) > 0 else 0
    
    # Evaluate ADE
    correct_ade = 0
    for idx in ade_indices:
        try:
            predicted_letter, _ = predict_mcq(model, ade_data['input'][idx], ade_prompt, enc, device, ctx)
            if predicted_letter == ade_data['output'][idx]:
                correct_ade += 1
        except Exception as e:
            print(f"error evaluating ade sample {idx}: {e}")
            continue
    
    results['ade'] = correct_ade / len(ade_indices) if len(ade_indices) > 0 else 0
    
    model.train()
    return results


scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# Get raw model for optimizer configuration and evaluation (unwrap DDP if needed)
raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(w_decay, lr, 0.6, 0.04, (beta1, beta2), device_type, optimizer_type='adamw')
# optimizer = raw_model.configure_optimizers(w_decay, lr, (beta1, beta2), device_type, optimizer_type='adamw')

train = InstructDataset(os.path.join('data', args.dataset), shuffle=True)
test_instruct = InstructDataset(os.path.join('data', 'mcq_test_small.jsonl'), shuffle=True)
test, fewshot = load_mcq_data(f'data/mcq_test_small.jsonl', 0, 100)

# Load medical datasets
if master_process:
    print("Loading medical datasets...")
    medmcqa_ignore = ['Biochemistry', 'Microbiology', 'Psychiatry', 'Social & Preventive Medicine']
    
    medqa_val  = load_dataset("GBaker/MedQA-USMLE-4-options", split='test')
    medqa_test = load_dataset("GBaker/MedQA-USMLE-4-options", split='train')
    
    medmcqa_val  = load_dataset("openlifescienceai/medmcqa", split='validation')
    medmcqa_test = load_dataset("openlifescienceai/medmcqa", split='train')
    medmcqa_test = medmcqa_test.filter(lambda x: x['subject_name'] not in medmcqa_ignore)
    medmcqa_test = medmcqa_test.select(range(min(10000, len(medmcqa_test)))) # limit size
    
    pubmedqa_val   = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train')
    pubmedqa_test  = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train')
    
    medqa_data, medqa_fewshot = load_med_data(medqa_val, medqa_test, 0, format_medqa_question)
    medmcqa_data, medmcqa_fewshot = load_med_data(medmcqa_val, medmcqa_test, 0, format_medmcqa_question)
    pubmedqa_data, pubmedqa_fewshot = load_med_data(pubmedqa_val, pubmedqa_test, 0, format_pubmedqa_question)
    ade_data, ade_fewshot = load_ade_data('data/ade.jsonl', 0)
    
    print(f"Loaded MedQA: {len(medqa_data['input'])} samples")
    print(f"Loaded MedMCQA: {len(medmcqa_data['input'])} samples")
    print(f"Loaded PubMedQA: {len(pubmedqa_data['input'])} samples")
    print(f"Loaded ADE: {len(ade_data['input'])} samples")

results = []

# Only evaluate on master process using raw_model (no DDP sync needed)
if master_process:
    string_acc, _ = evaluate_string_accuracy(raw_model, test, fewshot)
    print(string_acc)
    
    # Compute string accuracy for medical datasets
    medqa_string_acc = evaluate_string_accuracy_medical(raw_model, "MedQA", medqa_data, medqa_fewshot, n_samples=1000)
    medmcqa_string_acc = evaluate_string_accuracy_medical(raw_model, "MedMCQA", medmcqa_data, medmcqa_fewshot, n_samples=1000)
    pubmedqa_string_acc = evaluate_string_accuracy_medical(raw_model, "PubMedQA", pubmedqa_data, pubmedqa_fewshot, n_samples=1000)
    print(f"MedQA string accuracy: {medqa_string_acc:.4f}")
    print(f"MedMCQA string accuracy: {medmcqa_string_acc:.4f}")
    print(f"PubMedQA string accuracy: {pubmedqa_string_acc:.4f}")
    
    loss = estimate_loss(train, raw_model)
    test_loss = estimate_loss(test_instruct, raw_model)

    eval_accs, _ = evaluate_accuracy(raw_model, test, fewshot)
    acc = sum(eval_accs.values()) / len(eval_accs)
    arc_easy_acc = eval_accs.get('arc easy', 0.0)
    print(f"Initial ARC Easy accuracy: {arc_easy_acc:.4f}")
    
    # Evaluate on medical datasets
    med_results = evaluate_medical_datasets(raw_model, medqa_data, medqa_fewshot, medmcqa_data, medmcqa_fewshot, pubmedqa_data, pubmedqa_fewshot, ade_data, ade_fewshot, n_samples=1000)
    print(f"Initial MedQA accuracy: {med_results['medqa']:.4f}")
    print(f"Initial MedMCQA accuracy: {med_results['medmcqa']:.4f}")
    print(f"Initial PubMedQA accuracy: {med_results['pubmedqa']:.4f}")
    print(f"Initial ADE accuracy: {med_results['ade']:.4f}")

    if not args.no_wandb:
        wandb.log({
            'train/loss': loss,
            'test/loss': test_loss,
            'iteration': 0,
            'test/accuracy': acc,
            'test/arc_easy_accuracy': arc_easy_acc,
            'test/medqa_accuracy': med_results['medqa'],
            'test/medmcqa_accuracy': med_results['medmcqa'],
            'test/pubmedqa_accuracy': med_results['pubmedqa'],
            'test/ade_accuracy': med_results['ade']
        })

    results.append({
        'samples_seen': 0,
        'loss': loss,
        'test_loss': test_loss,
        'accuracy': acc,
        'arc_easy_accuracy': arc_easy_acc,
        'medqa_accuracy': med_results['medqa'],
        'medmcqa_accuracy': med_results['medmcqa'],
        'pubmedqa_accuracy': med_results['pubmedqa'],
        'ade_accuracy': med_results['ade']
    })

# Synchronize all processes before starting training
if ddp:
    dist.barrier()

model.train()
total_loss = 0
num_batches = 0

if args.n_batch is None:
    n_batch = len(train) // batch_size
else:
    n_batch = args.n_batch

for batch_idx in range(0, n_batch * batch_size, batch_size):

    X, Y = train.get_batch(batch_idx, batch_size)
    X = X.to(device)
    Y = Y.to(device)

    with ctx:
        _, loss = model(X, idx_filter=None, targets=Y, targets_filter=None)
    
    total_loss += loss.item()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Compute train accuracy on the batch we just trained on (only on master)
    if master_process:
        train_acc = compute_train_accuracy(X, Y, raw_model)
    else:
        train_acc = 0.0

    num_batches += 1

    # Log training loss (only on master process)
    if num_batches % args.log_int == 0:
        
        # Synchronize before evaluation
        if ddp:
            dist.barrier()
        
        if master_process:
            test_loss = estimate_loss(test_instruct, raw_model)
            eval_accs, _ = evaluate_accuracy(raw_model, test, fewshot)
            acc = sum(eval_accs.values()) / len(eval_accs)
            arc_easy_acc = eval_accs.get('arc easy', 0.0)
            
            # Evaluate on medical datasets
            med_results = evaluate_medical_datasets(raw_model, medqa_data, medqa_fewshot, medmcqa_data, medmcqa_fewshot, pubmedqa_data, pubmedqa_fewshot, ade_data, ade_fewshot, n_samples=1000)

            if not args.no_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/accuracy': train_acc,
                    'test/loss': test_loss,
                    'iteration': batch_idx,
                    'test/accuracy': acc,
                    'test/medqa_accuracy': med_results['medqa'],
                    'test/medmcqa_accuracy': med_results['medmcqa'],
                    'test/pubmedqa_accuracy': med_results['pubmedqa'],
                    'test/ade_accuracy': med_results['ade'],
                    'test/arc_easy_accuracy': arc_easy_acc
                })
            
            print(f"batch {num_batches} | train_loss {loss.item():.4f} | train_acc {train_acc:.4f} | test_loss {test_loss:.4f} | test_acc {acc:.4f} | medqa {med_results['medqa']:.4f} | medmcqa {med_results['medmcqa']:.4f} | pubmedqa {med_results['pubmedqa']:.4f} | ade {med_results['ade']:.4f} | arc_easy {arc_easy_acc:.4f}")

            results.append({
                'samples_seen': batch_idx,
                'loss': loss.item(),
                'train_accuracy': train_acc,
                'test_loss': test_loss,
                'accuracy': acc,
                'medqa_accuracy': med_results['medqa'],
                'medmcqa_accuracy': med_results['medmcqa'],
                'pubmedqa_accuracy': med_results['pubmedqa'],
                'ade_accuracy': med_results['ade'],
                'arc_easy_accuracy': arc_easy_acc
            })
        
        # Synchronize after evaluation
        if ddp:
            dist.barrier()

# Save results and checkpoint (only on master process)
if master_process:
    df = pd.DataFrame(results)
    df.to_csv(os.path.join('analysis/results/', 'instruct.csv'), index=False)

    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
    }

    model_name = args.model.split('.')[0] + '-instruct'
    print(f"completed training, saving checkpoint as {model_name}.pt")
    torch.save(checkpoint, os.path.join(save_path, f'{model_name}.pt'))

    if not args.no_wandb:
        wandb.finish()

# Cleanup DDP
if ddp:
    dist.destroy_process_group()