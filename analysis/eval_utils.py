"""
evaluates downstream performance of models on mcq_test, few-shot
"""
import os
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import numpy as np
from plotnine import *
import tiktoken
import json
from tqdm import tqdm
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT

def load_model(model_file, device):
    print(f"loading model: {model_file}")
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
    model.to(device)
    model.eval()
    return model

def build_fewshot_prompt(examples):
    """
    build few-shot prompt from examples

    inputs:
        examples: dict of lists w/ keys 'input' and 'output'

    outputs:
        prompt: single string w/ few-shot examples
    """
    prompt = ""
    for i in range(len(examples['input'])):
        prompt += examples['input'][i] + " " + examples['output'][i] + "\n\n"
    return prompt

@torch.no_grad()
def predict_mcq(model, question_and_choices, fewshot_prompt, enc, device, ctx, return_log_probs=False):
    """
    Predict mcq answers for a given model and test data.
    
    Returns: 
        predicted_letter: str, the predicted answer (A/B/C/D)
        truncated: bool, whether prompt was truncated
        answer_nll: dict, negative log likelihood for each choice (lower is better)
    """
    prompt = fewshot_prompt + question_and_choices
    
    # encode prompt
    truncated = False
    prompt_tokens = enc.encode(prompt)#, allowed_special={'<|endoftext|>'})
    
    if len(prompt_tokens) > 2048:
        prompt_tokens = prompt_tokens[-2048:]
        truncated = True
    
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # generate next token
    with ctx:
        logits, _ = model(input_tensor)
        
        # get logits for the last position
        next_token_logits = logits[0, -1, :]
        predicted_token_id = torch.argmax(next_token_logits).item()
        
        # Also compute log probabilities for answer choices
        log_probs = F.log_softmax(next_token_logits, dim=-1)
    
    # decode predicted token
    predicted_token = enc.decode([int(predicted_token_id)])
    
    # extract just the letter (in case there are spaces or other characters)
    predicted_letter = predicted_token.strip()
    if predicted_letter and predicted_letter[0] in ['A', 'B', 'C', 'D']:
        predicted_letter = predicted_letter[0]
    else:
        predicted_letter = 'X' # invalid prediction
    
    # Get negative log likelihoods for each choice (A, B, C, D)
    # NLL = -log_prob, where log_prob is negative, so NLL is positive
    # Lower NLL = higher confidence
    choice_tokens = {
        'A': enc.encode(' A')[0],
        'B': enc.encode(' B')[0],
        'C': enc.encode(' C')[0],
        'D': enc.encode(' D')[0],
    }
    
    answer_nll = {}
    for letter, token_id in choice_tokens.items():
        answer_nll[letter] = -log_probs[token_id].item()
    
    if return_log_probs:
        return predicted_letter, truncated, answer_nll
    else:
        return predicted_letter, truncated

@torch.no_grad()
def predict_mcq_batch(model, questions_and_choices, fewshot_prompt, enc, device, ctx, return_log_probs=False):
    """
    Predict mcq answers for a batch of questions.
    
    Args:
        model: GPT model
        questions_and_choices: list of question strings
        fewshot_prompt: few-shot prompt prefix
        enc: tokenizer
        device: compute device
        ctx: autocast context
        return_log_probs: whether to return negative log likelihoods
        
    Returns: 
        predicted_letters: list of str, the predicted answers (A/B/C/D)
        truncated_list: list of bool, whether each prompt was truncated
        answer_nlls: list of dict, negative log likelihood for each choice (if return_log_probs)
    """
    batch_size = len(questions_and_choices)
    
    # Encode all prompts
    all_tokens = []
    truncated_list = []
    
    for question in questions_and_choices:
        prompt = fewshot_prompt + question
        prompt_tokens = enc.encode(prompt)#, allowed_special={'<|endoftext|>'})
        
        if len(prompt_tokens) > 2048:
            prompt_tokens = prompt_tokens[-2048:]
            truncated_list.append(True)
        else:
            truncated_list.append(False)
        
        all_tokens.append(prompt_tokens)
    
    # Pad sequences to same length
    max_len = max(len(tokens) for tokens in all_tokens)
    padded_tokens = []
    
    for tokens in all_tokens:
        padding_len = max_len - len(tokens)
        # Pad on the left (prepend padding) with endoftext token
        padded = [enc.encode('<|endoftext|>')[0]] * padding_len + tokens
        padded_tokens.append(padded)
    
    input_tensor = torch.tensor(padded_tokens, dtype=torch.long, device=device)
    
    # Generate next tokens for batch
    with ctx:
        logits, _ = model(input_tensor)
        
        # Get logits for the last position of each sequence
        # Use -1 to get the last position, which works regardless of actual sequence length
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        
        predicted_token_ids = torch.argmax(next_token_logits, dim=-1).cpu().tolist()
        
        # Compute log probabilities for answer choices
        log_probs_batch = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
    
    # Get choice tokens
    choice_tokens = {
        'A': enc.encode(' A')[0],
        'B': enc.encode(' B')[0],
        'C': enc.encode(' C')[0],
        'D': enc.encode(' D')[0],
    }
    
    # Process predictions
    predicted_letters = []
    answer_nlls = []
    
    for i in range(batch_size):
        # Decode predicted token
        predicted_token = enc.decode([predicted_token_ids[i]])
        predicted_letter = predicted_token.strip()
        
        if predicted_letter and predicted_letter[0] in ['A', 'B', 'C', 'D']:
            predicted_letter = predicted_letter[0]
        else:
            predicted_letter = 'X'
        
        predicted_letters.append(predicted_letter)
        
        # Get NLLs for this sample
        if return_log_probs:
            answer_nll = {}
            for letter, token_id in choice_tokens.items():
                answer_nll[letter] = -log_probs_batch[i, token_id].item()
            answer_nlls.append(answer_nll)
    
    if return_log_probs:
        return predicted_letters, truncated_list, answer_nlls
    else:
        return predicted_letters, truncated_list

@torch.no_grad()
def predict_mcq_cloze(model, question_and_choices, fewshot_prompt, enc, device, ctx):
    """predict mcq answers for a given model and test data"""

    try:
        question = question_and_choices.split("Question: ")[1].split("Choices:")[0].strip()
        choices = question_and_choices.split('Answer:')[0].split("Choices:")[1].strip().split("\nChoice:")
        choices = [c.split('=')[0].strip() for c in choices]
        choices[0] = choices[0].split('Choice:')[1].strip()
    except ValueError:
        print(question_and_choices)
        print('question_and_choices format is incorrect')
        return 'X', True

    letters = ['A', 'B', 'C', 'D', 'E']

    logprobs = dict()

    for letter, choice in zip(letters[:len(choices)], choices):

        # compute logprobs
        prompt = f'Question: {question}\nAnswer: {choice}'
        prompt_tokens = enc.encode(prompt)#, allowed_special={'<|endoftext|>'})
        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        truncated = False
        if len(prompt_tokens) > 2048:
            prompt_tokens = prompt_tokens[-2048:]
            truncated = True

        total_logprob = 0.0        
        logits, _ = model(input_tensor, idx_filter=None, targets=input_tensor, targets_filter=None)
        logits = logits.squeeze(0)
        
        for i in range(1, len(prompt_tokens)):
            token_logits = logits[i - 1, :] 
            log_probs = F.log_softmax(token_logits, dim=-1)
            token_id = prompt_tokens[i]
            total_logprob += log_probs[token_id].item()
        
        logprobs[letter] = -total_logprob

    predicted_choice = min(logprobs, key=logprobs.get)

    return predicted_choice, truncated

@torch.no_grad()
def predict_mcq_cloze_batch(model, questions_and_choices, fewshot_prompt, enc, device, ctx, batch_size=32):
    """
    Predict mcq answers for a batch of questions using cloze evaluation.
    
    Args:
        model: GPT model
        questions_and_choices: list of question strings
        fewshot_prompt: few-shot prompt prefix (not used in cloze)
        enc: tokenizer
        device: compute device
        ctx: autocast context
        batch_size: batch size for processing choices
        
    Returns:
        predicted_choices: list of str, the predicted answers (A/B/C/D/E)
        truncated_list: list of bool, whether any choice was truncated for each question
    """
    predicted_choices = []
    truncated_list = []
    
    for question_and_choices in questions_and_choices:
        try:
            question = question_and_choices.split("Question: ")[1].split("Choices:")[0].strip()
            choices = question_and_choices.split('Answer:')[0].split("Choices:")[1].strip().split("\nChoice:")
            choices = [c.split('=')[0].strip() for c in choices]
            choices[0] = choices[0].split('Choice:')[1].strip()
        except ValueError:
            print(question_and_choices)
            print('question_and_choices format is incorrect')
            predicted_choices.append('X')
            truncated_list.append(True)
            continue
        
        letters = ['A', 'B', 'C', 'D', 'E']
        logprobs = dict()
        any_truncated = False
        
        # Process choices in batches
        for batch_start in range(0, len(choices), batch_size):
            batch_end = min(batch_start + batch_size, len(choices))
            batch_choices = choices[batch_start:batch_end]
            batch_letters = letters[batch_start:batch_end]
            
            # Tokenize all choices in this batch
            all_tokens = []
            for choice in batch_choices:
                prompt = f'Question: {question}\nAnswer: {choice}'
                prompt_tokens = enc.encode(prompt)#, allowed_special={'<|endoftext|>'})
                
                if len(prompt_tokens) > 2048:
                    prompt_tokens = prompt_tokens[-2048:]
                    any_truncated = True
                
                all_tokens.append(prompt_tokens)
            
            # Pad sequences to same length in this batch
            max_len = max(len(tokens) for tokens in all_tokens)
            padded_tokens = []
            
            for tokens in all_tokens:
                padding_len = max_len - len(tokens)
                padded = [enc.encode('<|endoftext|>')[0]] * padding_len + tokens
                padded_tokens.append(padded)
            
            input_tensor = torch.tensor(padded_tokens, dtype=torch.long, device=device)
            
            # Get logits for all sequences in batch
            with ctx:
                logits, _ = model(input_tensor, idx_filter=None, targets=input_tensor, targets_filter=None)
            
            # Get actual sequence length from logits (might be less than max_len)
            actual_seq_len = logits.shape[1]
            
            # Compute log probabilities for each sequence
            for idx, (tokens, letter) in enumerate(zip(all_tokens, batch_letters)):
                seq_len = len(tokens)
                padding_len = max_len - seq_len
                
                total_logprob = 0.0
                for i in range(1, seq_len):
                    # Calculate position in padded sequence
                    pos = padding_len + i - 1
                    # Skip if position is beyond actual logits length
                    if pos >= actual_seq_len:
                        continue
                    token_logits = logits[idx, pos, :]
                    log_probs = F.log_softmax(token_logits, dim=-1)
                    token_id = tokens[i]
                    total_logprob += log_probs[token_id].item()
                
                logprobs[letter] = -total_logprob
        
        predicted_choice = min(logprobs, key=logprobs.get)
        predicted_choices.append(predicted_choice)
        truncated_list.append(any_truncated)
    
    return predicted_choices, truncated_list

# Dataset configuration and formatting functions
# ================================================

PUBMED_MAP = {
    'yes': 'A',
    'maybe': 'B',
    'no': 'C'
}

def format_mmlu_question(question):
    formatted_choices = ""
    for i, choice in enumerate(question['choices']):
        formatted_choices += f"Choice: {choice} = {chr(65 + i)}\n"
    
    return f"""
    Question: {question['question']}\n
    Choices:
    {formatted_choices}
    Answer:""", chr(65 + question['answer'])  # 0 -> A, 1 -> B, etc.

def format_medmcqa_question(question):
    formatted_choices = ""
    for i, option in enumerate(['opa', 'opb', 'opc', 'opd']):
        formatted_choices += f"Choice: {question[option]} = {chr(65 + i)}\n"
    
    return f"""
    Question: {question['question']}\n
    Choices:
    {formatted_choices}
    Answer:""", chr(65 + question['cop'])  # 0 -> A, 1 -> B, etc.

def format_medqa_question(question):
    formatted_choices = ""
    for letter in question['options']:
        formatted_choices += f"Choice: {question['options'][letter]} = {letter}\n"
    
    return f"""
    Question: {question['question']}\n
    Choices:
    {formatted_choices}
    Answer:""", question['answer_idx']

def format_pubmedqa_question(question):
    formatted_choices = ""
    for letter, choice in zip(['A', 'B', 'C'], ['yes', 'maybe', 'no']):
        formatted_choices += f"Choice: {choice} = {letter}\n"
    
    return f"""
    Context: {' '.join(question['context']['contexts'])}\n
    Question: {question['question']}\n
    Choices:
    {formatted_choices}
    Answer:""", PUBMED_MAP[question['final_decision']]

def format_headqa_question(question):
    formatted_choices = ""
    # Sort answers by aid to ensure consistent ordering
    sorted_answers = sorted(question['answers'], key=lambda x: x['aid'])
    for i, answer_dict in enumerate(sorted_answers):
        formatted_choices += f"Choice: {answer_dict['atext']} = {chr(65 + i)}\n"
    
    # Find which index corresponds to the correct answer id (ra)
    correct_idx = None
    for i, answer_dict in enumerate(sorted_answers):
        if answer_dict['aid'] == question['ra']:
            correct_idx = i
            break
    
    return f"""
    Question: {question['qtext']}\n
    Choices:
    {formatted_choices}
    Answer:""", chr(65 + correct_idx) if correct_idx is not None else 'A'

def format_medconceptsqa_question(question):
    formatted_choices = ""
    for i, option_key in enumerate(['option1', 'option2', 'option3', 'option4']):
        formatted_choices += f"Choice: {question[option_key]} = {chr(65 + i)}\n"
    
    return f"""
    Question: {question['question']}\n
    Choices:
    {formatted_choices}
    Answer:""", question['answer_id']

def format_jsonl_question(item):
    """For pre-formatted JSONL files (ADE, CMExam, MedQA-MCMLE)"""
    return item['input'], item['output']

# Generic dataset loading functions
# ==================================

def load_hf_dataset(val_dataset, test_dataset, format_func, n_fewshot, max_samples, seed=42):
    """Generic loader for HuggingFace datasets"""
    val_dataset = val_dataset.shuffle(seed=seed)
    val_dataset = val_dataset.select(range(min(len(val_dataset), n_fewshot)))
    
    test_dataset = test_dataset.shuffle(seed=seed)
    test_dataset = test_dataset.select(range(min(len(test_dataset), max_samples)))

    fewshot = {'input': [], 'output': []}
    dset = {'input': [], 'output': []}

    for row in val_dataset:
        inp, out = format_func(row)
        fewshot['input'].append(inp)
        fewshot['output'].append(out)

    for row in test_dataset:
        inp, out = format_func(row)
        dset['input'].append(inp)
        dset['output'].append(out)

    return dset, fewshot

def load_jsonl_dataset(data_path, format_func, n_fewshot, max_samples, seed=42, 
                       split_field=None, val_splits=None, test_splits=None):
    """
    Generic loader for local JSONL files
    
    Args:
        data_path: path to JSONL file
        format_func: function to format each item
        n_fewshot: number of few-shot examples
        max_samples: max test samples
        seed: random seed
        split_field: field name for split info (e.g., 'split')
        val_splits: list of split values for validation (e.g., ['validation', 'dev'])
        test_splits: list of split values for test (e.g., ['train', 'test'])
    """
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    np.random.seed(seed)
    
    # If split field is specified, split data accordingly
    if split_field and val_splits and test_splits:
        fewshot_data = [item for item in data if item.get(split_field) in val_splits]
        test_data = [item for item in data if item.get(split_field) in test_splits]
    else:
        # Otherwise, shuffle and split
        np.random.shuffle(data)
        fewshot_data = data[:n_fewshot]
        test_data = data[n_fewshot:]
    
    # Shuffle and limit
    np.random.shuffle(fewshot_data)
    fewshot_data = fewshot_data[:n_fewshot]
    
    np.random.shuffle(test_data)
    test_data = test_data[:max_samples]
    
    fewshot = {'input': [], 'output': []}
    dset = {'input': [], 'output': []}
    
    for item in fewshot_data:
        inp, out = format_func(item)
        fewshot['input'].append(inp)
        fewshot['output'].append(out)
    
    for item in test_data:
        inp, out = format_func(item)
        dset['input'].append(inp)
        dset['output'].append(out)
    
    return dset, fewshot