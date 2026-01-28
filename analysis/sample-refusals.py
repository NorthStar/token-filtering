"""
Sample generations from refusal-trained models on held-out data:
- Other 50% of healthsearchqa (not used in training)
- Distinct subset of alpaca (not used in training)
"""

import os
import sys
import argparse
import json
import random
import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tiktokenizer import TikTokenizer
from model import GPTConfig, GPT
from paths import MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default=os.path.join(MODEL_PATH, 'refusal'), help='directory containing refusal-trained model checkpoints')
parser.add_argument('--temperature', type=float, default=0.8, help='sampling temperature')
parser.add_argument('--max_tokens', type=int, default=128, help='maximum tokens to generate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for generation')
parser.add_argument('--device', type=str, default='cuda', help='device to use')
parser.add_argument('--refusal_token', action='store_true')
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else 'cpu'

# Load tokenizer
tokenizer = TikTokenizer('cl100k_base')
cl100k_base = tiktoken.get_encoding('cl100k_base')

if args.refusal_token:
    enc = tiktoken.Encoding(
        name="cl100k_mask",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|mask|>": 100277,
            "<|refusal|>": 100278
        }
    )
else:
    enc = tiktoken.Encoding(
        name="cl100k_mask",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|mask|>": 100277
        }
    )

# Use same random seed as refusal-train.py for reproducible splits
random.seed(1337)

# Load healthsearchqa and get the OTHER 50% (held-out for evaluation)
print("Loading healthsearchqa dataset...")
healthsearchqa = load_dataset("katielink/healthsearchqa", "all_data")
healthsearchqa_data = list(healthsearchqa['train'])

# Same shuffle as in refusal-train.py
random.shuffle(healthsearchqa_data)
n_health = len(healthsearchqa_data) // 2

# Take the SECOND half (held-out for evaluation)
healthsearchqa_eval = healthsearchqa_data[n_health:]
print(f"Using {len(healthsearchqa_eval)} healthsearchqa examples (held-out 50%)")

# Load alpaca and get a distinct subset (not used in training)
print("Loading alpaca dataset...")
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
alpaca_data = list(alpaca)

# Same shuffle as in refusal-train.py
random.shuffle(alpaca_data)

# Training used first n_health items, so we use the NEXT n_health items
alpaca_eval = alpaca_data[n_health:2*n_health]
print(f"Using {len(alpaca_eval)} alpaca examples (distinct subset)")

# Collect all questions
questions = []

print("Processing healthsearchqa eval set...")
for example in healthsearchqa_eval:
    question = example['question']
    # Skip examples with None or empty question
    if not question or not question.strip():
        continue
    questions.append({
        'question': question,
        'dataset': 'healthsearchqa'
    })

print("Processing alpaca eval set...")
for example in alpaca_eval:
    instruction = example['instruction']
    input_text = example['input']
    
    # Skip examples with None or empty instruction
    if not instruction or not instruction.strip():
        continue
    
    # Format question: "<instruction>: <input>" if input not empty, else just instruction
    if input_text and input_text.strip():
        question = f"{instruction}: {input_text}"
    else:
        question = instruction
    
    questions.append({
        'question': question,
        'dataset': 'alpaca'
    })

print(f"Total questions: {len(questions)}")

# Create output directory
if args.refusal_token:
    output_dir = 'results/refusal-outputs-token'
else:
    output_dir = 'results/refusal-outputs'

os.makedirs(output_dir, exist_ok=True)

# Models to iterate over (refusal-trained versions)
mask_types = ['mask', 'nomask', 'remove', 'document']

for mask_type in mask_types:

    if args.refusal_token:
        model_name = f'{mask_type}-1816M-chat-sft-refusal-sft-token.pt'
    else:
        model_name = f'{mask_type}-1816M-chat-sft-refusal-sft.pt'
    
    model_path = os.path.join(args.model_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found, skipping...")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    # Remove '_orig_mod.' prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully ({model.get_num_params()/1e6:.1f}M parameters)")
    
    # Generate responses - separate files for each dataset
    for dataset_name in ['healthsearchqa', 'alpaca']:
        dataset_questions = [q for q in questions if q['dataset'] == dataset_name]
        output_path = os.path.join(output_dir, f'{mask_type}-{dataset_name}.jsonl')
        
        print(f"Generating for {dataset_name} ({len(dataset_questions)} questions)...")
        
        with open(output_path, 'w') as f:
            # Process in batches
            for batch_start in tqdm(range(0, len(dataset_questions), args.batch_size), desc=f"Generating {dataset_name}"):
                batch_end = min(batch_start + args.batch_size, len(dataset_questions))
                batch_questions = dataset_questions[batch_start:batch_end]
                
                # Prepare prompts
                prompts = []
                for item in batch_questions:
                    question = item['question']
                    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
                    prompts.append(prompt)
                
                # Tokenize all prompts
                prompt_tokens_list = [enc.encode(p, disallowed_special=()) for p in prompts]
                
                # Pad to same length for batching
                max_prompt_len = max(len(pt) for pt in prompt_tokens_list)
                padded_prompts = []
                prompt_lengths = []
                
                for pt in prompt_tokens_list:
                    prompt_lengths.append(len(pt))
                    # Pad on the left with 0s (typical for generation)
                    padded = [0] * (max_prompt_len - len(pt)) + pt
                    padded_prompts.append(padded)
                
                prompt_tensor = torch.tensor(padded_prompts, dtype=torch.long, device=device)
                
                # Generate
                with torch.no_grad():
                    generated = model.generate(
                        prompt_tensor,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=None,
                        beam_size=1
                    )
                
                # Decode and save each response
                for i, (item, prompt_len) in enumerate(zip(batch_questions, prompt_lengths)):
                    generated_tokens = generated[i].tolist()
                    
                    # Find where the actual prompt starts (after padding)
                    actual_start = max_prompt_len - prompt_len
                    # Extract tokens after the prompt
                    response_tokens = generated_tokens[max_prompt_len:]
                    
                    # Filter out invalid tokens (0 for padding, or outside vocab range)
                    # tiktoken vocab size for cl100k_base is ~100k
                    valid_tokens = [t for t in response_tokens if 0 < t < 100277]
                    
                    # Decode with error handling
                    try:
                        assistant_response = enc.decode(valid_tokens)
                    except Exception as e:
                        print(f"Warning: Error decoding tokens for item {batch_start + i}: {e}")
                        assistant_response = ""
                    
                    # Clean up - stop at end token if present
                    if "<|im_end|>" in assistant_response:
                        assistant_response = assistant_response.split("<|im_end|>")[0]
                    
                    # Write to JSONL
                    output_obj = {
                        "question": item['question'],
                        "answer": assistant_response,
                        "dataset": item['dataset']
                    }
                    f.write(json.dumps(output_obj) + '\n')
        
        print(f"Saved outputs to {output_path}")
    
    # Free memory
    del model
    del checkpoint
    torch.cuda.empty_cache()

print(f"\nDone! Processed models on {len(questions)} total questions.")
