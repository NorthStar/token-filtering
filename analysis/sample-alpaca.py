"""
Sample generations from chat-trained models on alpaca dataset
- Randomly select N questions from alpaca (N = # of healthsearchqa questions)
- Only use instruction-only examples (no input field)
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
parser.add_argument('--model_dir', type=str, default=os.path.join(MODEL_PATH, 'chat'), help='directory containing model checkpoints')
parser.add_argument('--temperature', type=float, default=0.8, help='sampling temperature')
parser.add_argument('--max_tokens', type=int, default=128, help='maximum tokens to generate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for generation')
parser.add_argument('--device', type=str, default='cuda', help='device to use')
parser.add_argument('--seed', type=int, default=42, help='random seed for sampling')
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else 'cpu'

# Load tokenizer
tokenizer = TikTokenizer('cl100k_base')
enc = tiktoken.get_encoding('cl100k_base')

# Load datasets
print("Loading datasets...")
healthsearchqa = load_dataset("katielink/healthsearchqa", "all_data")
alpaca = load_dataset("tatsu-lab/alpaca", split="train")

# Get N from healthsearchqa
n_questions = len(healthsearchqa['train'])
print(f"Number of healthsearchqa questions: {n_questions}")

# Filter alpaca to instruction-only examples (no input)
print("Filtering alpaca to instruction-only examples...")
alpaca_instruction_only = []
for example in alpaca:
    instruction = example['instruction']
    input_text = example['input']

    # Skip if no instruction or if there's an input
    if not instruction or not instruction.strip():
        continue
    if input_text and input_text.strip():
        continue

    alpaca_instruction_only.append({
        'question': instruction,
        'dataset': 'alpaca'
    })

print(f"Found {len(alpaca_instruction_only)} instruction-only alpaca examples")

# Randomly sample N questions
random.seed(args.seed)
if len(alpaca_instruction_only) < n_questions:
    print(f"Warning: Only {len(alpaca_instruction_only)} instruction-only examples available, using all")
    questions = alpaca_instruction_only
else:
    questions = random.sample(alpaca_instruction_only, n_questions)

print(f"Sampled {len(questions)} questions")

# Create output directory
output_dir = 'results/alpaca-outputs'
os.makedirs(output_dir, exist_ok=True)

# Models to iterate over
mask_types = ['mask', 'nomask', 'remove', 'document']

for mask_type in mask_types:
    model_name = f'{mask_type}-1816M-chat-sft.pt'
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

    # Generate responses
    output_path = os.path.join(output_dir, f'{mask_type}.jsonl')

    with open(output_path, 'w') as f:
        # Process in batches
        for batch_start in tqdm(range(0, len(questions), args.batch_size), desc="Generating"):
            batch_end = min(batch_start + args.batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]

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

print(f"\nDone! Processed {len(mask_types)} models on {len(questions)} questions.")
