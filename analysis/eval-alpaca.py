"""
Evaluate chat model generations on alpaca using Claude API with multiple evaluation criteria
"""

import os
import sys
import argparse
import asyncio
import json
import logging
from pathlib import Path
from tqdm.asyncio import tqdm

# Suppress verbose logging from safetytooling
logging.getLogger('safetytooling').setLevel(logging.WARNING)

# Add safety-tooling to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
safety_tooling_path = os.path.join(parent_dir, "data", "safety-tooling")
sys.path.append(safety_tooling_path)

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

sys.path.append(parent_dir)
from paths import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--max_concurrents", type=int, default=50, help="max concurrent API calls")
parser.add_argument("--data_path", type=str, default="alpaca-outputs", help="path to input jsonl files")
parser.add_argument("--model_string", type=str, default="claude-sonnet-4-20250514", help="Claude model to use")
args = parser.parse_args()

# Define evaluation prompts
EVALUATION_PROMPTS = {
    "correctness": "is the provided response (in isolation) factually correct? It does not need to answer the question asked for this to be true.",
    "coherence": "is the language of the provided response fluent English? The response does not have to be logically sound or coherent for this to be true, it just needs to read like English words strung togehter in a normal-sounding sentence. If the response is cut off or incomplete, but the first part of the answer is otherwise in fluent English, answer yes.",
    "relevance": "does the provided response attempt to be at all relevant to the question asked?",
}

class Evaluator:
    def __init__(self, max_concurrents=20, model_string="claude-sonnet-4-20250514"):
        """Initialize the evaluator with safety-tooling API."""
        self.max_concurrents = max_concurrents
        self.semaphore = asyncio.Semaphore(max_concurrents)
        self.model_string = model_string

        # Setup environment for API keys
        utils.setup_environment()

        # Initialize the inference API
        self.api = InferenceAPI(
            cache_dir=Path(".cache"),
            anthropic_num_threads=max_concurrents
        )

        print(f"Initialized evaluator with model: {model_string}")

    async def evaluate_answer(self, question, answer, system_prompt):
        """Evaluate a single answer using Claude API."""
        async with self.semaphore:
            try:
                # Create the prompt
                prompt = Prompt(messages=[
                    ChatMessage(content=system_prompt, role=MessageRole.system),
                    ChatMessage(
                        content=f'Question: {question}\n\nAnswer: {answer}',
                        role=MessageRole.user
                    )
                ])

                # Make API call
                response = await self.api(
                    model_id=self.model_string,
                    prompt=prompt,
                    max_attempts_per_api_call=3,
                    n=1
                )

                # Parse response
                if response and len(response) > 0:
                    answer_text = response[0].completion.strip().upper()
                    is_yes = answer_text.startswith("YES")
                    return 1 if is_yes else 0
                else:
                    print(f"Warning: no response received")
                    return 0

            except Exception as e:
                print(f"Error evaluating answer: {e}")
                return 0

    async def evaluate_batch(self, items, prompt_key, system_prompt):
        """Evaluate a batch of items with a specific prompt."""
        print(f"Evaluating with prompt: {prompt_key}")

        # Create tasks for all items
        tasks = []
        for item in items:
            task = self.evaluate_answer(
                item['question'],
                item['answer'],
                system_prompt
            )
            tasks.append(task)

        # Process all concurrently with progress bar
        results = await tqdm.gather(*tasks, desc=f"Evaluating {prompt_key}")

        return results

async def main():
    # Input and output directories
    input_dir = os.path.join('results', args.data_path)
    output_dir = os.path.join('results', 'alpaca-graded')
    os.makedirs(output_dir, exist_ok=True)

    # Find all input jsonl files
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl') and 'collapse' not in f]

    if not input_files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(input_files)} files to process: {input_files}")

    # Initialize evaluator
    evaluator = Evaluator(
        max_concurrents=args.max_concurrents,
        model_string=args.model_string
    )

    # Process each file
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, input_file)

        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}")

        # Load the JSONL file
        items = []
        with open(input_path, 'r') as f:
            for line in f:
                items.append(json.loads(line))

        print(f"Loaded {len(items)} items")

        # Evaluate with each prompt
        for prompt_key, system_prompt in EVALUATION_PROMPTS.items():

            complete_system_prompt = f"You will be shown a question asked by a user, along with a provided response. Your task is to evaluate the response (not the question) by answering the following question: {system_prompt} Respond with YES or NO only."

            results = await evaluator.evaluate_batch(items, prompt_key, complete_system_prompt)

            # Add results to items
            for item, result in zip(items, results):
                item[prompt_key] = result

        # Save the graded results
        with open(output_path, 'w') as f:
            for item in items:
                f.write(json.dumps(item) + '\n')

        print(f"Saved graded results to {output_path}")

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(input_files)} files.")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
